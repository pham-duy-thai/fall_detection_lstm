"""Kinematic post-processing for pose-based fall detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class PoseMetrics:
    center_y: float
    body_height: float
    box_ratio: float
    trunk_tilt_degrees: float
    downward_velocity_px_s: float
    downward_speed_body_heights_s: float
    downward_acceleration_body_heights_s2: float
    horizontal_duration_s: float


class FallPostProcessor:
    """Evaluate geometric posture and motion evidence over time.

    Velocity is positive when the torso center moves down in image coordinates.
    The threshold uses body-heights/second, so it is independent of resolution.
    A horizontal pose alone is not a fall. A fall requires a sustained horizontal
    pose plus a recent fast descent or meaningful joint movement after lying down.
    """

    def __init__(
        self,
        velocity_threshold: float = 0.80,
        aspect_ratio_threshold: float = 1.20,
        lying_duration_s: float = 1.5,
        fall_window_s: float = 3.0,
        bending_angle_degrees: float = 38.0,
        history_size: int = 30,
        velocity_window_s: float = 1.0,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.lying_duration_s = lying_duration_s
        self.fall_window_s = fall_window_s
        self.bending_angle_degrees = bending_angle_degrees
        self.history_size = max(3, int(history_size))
        self.velocity_window_s = max(0.1, float(velocity_window_s))
        self.reset()

    def reset(self) -> None:
        # Keep the actual keypoints as requested, along with measurements used
        # to estimate motion. maxlen bounds memory for long camera sessions.
        self.keypoint_history: deque[tuple[float, list[list[float]]]] = deque(maxlen=self.history_size)
        self.motion_history: deque[tuple[float, float, float]] = deque(maxlen=self.history_size)
        self.horizontal_since_s: float | None = None
        self.horizontal_motion_seen = False
        self.last_horizontal_motion = False
        self.last_recent_fast_drop = False
        self.last_fast_drop_s: float | None = None
        self.fall_latched = False
        self.last_metrics: PoseMetrics | None = None

    @staticmethod
    def _pose_metrics(keypoints: Iterable[Iterable[float]]) -> tuple[list[list[float]], float, float, float, float] | None:
        raw_points = [list(point) for point in keypoints]
        points = [list(map(float, point[:3])) for point in raw_points]
        if not points or any(not all(math.isfinite(v) for v in point[:2]) for point in points):
            return None
        points = [point for point in points if len(point) < 3 or point[2] >= 0.15]
        points = [point for point in points if not (abs(point[0]) < 1e-6 and abs(point[1]) < 1e-6)]
        if len(points) < 5:
            return None

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if height <= 1.0 or width <= 1.0:
            return None

        by_index = {}
        for index, point in enumerate(raw_points):
            if index >= 17:
                break
            values = list(map(float, point[:3]))
            if len(values) >= 2 and all(math.isfinite(v) for v in values[:2]):
                if len(values) < 3 or values[2] >= 0.15:
                    if not (abs(values[0]) < 1e-6 and abs(values[1]) < 1e-6):
                        by_index[index] = values[:2]
        shoulders = [by_index[i] for i in (5, 6) if i in by_index]
        hips = [by_index[i] for i in (11, 12) if i in by_index]
        if shoulders and hips:
            shoulder_center = [sum(p[axis] for p in shoulders) / len(shoulders) for axis in (0, 1)]
            hip_center = [sum(p[axis] for p in hips) / len(hips) for axis in (0, 1)]
            center_y = (shoulder_center[1] + hip_center[1]) / 2.0
            tilt = math.degrees(math.atan2(abs(hip_center[0] - shoulder_center[0]),
                                           abs(hip_center[1] - shoulder_center[1]) + 1e-6))
        else:
            center_y = sorted(ys)[len(ys) // 2]
            tilt = 0.0
        return points, center_y, height, width / height, tilt

    def _recent_vertical_motion(self, timestamp_s: float, body_height: float) -> tuple[float, float, float]:
        """Return Y speed, normalized speed and acceleration from recent history."""
        samples = [sample for sample in self.motion_history
                   if 0 <= timestamp_s - sample[0] <= self.velocity_window_s]
        if len(samples) < 2:
            return 0.0, 0.0, 0.0

        # Least-squares slope over the time window is less sensitive to one
        # noisy pose estimate than comparing adjacent frames.
        times = [sample[0] for sample in samples]
        ys = [sample[1] for sample in samples]
        mean_t = sum(times) / len(times)
        mean_y = sum(ys) / len(ys)
        denominator = sum((t - mean_t) ** 2 for t in times)
        velocity = (sum((t - mean_t) * (y - mean_y) for t, y in zip(times, ys)) / denominator
                    if denominator > 1e-9 else 0.0)
        normalized_velocity = max(0.0, velocity / max(1.0, body_height))

        # Compare slopes in the earlier and later halves to measure whether
        # downward motion is accelerating. This is retained for diagnostics;
        # the fall decision still requires a fast downward speed.
        midpoint = len(samples) // 2
        def slope(group: list[tuple[float, float, float]]) -> float:
            if len(group) < 2:
                return 0.0
            ts = [item[0] for item in group]
            values = [item[1] for item in group]
            mt = sum(ts) / len(ts)
            my = sum(values) / len(values)
            den = sum((t - mt) ** 2 for t in ts)
            return sum((t - mt) * (y - my) for t, y in zip(ts, values)) / den if den > 1e-9 else 0.0
        early_velocity = slope(samples[:midpoint + 1])
        late_velocity = slope(samples[midpoint:])
        half_window = max(0.05, (times[-1] - times[0]) / 2.0)
        acceleration = (late_velocity - early_velocity) / half_window / max(1.0, body_height)
        return velocity, normalized_velocity, acceleration

    def _keypoints_moving(self, current: list[list[float]], timestamp_s: float,
                          body_height: float) -> bool:
        """Detect meaningful joint movement over a short interval, ignoring pose jitter."""
        previous = next((sample for sample in reversed(self.keypoint_history)
                         if 0.25 <= timestamp_s - sample[0] <= 1.25), None)
        if previous is None:
            return False
        prior_points = previous[1]
        pairs = []
        for index, point in enumerate(current[:17]):
            if index >= len(prior_points):
                break
            old = prior_points[index]
            if len(point) < 2 or len(old) < 2:
                continue
            if (len(point) >= 3 and point[2] < 0.20) or (len(old) >= 3 and old[2] < 0.20):
                continue
            if not all(math.isfinite(float(value)) for value in (*point[:2], *old[:2])):
                continue
            if ((abs(float(point[0])) < 1e-6 and abs(float(point[1])) < 1e-6) or
                    (abs(float(old[0])) < 1e-6 and abs(float(old[1])) < 1e-6)):
                continue
            pairs.append((float(point[0]), float(point[1]), float(old[0]), float(old[1])))
        if len(pairs) < 3:
            return False
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        scale = max(1.0, body_height, extent)
        displacements = [math.hypot(x - old_x, y - old_y) / scale
                         for x, y, old_x, old_y in pairs]
        return sum(value >= 0.07 for value in displacements) >= 2 or max(displacements, default=0.0) >= 0.16

    def update(self, keypoints: Iterable[Iterable[float]] | None, timestamp_s: float,
               confidence: Iterable[float] | None = None) -> str:
        """Consume a frame and classify posture, motion, and fall state."""
        if keypoints is None:
            self.reset()
            return "NO_PERSON"
        keypoints = [list(point) for point in keypoints]
        if confidence is not None:
            scores = list(confidence)
            for index, point in enumerate(keypoints):
                if len(point) < 3 and index < len(scores):
                    point.append(float(scores[index]))

        measured = self._pose_metrics(keypoints)
        if measured is None:
            self.reset()
            return "NO_PERSON"
        points, center_y, body_height, box_ratio, trunk_tilt = measured

        # Drop stale samples after camera stalls or timestamp discontinuities.
        if self.motion_history and not (0 < timestamp_s - self.motion_history[-1][0] <= 2.0):
            self.keypoint_history.clear()
            self.motion_history.clear()
            self.horizontal_since_s = None
            self.horizontal_motion_seen = False
            self.last_fast_drop_s = None
            self.fall_latched = False
        motion_points = [list(map(float, point[:3])) for point in keypoints]
        keypoint_motion = self._keypoints_moving(motion_points, timestamp_s, body_height)
        self.keypoint_history.append((timestamp_s, motion_points))
        self.motion_history.append((timestamp_s, center_y, body_height))
        downward_velocity, normalized_downward_speed, normalized_acceleration = self._recent_vertical_motion(
            timestamp_s, body_height
        )
        if normalized_downward_speed >= self.velocity_threshold:
            self.last_fast_drop_s = timestamp_s

        is_horizontal = box_ratio >= self.aspect_ratio_threshold
        if is_horizontal:
            entered_horizontal = self.horizontal_since_s is None
            if self.horizontal_since_s is None:
                self.horizontal_since_s = timestamp_s
                self.horizontal_motion_seen = False
            # Ignore motion used to get into the horizontal posture. Only
            # activity after the person is already horizontal is corroboration.
            self.last_horizontal_motion = not entered_horizontal and keypoint_motion
            if self.last_horizontal_motion:
                self.horizontal_motion_seen = True
        else:
            self.horizontal_since_s = None
            self.horizontal_motion_seen = False
            self.last_horizontal_motion = False
            if trunk_tilt < self.bending_angle_degrees / 2:
                self.last_fast_drop_s = None
                self.fall_latched = False

        horizontal_duration = (timestamp_s - self.horizontal_since_s) if self.horizontal_since_s is not None else 0.0
        recent_fast_drop = self.last_fast_drop_s is not None and 0 <= timestamp_s - self.last_fast_drop_s <= self.fall_window_s
        self.last_recent_fast_drop = recent_fast_drop
        # Being horizontal by itself is not a fall. Confirm after a fast drop
        # or after meaningful joint activity while the person remains horizontal.
        confirmed_fall = (is_horizontal and horizontal_duration >= self.lying_duration_s and
                          (recent_fast_drop or self.horizontal_motion_seen))
        if confirmed_fall:
            self.fall_latched = True

        self.last_metrics = PoseMetrics(center_y, body_height, box_ratio, trunk_tilt,
                                        downward_velocity, normalized_downward_speed,
                                        normalized_acceleration,
                                        horizontal_duration)

        if self.fall_latched and (is_horizontal or trunk_tilt >= self.bending_angle_degrees / 2):
            return "FALL"
        if is_horizontal:
            return "LYING"
        if trunk_tilt >= self.bending_angle_degrees:
            return "BENDING"
        return "NORMAL"

    def hold_state(self) -> str:
        """Return the last classification without advancing temporal state."""
        if self.fall_latched:
            return "FALL"
        if self.last_metrics is None:
            return "INVALID_FRAME"
        if self.last_metrics.box_ratio >= self.aspect_ratio_threshold:
            return "LYING"
        if self.last_metrics.trunk_tilt_degrees >= self.bending_angle_degrees:
            return "BENDING"
        return "NORMAL"


def postprocess_sequence(
    frame_keypoints: Iterable[Iterable[Iterable[float]] | None],
    fps: float,
    **thresholds: float,
) -> list[str]:
    """Convenience API for a whole clip: one keypoint array (or None) per frame."""
    processor = FallPostProcessor(**thresholds)
    safe_fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    return [processor.update(frame, index / safe_fps) for index, frame in enumerate(frame_keypoints)]
