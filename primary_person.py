"""Select the main person detection in a configurable central image region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PrimaryPerson:
    index: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    area: float


@dataclass(frozen=True)
class PrimaryPose:
    detection: PrimaryPerson
    keypoints_xy: np.ndarray
    keypoint_confidence: np.ndarray | None


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def select_primary_person(
    result_or_boxes: Any,
    frame_shape: Sequence[int],
    *,
    confidence_threshold: float = 0.20,
    person_class_id: int | None = 0,
    center_region: tuple[float, float, float, float] = (0.15, 0.10, 0.85, 0.92),
    min_area_ratio: float = 0.015,
    center_filter: Any = None,
    confidences: Any = None,
    class_ids: Any = None,
) -> PrimaryPerson | None:
    """Return the largest confident person whose box center is in the ROI.

    ``result_or_boxes`` can be an Ultralytics ``Results`` object or explicit
    ``xyxy`` boxes. For explicit boxes, pass ``confidences`` and ``class_ids``
    as extra keyword arguments. The region is normalized as (left, top, right,
    bottom). No fallback to edge detections is made when the region is empty.
    """
    if hasattr(result_or_boxes, "boxes"):
        boxes_obj = result_or_boxes.boxes
        if boxes_obj is None:
            return None
        raw_boxes = boxes_obj.xyxy
        confidences = confidences if confidences is not None else boxes_obj.conf
        class_ids = class_ids if class_ids is not None else boxes_obj.cls
    elif isinstance(result_or_boxes, dict):
        raw_boxes = result_or_boxes.get("xyxy", [])
        confidences = confidences if confidences is not None else result_or_boxes.get("confidences")
        class_ids = class_ids if class_ids is not None else result_or_boxes.get("class_ids")
    else:
        raw_boxes = result_or_boxes

    if confidences is None or class_ids is None:
        # Results objects always supply these. Bare boxes may instead use the
        # explicit wrapper form documented above.
        return None
    boxes = _as_numpy(raw_boxes).astype(np.float32, copy=False).reshape(-1, 4)
    scores = _as_numpy(confidences).astype(np.float32, copy=False).reshape(-1)
    classes = _as_numpy(class_ids).astype(np.int64, copy=False).reshape(-1)
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("frame_shape must contain positive height and width")
    left, top, right, bottom = center_region
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("center_region must be normalized bounds (left, top, right, bottom)")

    image_area = float(frame_width * frame_height)
    roi_x1, roi_y1 = left * frame_width, top * frame_height
    roi_x2, roi_y2 = right * frame_width, bottom * frame_height
    candidates: list[tuple[tuple[float, float, float], PrimaryPerson]] = []
    for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        if not np.isfinite(box).all() or not np.isfinite(score) or score < confidence_threshold:
            continue
        if person_class_id is not None and int(class_id) != person_class_id:
            continue
        x1 = float(np.clip(box[0], 0, frame_width))
        y1 = float(np.clip(box[1], 0, frame_height))
        x2 = float(np.clip(box[2], 0, frame_width))
        y2 = float(np.clip(box[3], 0, frame_height))
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        area = width * height
        if area / image_area < min_area_ratio:
            continue
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
            continue
        if center_filter is not None and not center_filter((center_x / frame_width, center_y / frame_height)):
            continue
        # Largest area wins. Confidence then distance to ROI center break ties.
        distance = ((center_x - (roi_x1 + roi_x2) / 2) / frame_width) ** 2 + \
                   ((center_y - (roi_y1 + roi_y2) / 2) / frame_height) ** 2
        selected = PrimaryPerson(index, (x1, y1, x2, y2), float(score), area)
        candidates.append(((area, float(score), -distance), selected))

    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def infer_primary_person_pose(
    frame: np.ndarray,
    person_detector: Any,
    pose_model: Any,
    *,
    detector_confidence: float = 0.10,
    pose_confidence: float = 0.15,
    selection_confidence: float = 0.20,
    person_class_id: int = 0,
    center_region: tuple[float, float, float, float] = (0.15, 0.10, 0.85, 0.92),
    center_filter: Any = None,
    crop_padding: float = 0.12,
) -> PrimaryPose | None:
    """Detect first, then run pose estimation only on the selected person's crop.

    This strict two-stage helper accepts separate Ultralytics detection and
    pose models. It returns keypoints in full-frame pixel coordinates.
    """
    detection_result = person_detector(
        frame, conf=detector_confidence, classes=[person_class_id], verbose=False
    )[0]
    detection = select_primary_person(
        detection_result,
        frame.shape,
        confidence_threshold=selection_confidence,
        person_class_id=person_class_id,
        center_region=center_region,
        center_filter=center_filter,
    )
    if detection is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    pad_x = max(8, int((x2 - x1) * crop_padding))
    pad_y = max(8, int((y2 - y1) * crop_padding))
    crop_x1 = max(0, int(x1) - pad_x)
    crop_y1 = max(0, int(y1) - pad_y)
    crop_x2 = min(frame_width, int(np.ceil(x2)) + pad_x)
    crop_y2 = min(frame_height, int(np.ceil(y2)) + pad_y)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    pose_result = pose_model(crop, conf=pose_confidence, verbose=False)[0]
    if pose_result.keypoints is None:
        return None
    refined = select_primary_person(
        pose_result,
        crop.shape,
        confidence_threshold=pose_confidence,
        person_class_id=None,
        center_region=(0.02, 0.02, 0.98, 0.98),
        min_area_ratio=0.01,
    )
    if refined is None or refined.index >= len(pose_result.keypoints.xy):
        return None

    keypoints = _as_numpy(pose_result.keypoints.xy[refined.index]).astype(np.float32, copy=True)
    keypoints[:, 0] += crop_x1
    keypoints[:, 1] += crop_y1
    confidence = None
    if pose_result.keypoints.conf is not None:
        confidence = _as_numpy(pose_result.keypoints.conf[refined.index]).astype(np.float32, copy=True)
    return PrimaryPose(detection, keypoints, confidence)

