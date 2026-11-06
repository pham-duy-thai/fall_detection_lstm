"""LSTM model for classifying normalized pose-keypoint sequences.

Expected input shape: (batch, frames, keypoints * coordinates).
For COCO/YOLOv8-pose with 17 (x, y) points, input_size is 34.
"""

from __future__ import annotations

import torch
from torch import nn


class FallDetectionLSTM(nn.Module):
    """Classify a pose sequence into FALL, BENDING_DOWN, or WALKING_STANDING.

    ``forward`` returns raw logits (no softmax), ready for CrossEntropyLoss.
    """

    CLASS_NAMES = ("FALL", "BENDING_DOWN", "WALKING_STANDING")

    def __init__(
        self,
        num_keypoints: int = 17,
        coordinates_per_keypoint: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()
        if num_keypoints < 1 or coordinates_per_keypoint < 2:
            raise ValueError("Expected at least one keypoint with x and y coordinates")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_keypoints = num_keypoints
        self.coordinates_per_keypoint = coordinates_per_keypoint
        self.input_size = num_keypoints * coordinates_per_keypoint
        self.class_names = self.CLASS_NAMES

        kind = rnn_type.upper()
        recurrent = nn.LSTM if kind == "LSTM" else nn.GRU if kind == "GRU" else None
        if recurrent is None:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
        self.rnn_type = kind
        self.recurrent = recurrent(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(self.CLASS_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 3-class logits for ``x`` shaped (B, T, K*2) or (B, T, K, 2)."""
        if x.ndim == 4:
            x = x.flatten(start_dim=2)
        if x.ndim != 3:
            raise ValueError("Input must have shape (B,T,K*2) or (B,T,K,2)")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features per frame, got {x.shape[-1]}")
        _, state = self.recurrent(x)
        hidden = state[0] if self.rnn_type == "LSTM" else state
        return self.classifier(self.dropout(hidden[-1]))


def normalize_keypoint_sequence(
    keypoints: torch.Tensor,
    image_width: float,
    image_height: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Translate/scale keypoints per frame to reduce dependence on location.

    ``keypoints`` has shape (T,K,2) or (B,T,K,2), in pixel coordinates.
    Each frame is centered at the midpoint of the visible person's bounding
    box, then x/y are divided by that box's width/height. The image dimensions
    are used first to convert pixels to a resolution-independent [0,1] space.
    This preserves relative pose and body proportions while removing absolute
    screen position. Use the same procedure for train, validation, and inference.
    """
    if keypoints.ndim not in (3, 4) or keypoints.shape[-1] != 2:
        raise ValueError("keypoints must have shape (T,K,2) or (B,T,K,2)")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    points = keypoints.to(dtype=torch.float32).clone()
    points[..., 0] /= image_width
    points[..., 1] /= image_height
    valid = torch.isfinite(points).all(dim=-1)
    # Treat missing/non-finite detector points as zero for bbox calculation;
    # callers should additionally provide visibility/confidence as features if
    # the detector has them.
    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    inf = torch.finfo(points.dtype).max
    min_x = points[..., 0].masked_fill(~valid, inf).amin(dim=-1, keepdim=True)
    max_x = points[..., 0].masked_fill(~valid, -inf).amax(dim=-1, keepdim=True)
    min_y = points[..., 1].masked_fill(~valid, inf).amin(dim=-1, keepdim=True)
    max_y = points[..., 1].masked_fill(~valid, -inf).amax(dim=-1, keepdim=True)
    # If an entire frame has no valid points, use a safe unit box.
    has_points = valid.any(dim=-1, keepdim=True)
    min_x = torch.where(has_points, min_x, torch.zeros_like(min_x))
    max_x = torch.where(has_points, max_x, torch.ones_like(max_x))
    min_y = torch.where(has_points, min_y, torch.zeros_like(min_y))
    max_y = torch.where(has_points, max_y, torch.ones_like(max_y))
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    width = (max_x - min_x).clamp_min(eps)
    height = (max_y - min_y).clamp_min(eps)
    points[..., 0] = (points[..., 0] - center_x) / width
    points[..., 1] = (points[..., 1] - center_y) / height
    points = points.masked_fill(~valid.unsqueeze(-1), 0.0)
    return points

