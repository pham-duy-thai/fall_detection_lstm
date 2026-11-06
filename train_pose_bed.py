"""Fine-tune YOLOv8-Pose to detect the target lying-person class."""

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "pose_bed_data.yaml"
INITIAL_WEIGHTS = ROOT / "yolov8n-pose.pt"


def main() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")
    if not INITIAL_WEIGHTS.is_file():
        raise FileNotFoundError(f"Initial pose weights not found: {INITIAL_WEIGHTS}")

    model = YOLO(str(INITIAL_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        task="pose",
        epochs=100,
        patience=25,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        # +/-60 degrees includes near-horizontal orientations without forcing
        # every sample through extreme +/-90 degree rotations.
        degrees=60.0,
        fliplr=0.50,
        # Use only when upside-down views are plausible for the camera.
        flipud=0.10,
        translate=0.10,
        scale=0.35,
        # Moderate mosaic/mixup preserve more realistic bed and person geometry.
        mosaic=0.50,
        mixup=0.05,
        close_mosaic=10,
        project=str(ROOT / "runs" / "pose"),
        name="lying_person_pose_v1",
        plots=True,
        save=True,
    )


if __name__ == "__main__":
    main()
