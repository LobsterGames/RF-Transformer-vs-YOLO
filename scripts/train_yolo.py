from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "sample_data" / "dataset.yaml"

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    # Train the model with exact hyperparameters from CLI
    results = model.train(
        data=DATA_PATH,
        epochs=10,
        imgsz=1024,
        batch=8,
        device="cpu",
        cache="ram",
        workers=20,
        conf=0.10,
        # Disable spatial & color augmentations (preserves RF spectrogram physics)
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        # Output directory structure
        project="yolo_runs",
        name="yolo_train",
    )