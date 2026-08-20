from ultralytics import YOLO
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "models" / "yolo.pt"
DATA_YAML = ROOT / "data" / "sample_data" / "dataset.yaml"  

def main():
    print(f"Loading YOLO weights from: {WEIGHTS_PATH}")
    
    model = YOLO(WEIGHTS_PATH)
    
    print("Starting validation...")
    
    metrics = model.val(
        data=DATA_YAML,
        split='val',      
        imgsz=1024,        
        batch=16,         
        device="cpu"          # 0 for GPU, 'cpu' for CPU
    )

if __name__ == '__main__':
    main()