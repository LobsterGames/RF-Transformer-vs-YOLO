import torch
from pathlib import Path
import torch.nn as nn
import sys

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT / "scripts"))

from train_transformer import UpgradedRFTransformer, SpecialistRFDataset, DataLoader


VAL_DIR = ROOT / "data" / "sample_data" / "npy_1d" / "val"

WEIGHTS_PATH = ROOT / "models" / "transformer.pt"

TARGET_CLASSES = ['psk', 'qam']
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3

def main():
    print("--- Initializing Specialist RF Dataset ---")
    val_dataset = SpecialistRFDataset(VAL_DIR, target_subclasses=TARGET_CLASSES)

    print(f"Loaded {len(val_dataset)} validation samples.")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate 5-class Transformer with 4-channel feature expansion
    model = UpgradedRFTransformer(seq_len=1024, embed_dim=64, num_classes=2).to(device)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

    state_dict = checkpoint['model_state_dict']

    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()



    # --- VALIDATION PHASE ---
    model.eval()
    total_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            total_val_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == y_batch).sum().item()
            total_val += y_batch.size(0)
            
    val_loss = total_val_loss / total_val
    val_acc = (correct_val / total_val) * 100.0

    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    main()