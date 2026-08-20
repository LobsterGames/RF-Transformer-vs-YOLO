import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "sample_data" / "npy_1d" / "train"
VAL_DIR = ROOT / "data" / "sample_data" / "npy_1d" / "val"
OUTPUT_DIR = ROOT / "models" / "transformer.pt"

TARGET_CLASSES = ['psk', 'qam']
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3

# =====================================================================
# 1. SPECIALIST DATASET CLASS (BALANCED SAMPLES PER CLASS)
# =====================================================================
class SpecialistRFDataset(Dataset):
    def __init__(self, data_dir, target_subclasses=['psk', 'qam']):
        self.data_dir = data_dir
        self.label_map = {name: i for i, name in enumerate(target_subclasses)}
        
        class_files = defaultdict(list)
        for f in os.listdir(data_dir):
            if f.endswith('.npy'):
                parts = f.split('_')
                if len(parts) >= 2 and parts[1] in self.label_map:
                    class_files[parts[1]].append((f, self.label_map[parts[1]]))
                    
        # Equalize sample count across all target classes
        min_count = min(len(files) for files in class_files.values()) if class_files else 0
        self.file_list = []
        for cls_name, files in class_files.items():
            self.file_list.extend(files[:min_count])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name, class_id = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        # Load the saved [2, 1024] numpy array
        iq_data = np.load(file_path)
        tensor_x = torch.from_numpy(iq_data).float()
        tensor_y = torch.tensor(class_id, dtype=torch.long)
        
        return tensor_x, tensor_y


# =====================================================================
# 2. DYNAMIC 4-CHANNEL FEATURE EXTRACTOR
# =====================================================================
class FeatureExtractor1D(nn.Module):
    """Dynamically converts [B, 2, N] raw I/Q tensors into [B, 4, N] tensors on GPU."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [Batch, 2, Sequence_Length]
        I = x[:, 0, :]
        Q = x[:, 1, :]
        
        # 1. Instantaneous Amplitude: sqrt(I^2 + Q^2)
        A = torch.sqrt(I**2 + Q**2 + 1e-12)
        
        # 2. Differential Phase (dphi/dt): angle(z[n] * conj(z[n-1]))
        z = torch.complex(I, Q)
        conj_prod = z[:, 1:] * torch.conj(z[:, :-1])
        dphi_raw = torch.angle(conj_prod) / 3.141592653589793  # Normalized to [-1, 1]
        
        # Pad first sample to maintain exact sequence length (e.g. 1024)
        dphi = torch.cat((dphi_raw[:, :1], dphi_raw), dim=1)
        
        # Stack into [Batch, 4, Sequence_Length] tensor
        return torch.stack((I, Q, A, dphi), dim=1)


# =====================================================================
# 3. 1D TRANSFORMER ARCHITECTURE WITH 4-CHANNEL CONV STEM
# =====================================================================
class ConvStem1D(nn.Module):
    """Extracts local temporal, amplitude & phase features prior to Transformer self-attention."""
    def __init__(self, in_channels=4, embed_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),  # -> [B, 32, 512]
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, embed_dim, kernel_size=5, stride=2, padding=2),    # -> [B, 64, 256]
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # -> [B, 64, 128]
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.stem(x)         # [Batch, Embed_Dim, Num_Patches]
        return x.transpose(1, 2) # [Batch, Num_Patches, Embed_Dim]


class UpgradedRFTransformer(nn.Module):
    def __init__(self, seq_len=1024, embed_dim=64, num_heads=4, depth=4, num_classes=2):
        super().__init__()
        # On-the-Fly 4-Channel Feature Extractor
        self.feature_extractor = FeatureExtractor1D()
        
        # 4-Channel ConvStem (I, Q, Amplitude, Differential Phase)
        self.stem = ConvStem1D(in_channels=4, embed_dim=embed_dim)
        
        # After 3 stride-2 convolutions, sequence length 1024 becomes 128
        num_patches = seq_len // 8  
        
        # Learnable [CLS] token and Position Encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        
        # Expand raw [B, 2, 1024] I/Q to [B, 4, 1024] (I, Q, Amplitude, dphi/dt)
        x = self.feature_extractor(x)
        
        x = self.stem(x) # [B, 128, 64]
        
        # Append [CLS] Token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, 64]
        x = torch.cat((cls_tokens, x), dim=1)         # [B, 129, 64]
        
        x = x + self.pos_embed
        x = self.transformer(x)
        
        # Extract represented features ONLY from [CLS] token output
        cls_out = x[:, 0, :]
        cls_out = self.norm(cls_out)
        logits = self.head(cls_out)
        return logits


# =====================================================================
# 4. TRAINING & EVALUATION LOOP
# =====================================================================
def train_transformer():
    
    print("--- Initializing Specialist RF Dataset ---")
    train_dataset = SpecialistRFDataset(TRAIN_DIR, target_subclasses=TARGET_CLASSES)
    val_dataset = SpecialistRFDataset(VAL_DIR, target_subclasses=TARGET_CLASSES)
    
    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(val_dataset)} validation samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Enhanced Specialist Transformer on device: {device}\n")
    
    # Instantiate 5-class Transformer with 4-channel feature expansion
    model = UpgradedRFTransformer(seq_len=1024, embed_dim=64, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            
        scheduler.step()
        train_loss = total_train_loss / total_train
        train_acc = (correct_train / total_train) * 100.0
        
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
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_classes': TARGET_CLASSES,
                'class_mapping': {i: name for i, name in enumerate(TARGET_CLASSES)}
            }, OUTPUT_DIR)
            #print(f" --> Saved new best checkpoint (Val Acc: {val_acc:.2f}%)")

    print(f"\n[SUCCESS] Training complete! Model weights saved to '{OUTPUT_DIR.stem}'")


if __name__ == "__main__":
    train_transformer()