import os
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils import *
from utils import evaluate_full
from model import *

# =========================
# Configuration
# =========================
GECCO_PATH = "gecco.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

# Time series windowing
WINDOW_SIZE = 60        # 60 minutes = 1 hour window
STRIDE = 1              # Slide 1 step at a time
TRAIN_RATIO = 0.5       # Chronological split: first 50% train, last 50% test

# Model hyperparameters
HIDDEN_DIM = 512
LATENT_DIM = 64
BETA = 1.0
SIGMA_PRIOR = 0.5
NUM_EPOCHS_CVAE = 200
NUM_EPOCHS_DETECTOR = 50
BATCH_SIZE = 64
LR = 1e-4


# =========================
# 1. Load & Window GECCO Data
# =========================
def load_gecco_windowed(csv_path, window_size=60, stride=1, train_ratio=0.5):
    """
    Load GECCO water quality dataset and create sliding windows.
    
    - Chronological train/test split (no shuffle — preserves temporal order).
    - Window label: 1 (anomaly) if ANY point in the window is anomalous.
    - Features are standardized using TRAINING set statistics only.
    
    Args:
        csv_path: Path to gecco.csv
        window_size: Number of time steps per window
        stride: Sliding window step size
        train_ratio: Fraction of data for training (chronological split)
    
    Returns:
        D_train, y_train, D_test, y_test as torch tensors
        input_dim: flattened dimension (window_size * num_features)
        num_features: number of raw features (9)
    """
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    features = df.drop('label', axis=1).values  # (N, 9)
    num_features = features.shape[1]
    
    # Chronological split BEFORE windowing (prevents data leakage)
    split_idx = int(len(features) * train_ratio)
    
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    test_features = features[split_idx:]
    test_labels = labels[split_idx:]
    
    # Standardize using training set statistics
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)  # Use train stats!
    
    def create_windows(feats, labs, win_size, step):
        windows, window_labels = [], []
        for i in range(0, len(feats) - win_size + 1, step):
            w = feats[i:i + win_size]                    # (window_size, 9)
            # Anomaly if ANY point in window is anomalous
            l = 1.0 if labs[i:i + win_size].any() else 0.0
            windows.append(w)
            window_labels.append(l)
        X = np.array(windows)           # (num_windows, window_size, 9)
        y = np.array(window_labels)     # (num_windows,)
        return X, y
    
    X_train, y_train = create_windows(train_features, train_labels, window_size, stride)
    X_test, y_test = create_windows(test_features, test_labels, window_size, stride)
    
    print(f"=== GECCO Dataset Loaded ===")
    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Num features: {num_features}")
    print(f"Train windows: {len(X_train)} (anomaly: {y_train.sum():.0f}, "
          f"ratio: {y_train.mean()*100:.2f}%)")
    print(f"Test windows:  {len(X_test)} (anomaly: {y_test.sum():.0f}, "
          f"ratio: {y_test.mean()*100:.2f}%)")
    
    # Flatten windows for current model architecture: (N, window_size * 9)
    input_dim = window_size * num_features
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    D_train = torch.tensor(X_train_flat, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    D_test = torch.tensor(X_test_flat, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return D_train, y_train, D_test, y_test, input_dim, num_features


# =========================
# 2. Load Data
# =========================
D_train, y_train, D_test, y_test, input_dim, num_features = load_gecco_windowed(
    GECCO_PATH, window_size=WINDOW_SIZE, stride=STRIDE, train_ratio=TRAIN_RATIO
)

train_dataset = TensorDataset(D_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# =========================
# 3. Train Beta-CVAE with Enhanced KL
# =========================
print(f"\n=== Training Beta-CVAE (input_dim={input_dim}) ===")
beta_cvae = BetaCVAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                      latent_dim=LATENT_DIM, beta=BETA).to(device)
optimizer_cvae = Adam(beta_cvae.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS_CVAE):
    loss_cvae = train_beta_cvae(beta_cvae, train_loader, optimizer_cvae, device,
                                 sigma_prior=SIGMA_PRIOR)
    if (epoch + 1) % 10 == 0:
        print(f"[Beta-CVAE] Epoch {epoch+1}/{NUM_EPOCHS_CVAE}, loss={loss_cvae:.2f}")


# =========================
# 4. Generate Synthetic Anomalies for Oversampling
# =========================
beta_cvae.eval()

minority_mask = (y_train == 1)
majority_mask = (y_train == 0)
num_minority = minority_mask.sum().item()
num_majority = majority_mask.sum().item()
num_generate = num_majority - num_minority

print(f"\n=== Generating {num_generate} synthetic anomalies ===")
print(f"Class 0 (normal): {num_majority}, Class 1 (anomaly): {num_minority}")

with torch.no_grad():
    # Sample z from prior (use sigma_prior for tighter sampling)
    z_sample = torch.randn(num_generate, beta_cvae.latent_dim).to(device) * SIGMA_PRIOR
    y_synthetic_c = torch.full((num_generate, 1), 1.0, device=device)
    X_synthetic = beta_cvae.decode(z_sample, y_synthetic_c).cpu()

y_synthetic_labels = torch.ones(num_generate)

D_train_final = torch.cat([D_train, X_synthetic], dim=0)
y_train_final = torch.cat([y_train, y_synthetic_labels], dim=0)


# =========================
# 5. Train TransformerDetector on Augmented Data
# =========================
train_dataset_final = TensorDataset(D_train_final, y_train_final)
test_dataset = TensorDataset(D_test, y_test)
train_loader_final = DataLoader(train_dataset_final, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"\nAfter oversampling with Beta-CVAE:")
unique, counts = np.unique(y_train_final.numpy(), return_counts=True)
print("Class distribution:", dict(zip(unique.astype(int), counts)))

print(f"\n=== Training TransformerDetector ===")
model = TransformerDetector(input_size=input_dim).to(device)
optimizer_tf = Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

best_auc = 0.0
best_f1 = 0.0
best_aupr = 0.0
best_epoch = 0
for epoch in range(NUM_EPOCHS_DETECTOR):
    train_loss = train_detector(model, train_loader_final, optimizer_tf, criterion, device)
    if (epoch + 1) % 10 == 0:
        print(f"\n[Transformer] Epoch {epoch+1}/{NUM_EPOCHS_DETECTOR}, Loss={train_loss:.4f}")
        print("Test set evaluation:")
        auroc, aupr, f1, _ = evaluate_full(model, test_loader, device)
        if auroc and auroc > best_auc:
            best_auc = auroc
            best_f1 = f1
            best_aupr = aupr
            best_epoch = epoch + 1
            # Save best model
            best_detector_path = os.path.join(save_dir, "transformer_detector_gecco.pth")
            torch.save(model.state_dict(), best_detector_path)
            print(f"  >> New best model saved (epoch {best_epoch})")
        print("-" * 40)

print(f"\n{'='*50}")
print(f"Best results (epoch {best_epoch}):")
print(f"  AUC-ROC: {best_auc:.4f}")
print(f"  AUPR:    {best_aupr:.4f}")
print(f"  Best F1: {best_f1:.4f}")
print(f"{'='*50}")


# =========================
# 6. Save Models
# =========================
vae_path = os.path.join(save_dir, "beta_cvae_gecco.pth")
torch.save(beta_cvae.state_dict(), vae_path)
print(f"Beta-CVAE saved to: {vae_path}")

# Detector already saved at best epoch above
print(f"TransformerDetector (best) saved to: {best_detector_path}")

# Save config for SwiftHydra to use
config = {
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "train_ratio": TRAIN_RATIO,
    "input_dim": input_dim,
    "num_features": num_features,
    "hidden_dim": HIDDEN_DIM,
    "latent_dim": LATENT_DIM,
    "beta": BETA,
    "sigma_prior": SIGMA_PRIOR,
}
import json
config_path = os.path.join(save_dir, "gecco_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved to: {config_path}")
