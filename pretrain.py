import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
from model import *

# =========================
# Configuration
# =========================
dataset_path = r"ADBench_datasets/7_Cardiotocography.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA_PRIOR = 0.5  # Enhanced KL prior (matches SwiftHydra.py)

# =========================
# Step 1: Load and preprocess data
# =========================
X_all, y_all = load_adbench_data(dataset_path)
input_dim = X_all.shape[1]

scaler = StandardScaler()
X_all = torch.tensor(scaler.fit_transform(X_all)).float()

D_train_np, D_test_np, y_train_np, y_test_np = train_test_split(
    X_all.numpy(), y_all.numpy(), test_size=0.6, random_state=42, stratify=y_all
)
D_train = torch.tensor(D_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
D_test = torch.tensor(D_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

train_dataset = TensorDataset(D_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# =========================
# Step 2: Train Beta-CVAE with Enhanced KL
# =========================
beta_cvae = BetaCVAE(input_dim=input_dim, hidden_dim=512, latent_dim=64, beta=1.0).to(device)
optimizer_cvae = Adam(beta_cvae.parameters(), lr=1e-4)

num_epochs_cvae = 450
for epoch in range(num_epochs_cvae):
    loss_cvae = train_beta_cvae(beta_cvae, train_loader, optimizer_cvae, device,
                                 sigma_prior=SIGMA_PRIOR)
    if (epoch + 1) % 10 == 0:
        print(f"[Beta-CVAE] Epoch {epoch+1}/{num_epochs_cvae}, loss={loss_cvae:.2f}")

# =========================
# Step 3: Generate synthetic data for oversampling
# =========================
beta_cvae.eval()

minority_mask = (y_train == 1)
X_minority = D_train[minority_mask]
majority_mask = (y_train == 0)
X_majority = D_train[majority_mask]

num_generate = len(X_majority) - len(X_minority)

with torch.no_grad():
    z_uniform = (torch.rand(num_generate, beta_cvae.latent_dim) * 4.0) - 2.0
    z_uniform = z_uniform.to(device)
    # Use y=1.0 for consistent conditioning (not 0.9)
    y_synthetic_c = torch.full((num_generate, 1), 1.0, device=device)
    X_synthetic = beta_cvae.decode(z_uniform, y_synthetic_c)
    X_synthetic = X_synthetic.cpu()

y_synthetic_labels = torch.ones(num_generate)

D_train_final = torch.cat([D_train, X_synthetic], dim=0)
y_train_final = torch.cat([y_train, y_synthetic_labels], dim=0)

# =========================
# Step 4: Train TransformerDetector on augmented dataset
# =========================
train_dataset_final = TensorDataset(D_train_final, y_train_final)
test_dataset = TensorDataset(D_test, y_test)
train_loader_final = DataLoader(train_dataset_final, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print("After oversampling with Beta-CVAE:")
unique, counts = np.unique(y_train_final.numpy(), return_counts=True)
print("Class distribution in training set:", dict(zip(unique.astype(int), counts)))

model = TransformerDetector(input_size=input_dim).to(device)
optimizer_tf = Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

num_epochs_tf = 50
for epoch in range(num_epochs_tf):
    train_loss = train_detector(model, train_loader_final, optimizer_tf, criterion, device)
    if (epoch + 1) % 10 == 0:
        print(f"[Transformer] Epoch {epoch+1}/{num_epochs_tf}, Loss={train_loss:.4f}")
        print("Test set evaluation:")
        evaluate_with_classification_report_and_auc(model, test_loader, device)
        print("-" * 40)

# =========================
# Step 5: Save models
# =========================
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

vae_path = os.path.join(save_dir, "beta_cvae.pth")
torch.save(beta_cvae.state_dict(), vae_path)
print(f"Beta-CVAE model saved to: {vae_path}")

detector_path = os.path.join(save_dir, "transformer_detector.pth")
torch.save(model.state_dict(), detector_path)
print(f"TransformerDetector model saved to: {detector_path}")
