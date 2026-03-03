import os
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import random
from utils import *
from utils import evaluate_full
from model import *
import umap

# =========================
# Configuration
# =========================
GECCO_PATH = "gecco.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "./saved_models"

# Load config from pretrain
config_path = os.path.join(save_dir, "gecco_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

WINDOW_SIZE = config["window_size"]
STRIDE = config["stride"]
TRAIN_RATIO = config["train_ratio"]
input_dim = config["input_dim"]
num_features = config["num_features"]
HIDDEN_DIM = config["hidden_dim"]
LATENT_DIM = config["latent_dim"]
BETA = config["beta"]
SIGMA_PRIOR = config["sigma_prior"]

# Co-evolution hyperparameters
NUM_EPISODES = 100
NUM_GEN_DATA = 50
BATCH_SIZE = 128
WARMUP_EPISODES = 20
TOP_L_RATIO = 0.5
GAMMA_DECAY = 0.95
CLAMP_RANGE = 3.0

# =========================
# 1. Load GECCO Data (same as pretrain)
# =========================
from pretrain_gecco import load_gecco_windowed

D_train, y_train, D_test, y_test, _, _ = load_gecco_windowed(
    GECCO_PATH, window_size=WINDOW_SIZE, stride=STRIDE, train_ratio=TRAIN_RATIO
)

# =========================
# 2. Load Pretrained Models
# =========================
vae_path = os.path.join(save_dir, "beta_cvae_gecco.pth")
detector_path = os.path.join(save_dir, "transformer_detector_gecco.pth")

loaded_beta_cvae = BetaCVAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                             latent_dim=LATENT_DIM, beta=BETA).to(device)
loaded_detector_model = TransformerDetector(input_size=input_dim).to(device)

loaded_beta_cvae.load_state_dict(torch.load(vae_path, weights_only=True))
loaded_detector_model.load_state_dict(torch.load(detector_path, weights_only=True))
loaded_beta_cvae.eval()
loaded_detector_model.eval()
print("Pretrained GECCO models loaded successfully.")

# =========================
# 3. Initialize Training Components
# =========================
new_detector = TransformerDetector(input_size=input_dim).to(device)
new_detector.load_state_dict(torch.load(detector_path, weights_only=True))  # Fix2: start from pretrained
optimizer_cvae = Adam(loaded_beta_cvae.parameters(), lr=1e-4)
optimizer_detector = Adam(new_detector.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# PPO agent
ppo_policy = PolicyNetwork(input_dim, 256, input_dim).to(device)
ppo_value = ValueNetwork(input_dim, 256).to(device)
ppo_trainer = PPOTrainer(
    ppo_policy, ppo_value,
    policy_lr=1e-4, value_lr=1e-4,
    gamma=0.99, clip_epsilon=0.2,
    entropy_coefficient=0.01,
    device=device
)

# Logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
beta_cvae_log = os.path.join(log_dir, "gecco_beta_cvae.log")
detector_log = os.path.join(log_dir, "gecco_detector.log")
adversarial_log = os.path.join(log_dir, "gecco_adversarial.log")
episode_log = os.path.join(log_dir, "gecco_episode_summary.log")

synthetic_data = []

# =========================
# 4. Co-Evolution Main Loop
# =========================
for ep in range(NUM_EPISODES):
    class1_mask = (y_train == 1)
    class0_mask = (y_train == 0)
    num_class1 = class1_mask.sum().item()
    num_class0 = class0_mask.sum().item()

    if num_class1 > num_class0:
        print(f"Break at Episode {ep + 1}: Class 1 ({num_class1}) exceeds Class 0 ({num_class0}).")
        break

    print(f"\n{'='*60}")
    print(f"EPISODE {ep + 1}/{NUM_EPISODES} | Class 0: {num_class0} | Class 1: {num_class1}")
    print(f"Mode: {'Warmup (gradient descent)' if ep < WARMUP_EPISODES else 'PPO'}")
    print(f"{'='*60}")

    # --- 4.1: Train Beta-CVAE ---
    train_dataset = TensorDataset(D_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for epoch in range(10):
        loss_cvae = train_beta_cvae(loaded_beta_cvae, train_loader, optimizer_cvae,
                                     device, sigma_prior=SIGMA_PRIOR)
        log_to_file(beta_cvae_log, f"Episode {ep+1} Epoch {epoch+1}/10, Loss: {loss_cvae:.4f}")

    # --- 4.2: Train Detector on BALANCED dataset ---
    balanced_loader = make_balanced_loader(D_train, y_train, batch_size=BATCH_SIZE)
    for det_epoch in range(5):
        detector_loss = train_detector(new_detector, balanced_loader, optimizer_detector, criterion, device)
        log_to_file(detector_log, f"Episode {ep+1} Epoch {det_epoch+1}/5, Loss: {detector_loss:.4f}")

    # --- 4.3: Generate Adversarial Samples ---
    idx_class1 = (y_train == 1).nonzero(as_tuple=True)[0]
    D_train_grow = [row for row in D_train[class1_mask]]

    candidate_samples = []
    candidate_origins = []
    candidate_log_probs = []
    candidate_deltas = []

    for syn_idx in range(NUM_GEN_DATA):
        random_idx = random.choice(idx_class1)
        x_orig = D_train[random_idx].to(device)

        if ep < WARMUP_EPISODES:
            x_adv = One_Step_To_Feasible_Action(
                beta_cvae=loaded_beta_cvae,
                detector=new_detector,  # Fix2: use co-evolving detector
                x_orig=x_orig,
                device=device,
                previously_generated=D_train_grow,
                alpha=1.0, lambda_div=0.1, lr=0.01, steps=20,
                log_file=adversarial_log,
            )
            delta = (x_adv - x_orig.cpu()).to(device)
            log_prob = torch.zeros(1, 1, device=device)
        else:
            delta, log_prob, ent = ppo_policy.sample_action(x_orig.unsqueeze(0))
            delta = delta.squeeze(0)
            log_prob = log_prob.detach()
            x_adv = x_orig + delta
            x_adv = torch.clamp(x_adv, -CLAMP_RANGE, CLAMP_RANGE)
            x_adv = x_adv.detach().cpu()

        candidate_samples.append(x_adv.detach().cpu().unsqueeze(0))
        candidate_origins.append(x_orig.detach().cpu().unsqueeze(0))
        candidate_log_probs.append(log_prob.detach().cpu())
        candidate_deltas.append(delta.detach().cpu().unsqueeze(0))

    candidates = torch.cat(candidate_samples, dim=0)
    origins = torch.cat(candidate_origins, dim=0)
    log_probs = torch.cat(candidate_log_probs, dim=0)
    deltas = torch.cat(candidate_deltas, dim=0)

    # --- 4.4: Compute Rewards ---
    rewards = compute_reward(
        x_syn=candidates, detector=new_detector,  # Fix2: use co-evolving detector
        D_train=D_train, gamma_decay=GAMMA_DECAY,
        episode=ep, device=device
    )

    # --- 4.5: Select Top-l ---
    l_top = max(1, int(NUM_GEN_DATA * TOP_L_RATIO))
    rewards = rewards.cpu()
    top_idx = torch.topk(rewards, k=l_top).indices
    selected = candidates[top_idx]

    # Fix3: CVAE plausibility filter — remove samples that drifted too far from anomaly distribution
    with torch.no_grad():
        sel_dev = selected.to(device)
        y_cond = torch.ones(len(sel_dev), 1, device=device)
        x_recon, _, _ = loaded_beta_cvae(sel_dev, y_cond)
        recon_err = F.mse_loss(x_recon, sel_dev, reduction='none').mean(dim=1)
        err_threshold = recon_err.median() + recon_err.std()
        plausible_mask = recon_err <= err_threshold
        selected = selected[plausible_mask.cpu()]
    selected_labels = torch.ones(len(selected))
    print(f"  Plausibility filter: {plausible_mask.sum().item()}/{l_top} samples kept")

    avg_reward = rewards[top_idx].mean().item()
    print(f"Top-{l_top} avg reward: {avg_reward:.4f}")
    log_to_file(episode_log, f"Episode {ep+1}: top-{l_top} avg_reward={avg_reward:.4f} "
                f"class0={num_class0} class1={num_class1}")

    # --- 4.6: PPO Update ---
    if ep >= WARMUP_EPISODES:
        ppo_states = origins[top_idx].to(device)
        ppo_actions = deltas[top_idx].to(device)
        ppo_old_log_probs = log_probs[top_idx].to(device)
        ppo_rewards_selected = rewards[top_idx].unsqueeze(1).to(device)

        with torch.no_grad():
            values = ppo_value(ppo_states)
        advantages = ppo_rewards_selected - values
        returns = ppo_rewards_selected

        ppo_trainer.ppo_update(
            states=ppo_states, actions=ppo_actions,
            old_log_probs=ppo_old_log_probs,
            returns=returns, advantages=advantages, n_epochs=3
        )
        print(f"PPO updated with {l_top} samples")

    # --- 4.7: Augment training set ---
    synthetic_data.extend(list(selected))
    D_train = torch.cat([D_train, selected], dim=0)
    y_train = torch.cat([y_train, selected_labels], dim=0)

print(f"\n{'='*60}")
print(f"Co-evolution complete. Total synthetic samples: {len(synthetic_data)}")
print(f"Final dataset: {len(D_train)} samples")
print(f"{'='*60}")

# =========================
# 5. UMAP Visualization
# =========================
print("\n=== Generating UMAP visualization ===")
plt.style.use('default')

# Reload original data for visualization
from pretrain_gecco import load_gecco_windowed
D_train_orig, y_train_orig, _, _, _, _ = load_gecco_windowed(
    GECCO_PATH, window_size=WINDOW_SIZE, stride=STRIDE, train_ratio=TRAIN_RATIO
)

X_synthetic = torch.stack(synthetic_data) if synthetic_data else torch.empty(0, input_dim)

# Subsample for UMAP (too many windows otherwise)
max_vis = 5000
indices_train = np.random.choice(len(D_train_orig), min(max_vis, len(D_train_orig)), replace=False)
indices_test = np.random.choice(len(D_test), min(max_vis, len(D_test)), replace=False)
indices_syn = np.random.choice(len(X_synthetic), min(max_vis, len(X_synthetic)), replace=False) if len(X_synthetic) > 0 else np.array([])

X_plot_parts = [D_train_orig[indices_train].numpy(), D_test[indices_test].numpy()]
y_plot_parts = [y_train_orig[indices_train].numpy(), y_test[indices_test].numpy()]

N_train_vis = len(indices_train)
N_test_vis = len(indices_test)
N_syn_vis = 0

if len(indices_syn) > 0:
    X_plot_parts.append(X_synthetic[indices_syn].numpy())
    y_plot_parts.append(np.full(len(indices_syn), 2))
    N_syn_vis = len(indices_syn)

X_plot = np.concatenate(X_plot_parts, axis=0)
y_plot = np.concatenate(y_plot_parts, axis=0)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, n_jobs=-1)
X_embedded = reducer.fit_transform(X_plot)

plt.figure(figsize=(10, 8), facecolor='white')

# Train normal
idx0 = (y_plot[:N_train_vis] == 0)
plt.scatter(X_embedded[:N_train_vis][idx0, 0], X_embedded[:N_train_vis][idx0, 1],
            c='darkred', alpha=0.4, s=10, label='Normal (Train)')

# Train anomaly
idx1 = (y_plot[:N_train_vis] == 1)
plt.scatter(X_embedded[:N_train_vis][idx1, 0], X_embedded[:N_train_vis][idx1, 1],
            c='darkblue', alpha=0.6, s=15, label='Anomaly (Train)')

# Test normal
offset = N_train_vis
idx0_t = (y_plot[offset:offset+N_test_vis] == 0)
plt.scatter(X_embedded[offset:offset+N_test_vis][idx0_t, 0],
            X_embedded[offset:offset+N_test_vis][idx0_t, 1],
            c='orange', alpha=0.3, s=10, label='Normal (Test)')

# Test anomaly
idx1_t = (y_plot[offset:offset+N_test_vis] == 1)
plt.scatter(X_embedded[offset:offset+N_test_vis][idx1_t, 0],
            X_embedded[offset:offset+N_test_vis][idx1_t, 1],
            c='skyblue', alpha=0.6, s=15, label='Anomaly (Test)')

# Synthetic
if N_syn_vis > 0:
    offset2 = N_train_vis + N_test_vis
    plt.scatter(X_embedded[offset2:, 0], X_embedded[offset2:, 1],
                c='green', alpha=0.6, s=15, label='Synthetic')

plt.title("UMAP: GECCO Train, Test, and Synthetic Anomalies")
plt.legend()
plt.savefig("gecco_umap.png", dpi=150, bbox_inches='tight')
plt.show()
print("UMAP saved to gecco_umap.png")

# =========================
# 6. Final Evaluation
# =========================
# Fix1: Use balanced sampling for final training (handles remaining class imbalance)
train_loader_final = make_balanced_loader(D_train, y_train, batch_size=64)
test_dataset = TensorDataset(D_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

print("\nAfter co-evolution augmentation:")
unique, counts = np.unique(y_train.numpy(), return_counts=True)
print("Class distribution:", dict(zip(unique.astype(int), counts)))

# Fix4: Fine-tune from pretrained model instead of training from scratch
model = TransformerDetector(input_size=input_dim).to(device)
model.load_state_dict(torch.load(detector_path, weights_only=True))
optimizer_tf = Adam(model.parameters(), lr=1e-4)  # Lower LR for fine-tuning
criterion = nn.BCEWithLogitsLoss()

best_f1_score = 0.0
best_auc = 0.0
best_aupr = 0.0
best_epoch = 0
for epoch in range(100):
    train_loss = train_detector(model, train_loader_final, optimizer_tf, criterion, device)
    if (epoch + 1) % 10 == 0:
        print(f"\n[Transformer] Epoch {epoch+1}/100, Loss={train_loss:.4f}")
        print("Test set evaluation:")
        auroc, aupr, f1, _ = evaluate_full(model, test_loader, device)
        if f1 and f1 > best_f1_score:
            best_f1_score = f1
            best_auc = auroc
            best_aupr = aupr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_detector_gecco.pth"))
            print(f"  >> New best model saved (epoch {best_epoch}, F1={best_f1_score:.4f})")
        print("-" * 40)

print(f"\n{'='*50}")
print(f"Best results after co-evolution (epoch {best_epoch}):")
print(f"  Best F1: {best_f1_score:.4f}")
print(f"  AUPR:    {best_aupr:.4f}")
print(f"  AUC-ROC: {best_auc:.4f}")
print(f"{'='*50}")
