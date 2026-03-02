import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
from utils import *
from model import *
import umap

# =========================
# Configuration
# =========================
dataset_path = r"ADBench_datasets/7_Cardiotocography.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "./saved_models"
vae_path = os.path.join(save_dir, "beta_cvae.pth")
detector_path = os.path.join(save_dir, "transformer_detector.pth")

# Hyperparameters
NUM_EPISODES = 100
NUM_GEN_DATA = 50
BATCH_SIZE = 128
WARMUP_EPISODES = 20
TOP_L_RATIO = 0.5          # Keep top 50% of generated samples
SIGMA_PRIOR = 0.5          # Enhanced KL prior (< 1.0 for tighter latent space)
GAMMA_DECAY = 0.95         # Reward gamma decay: early=diversity, late=deceive
CLAMP_RANGE = 3.0          # Clamp generated samples to normalized range

# =========================
# 1. Load Data
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

# =========================
# 2. Load Pretrained Models
# =========================
loaded_beta_cvae = BetaCVAE(input_dim=input_dim, hidden_dim=512, latent_dim=64, beta=1.0).to(device)
loaded_detector_model = TransformerDetector(input_size=input_dim).to(device)

loaded_beta_cvae.load_state_dict(torch.load(vae_path, weights_only=True))
loaded_detector_model.load_state_dict(torch.load(detector_path, weights_only=True))
loaded_beta_cvae.eval()
loaded_detector_model.eval()
print("Pretrained models loaded successfully.")

# =========================
# 3. Initialize Training Components
# =========================
# New detector for co-evolution (separate from pretrained one used for reward)
new_detector = TransformerDetector(input_size=input_dim).to(device)
optimizer_cvae = Adam(loaded_beta_cvae.parameters(), lr=1e-4)
optimizer_detector = Adam(new_detector.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# PPO agent (activates after warmup)
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
beta_cvae_log = os.path.join(log_dir, "beta_cvae.log")
detector_log = os.path.join(log_dir, "detector.log")
adversarial_log = os.path.join(log_dir, "adversarial_samples.log")
episode_log = os.path.join(log_dir, "episode_summary.log")

synthetic_data = []

# =========================
# 4. Co-Evolution Main Loop
# =========================
for ep in range(NUM_EPISODES):
    class1_mask = (y_train == 1)
    class0_mask = (y_train == 0)
    num_class1 = class1_mask.sum().item()
    num_class0 = class0_mask.sum().item()

    # Stop if anomaly class exceeds normal class
    if num_class1 > num_class0:
        print(f"Break at Episode {ep + 1}: Class 1 ({num_class1}) exceeds Class 0 ({num_class0}).")
        break

    print(f"\n{'='*60}")
    print(f"EPISODE {ep + 1}/{NUM_EPISODES} | Class 0: {num_class0} | Class 1: {num_class1}")
    print(f"Mode: {'Warmup (gradient descent)' if ep < WARMUP_EPISODES else 'PPO'}")
    print(f"{'='*60}")

    # --- 4.1: Train Beta-CVAE on current D_train ---
    train_dataset = TensorDataset(D_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    num_epochs_cvae = 10
    for epoch in range(num_epochs_cvae):
        loss_cvae = train_beta_cvae(loaded_beta_cvae, train_loader, optimizer_cvae,
                                     device, sigma_prior=SIGMA_PRIOR)
        log_to_file(beta_cvae_log, f"Episode {ep+1} Epoch {epoch+1}/{num_epochs_cvae}, Loss: {loss_cvae:.4f}")

    # --- 4.2: Train Detector on BALANCED dataset ---
    balanced_loader = make_balanced_loader(D_train, y_train, batch_size=BATCH_SIZE)
    detector_epochs = 5
    for det_epoch in range(detector_epochs):
        detector_loss = train_detector(new_detector, balanced_loader, optimizer_detector, criterion, device)
        log_to_file(detector_log, f"Episode {ep+1} Epoch {det_epoch+1}/{detector_epochs}, Loss: {detector_loss:.4f}")

    # --- 4.3: Generate Adversarial Samples ---
    idx_class1 = (y_train == 1).nonzero(as_tuple=True)[0]
    D_train_grow = [row for row in D_train[class1_mask]]

    candidate_samples = []
    candidate_origins = []    # Track which x_orig produced each sample
    candidate_log_probs = []  # Track log_probs for PPO update
    candidate_deltas = []

    for syn_idx in range(NUM_GEN_DATA):
        random_idx = random.choice(idx_class1)
        x_orig = D_train[random_idx].to(device)

        if ep < WARMUP_EPISODES:
            # --- Warmup: gradient descent in latent space ---
            x_adv = One_Step_To_Feasible_Action(
                beta_cvae=loaded_beta_cvae,
                detector=loaded_detector_model,
                x_orig=x_orig,
                device=device,
                previously_generated=D_train_grow,
                alpha=1.0,
                lambda_div=0.1,
                lr=0.01,
                steps=20,
                log_file=adversarial_log,
            )
            delta = (x_adv - x_orig.cpu()).to(device)
            log_prob = torch.zeros(1, 1, device=device)  # No PPO log_prob in warmup
        else:
            # --- PPO: policy generates delta ---
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

    candidates = torch.cat(candidate_samples, dim=0)   # (NUM_GEN_DATA, input_dim)
    origins = torch.cat(candidate_origins, dim=0)       # (NUM_GEN_DATA, input_dim)
    log_probs = torch.cat(candidate_log_probs, dim=0)   # (NUM_GEN_DATA, 1)
    deltas = torch.cat(candidate_deltas, dim=0)         # (NUM_GEN_DATA, input_dim)

    # --- 4.4: Compute Rewards (Swift Hydra formulation) ---
    rewards = compute_reward(
        x_syn=candidates,
        detector=loaded_detector_model,
        D_train=D_train,
        gamma_decay=GAMMA_DECAY,
        episode=ep,
        device=device
    )

    # --- 4.5: Select Top-l samples by reward ---
    l_top = max(1, int(NUM_GEN_DATA * TOP_L_RATIO))
    top_idx = torch.topk(rewards, k=l_top).indices
    selected = candidates[top_idx]
    selected_labels = torch.ones(len(selected))

    avg_reward = rewards[top_idx].mean().item()
    print(f"Top-{l_top} avg reward: {avg_reward:.4f}")
    log_to_file(episode_log, f"Episode {ep+1}: top-{l_top} avg_reward={avg_reward:.4f} "
                f"class0={num_class0} class1={num_class1}")

    # --- 4.6: PPO Update (after warmup) ---
    if ep >= WARMUP_EPISODES:
        ppo_states = origins[top_idx].to(device)
        ppo_actions = deltas[top_idx].to(device)
        ppo_old_log_probs = log_probs[top_idx].to(device)
        ppo_rewards_selected = rewards[top_idx].unsqueeze(1).to(device)

        # Use reward as return (single-step MDP)
        # Advantage = reward - value baseline
        with torch.no_grad():
            values = ppo_value(ppo_states)
        advantages = ppo_rewards_selected - values
        returns = ppo_rewards_selected

        ppo_trainer.ppo_update(
            states=ppo_states,
            actions=ppo_actions,
            old_log_probs=ppo_old_log_probs,
            returns=returns,
            advantages=advantages,
            n_epochs=3
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
# 5. Visualize Generated Data (UMAP)
# =========================
plt.style.use('default')

# Reset to original data for visualization
D_train_orig = torch.tensor(D_train_np, dtype=torch.float32)
y_train_orig = torch.tensor(y_train_np, dtype=torch.float32)

X_synthetic = torch.stack(synthetic_data) if synthetic_data else torch.empty(0, input_dim)

X_plot = torch.cat([D_train_orig, D_test, X_synthetic], dim=0).numpy()

N_train = len(D_train_orig)
N_test = len(D_test)
N_synthetic = len(X_synthetic)
y_plot = np.concatenate([
    y_train_orig.numpy(),
    y_test.numpy(),
    np.full((N_synthetic,), 2)
], axis=0)

scaler_plot = StandardScaler()
X_plot_scaled = scaler_plot.fit_transform(X_plot)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, n_jobs=-1)
X_embedded = reducer.fit_transform(X_plot_scaled)

plt.figure(figsize=(10, 8), facecolor='white')

idx0_train = (y_plot[:N_train] == 0)
plt.scatter(X_embedded[:N_train][idx0_train, 0], X_embedded[:N_train][idx0_train, 1],
            c='darkred', alpha=0.6, label='Class 0 (Train)')

idx1_train = (y_plot[:N_train] == 1)
plt.scatter(X_embedded[:N_train][idx1_train, 0], X_embedded[:N_train][idx1_train, 1],
            c='darkblue', alpha=0.6, label='Class 1 (Train)')

idx0_test = (y_plot[N_train:N_train + N_test] == 0)
plt.scatter(X_embedded[N_train:N_train + N_test][idx0_test, 0],
            X_embedded[N_train:N_train + N_test][idx0_test, 1],
            c='orange', alpha=0.6, label='Class 0 (Test)')

idx1_test = (y_plot[N_train:N_train + N_test] == 1)
plt.scatter(X_embedded[N_train:N_train + N_test][idx1_test, 0],
            X_embedded[N_train:N_train + N_test][idx1_test, 1],
            c='skyblue', alpha=0.6, label='Class 1 (Test)')

if N_synthetic > 0:
    idx_syn = (y_plot[N_train + N_test:] == 2)
    plt.scatter(X_embedded[N_train + N_test:][idx_syn, 0],
                X_embedded[N_train + N_test:][idx_syn, 1],
                c='green', alpha=0.6, label='Synthetic')

plt.title("UMAP Visualization: Train, Test, and Synthetic Data")
plt.legend()
plt.savefig("umap_visualization.png", dpi=150, bbox_inches='tight')
plt.show()
print("UMAP saved to umap_visualization.png")

# =========================
# 6. Final Evaluation
# =========================
train_dataset_final = TensorDataset(D_train, y_train)
test_dataset = TensorDataset(D_test, y_test)
train_loader_final = DataLoader(train_dataset_final, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print("\nAfter co-evolution augmentation:")
unique, counts = np.unique(y_train.numpy(), return_counts=True)
print("Class distribution:", dict(zip(unique.astype(int), counts)))

# Train final detector from scratch on augmented data
model = TransformerDetector(input_size=input_dim).to(device)
optimizer_tf = Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
num_epochs_tf = 100

best_auc = 0.0
for epoch in range(num_epochs_tf):
    train_loss = train_detector(model, train_loader_final, optimizer_tf, criterion, device)
    if (epoch + 1) % 10 == 0:
        print(f"\n[Transformer] Epoch {epoch + 1}/{num_epochs_tf}, Loss={train_loss:.4f}")
        print("Test set evaluation:")
        _, auc = evaluate_with_classification_report_and_auc(model, test_loader, device, threshold=0.3)
        if auc and auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_detector.pth"))
            print(f"New best AUC-ROC: {best_auc:.4f} (saved)")
        print("-" * 40)

print(f"\nBest AUC-ROC: {best_auc:.4f}")
