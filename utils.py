import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, precision_recall_curve)
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# =========================
# 1. Load & Utility Functions
# =========================

def load_adbench_data(dataset_path):
    """
    Load dataset from a .npz file.
    Assumes the file contains:
    - 'X': Feature matrix (N, d)
    - 'y': Labels (N,)
    """
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def evaluate_with_classification_report_and_auc(model, test_loader, device, threshold=0.5):
    """
    Evaluate a model using classification report and AUC-ROC metric.
    Legacy interface — calls evaluate_full internally.
    """
    auroc, aupr, best_f1, report = evaluate_full(model, test_loader, device)
    return report, auroc


def evaluate_full(model, test_loader, device):
    """
    Full evaluation with AUC-ROC, AUPR, and Best F1 (threshold-optimized).
    
    Follows the evaluation protocol used by GenIAS, CARLA, and other TSAD papers:
    - AUC-ROC: Area under ROC curve
    - AUPR: Area under Precision-Recall curve (important for imbalanced data)
    - Best F1: Maximum F1 score across all possible thresholds
    
    Returns:
        auroc, aupr, best_f1, classification_report_str
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).squeeze()
            all_preds.append(y_pred.cpu())
            all_labels.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    if len(set(labels)) <= 1:
        print("Evaluation: Only one class present in labels — metrics undefined.")
        return None, None, None, None

    # 1. AUC-ROC
    auroc = roc_auc_score(labels, preds)

    # 2. AUPR (Average Precision Score)
    aupr = average_precision_score(labels, preds)

    # 3. Best F1 (optimal threshold search)
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_f1_idx = f1_scores.argmax()
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    # Classification report at best threshold
    binary_preds = (preds > best_threshold).astype(int)
    report = classification_report(labels, binary_preds, target_names=['Normal', 'Anomaly'])

    print(f"AUC-ROC:  {auroc:.4f}")
    print(f"AUPR:     {aupr:.4f}")
    print(f"Best F1:  {best_f1:.4f} (threshold={best_threshold:.4f})")
    print(report)

    return auroc, aupr, best_f1, report

def log_to_file(file_path, message):
    """Append a log message to the specified file."""
    with open(file_path, "a") as file:
        file.write(message + "\n")


# =========================
# 2. Loss Functions
# =========================

def beta_cvae_loss_fn(x, x_recon, mean, logvar, beta=4.0, sigma_prior=0.5):
    """
    Compute Beta-CVAE loss with Enhanced KL Divergence (inspired by GenIAS).
    
    Using sigma_prior < 1.0 enforces tighter latent representations of normal
    samples, improving separation between normal and anomalous data.
    Standard KL corresponds to sigma_prior=1.0.
    
    Args:
        x: Original input data.
        x_recon: Reconstructed data.
        mean: Mean of latent space distribution.
        logvar: Log variance of latent space distribution.
        beta: Weight for KL divergence.
        sigma_prior: Prior standard deviation (< 1.0 for tighter latent space).
    Returns:
        Total loss (scalar).
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # Enhanced KL with tunable prior variance
    sigma_prior_sq = sigma_prior ** 2
    kl_loss = -0.5 * torch.sum(
        1 + logvar
        - (mean ** 2) / sigma_prior_sq
        - logvar.exp() / sigma_prior_sq
        + 2 * torch.log(torch.tensor(sigma_prior, device=x.device))
    )
    return recon_loss + beta * kl_loss


# =========================
# 3. Training Functions
# =========================

def train_beta_cvae(model, data_loader, optimizer, device, sigma_prior=0.5):
    """
    Train Beta-CVAE model for one epoch.
    """
    model.train()
    total_loss = 0
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        x_recon, mean, logvar = model(x_batch, y_batch)
        loss = beta_cvae_loss_fn(x_batch, x_recon, mean, logvar,
                                  beta=model.beta, sigma_prior=sigma_prior)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def train_detector(model, train_loader, optimizer, criterion, device):
    """
    Train a detector model for one epoch.
    """
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def make_balanced_loader(D_train, y_train, batch_size=64):
    """
    Create a DataLoader with balanced sampling (equal normal/anomaly per batch).
    Uses WeightedRandomSampler to handle class imbalance.
    """
    class_counts = torch.bincount(y_train.long())
    weights = 1.0 / class_counts.float()
    sample_weights = weights[y_train.long()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)
    dataset = TensorDataset(D_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# =========================
# 4. Reward Functions
# =========================

torch.manual_seed(0)
np.random.seed(0)

def compute_entropy(D_train, x_syn, n_bins=50):
    """
    Estimate entropy increase after adding x_syn to D_train.
    Uses histogram-based approximation over feature dimensions.
    
    Args:
        D_train: Current training data (N, d).
        x_syn: New synthetic sample (1, d) or (d,).
    Returns:
        Entropy estimate (scalar tensor).
    """
    if x_syn.dim() == 1:
        x_syn = x_syn.unsqueeze(0)
    
    combined = torch.cat([D_train, x_syn], dim=0)
    entropy = 0.0
    for d in range(combined.shape[1]):
        col = combined[:, d]
        hist = torch.histc(col, bins=n_bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy += -(probs * torch.log(probs)).sum()
    
    return entropy / combined.shape[1]  # Average entropy across dimensions

def compute_reward(x_syn, detector, D_train, gamma_decay, episode, device):
    """
    Compute reward following Swift Hydra's formulation:
        R = gamma^episode * H(D_train ∪ x_syn) - log(W(x_syn))
    
    Early episodes: gamma^episode ≈ 1, emphasis on diversity (entropy).
    Later episodes: gamma^episode → 0, emphasis on deceiving detector.
    
    Args:
        x_syn: Generated synthetic samples (N, d).
        detector: Trained detector model.
        D_train: Current training data.
        gamma_decay: Decay factor for entropy weight (0 < gamma < 1).
        episode: Current episode number.
        device: Computation device.
    Returns:
        Reward tensor (N,).
    """
    detector.eval()
    x_syn = x_syn.to(device)
    
    with torch.no_grad():
        detect_prob = detector(x_syn).view(-1)
    
    # Entropy term (batched approximation)
    D_train_dev = D_train.to(device)
    entropy_rewards = []
    for i in range(x_syn.shape[0]):
        ent = compute_entropy(D_train_dev, x_syn[i])
        entropy_rewards.append(ent)
    entropy_term = torch.stack(entropy_rewards).to(device)
    
    # Combined reward with gamma decay
    gamma_weight = gamma_decay ** episode
    reward = gamma_weight * entropy_term - torch.log(detect_prob + 1e-8)
    
    return reward


# =========================
# 5. Utility Functions
# =========================

def to_tensor(x, device="cpu", dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)


def One_Step_To_Feasible_Action(
        beta_cvae,
        detector,
        x_orig,
        device,
        previously_generated=None,
        alpha=1.0,
        lambda_div=0.1,
        lr=0.001,
        steps=50,
        log_file=None
):
    """
    Generate adversarial samples by gradient descent in latent space.
    Used during warmup episodes before PPO takes over.
    
    Optimizes: min prob_class1 + lambda_div * similarity_to_previous
    (i.e., generate samples that fool detector AND are diverse)
    """
    beta_cvae.eval()
    detector.eval()

    if previously_generated is None:
        previously_generated = []

    x_orig = x_orig.to(device).unsqueeze(0)
    y_class1 = torch.full((1, 1), 1.0, device=device)  # Use 1.0 for consistent conditioning

    # Encode input data into latent space
    with torch.no_grad():
        mean, logvar = beta_cvae.encode(x_orig, y_class1)
        z = beta_cvae.reparameterize(mean, logvar).detach().clone()
    
    z.requires_grad_(True)

    # Optimize latent space representation
    optimizer_z = torch.optim.Adam([z], lr=lr)
    for step in range(steps):
        optimizer_z.zero_grad()

        x_synthetic = beta_cvae.decode(z, y_class1)
        prob_class1 = detector(x_synthetic)

        # Diversity term
        if previously_generated:
            x_old_cat = torch.stack(previously_generated, dim=0).to(device)
            dist = torch.norm(x_synthetic - x_old_cat, p=2, dim=1)
            diversity_term = torch.exp(-alpha * dist).sum()
        else:
            diversity_term = torch.tensor(0.0, device=device)

        # Loss = detection probability + similarity penalty (minimize both)
        loss = prob_class1.mean() + lambda_div * diversity_term
        loss.backward()
        optimizer_z.step()

    deceive_reward = 1.0 / (prob_class1.item() + 1e-4)
    div_val = diversity_term.item() if torch.is_tensor(diversity_term) else diversity_term
    div_reward = 1.0 / (div_val + 1e-4)
    print(f"Deceiving Detector Reward: {deceive_reward:.4f}",
          f"Diversity reward: {div_reward:.4f}",
          f"Loss: {loss.item():.4f}")

    if log_file:
        log_to_file(log_file, f"deceive={deceive_reward:.4f} div={div_reward:.4f} loss={loss.item():.4f}")

    with torch.no_grad():
        x_adv = beta_cvae.decode(z, y_class1).detach().cpu().squeeze(0)
    return x_adv
