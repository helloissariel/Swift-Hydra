import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.optim import Adam

# =========================
# 2. Beta-CVAE
# =========================
class BetaCVAE(nn.Module):
    """
    Beta Conditional Variational Autoencoder (Beta-CVAE) for binary tasks:
      - Encoder processes (x, y).
      - Decoder processes (z, y).
      - Beta > 1 emphasizes the KL divergence to encourage a more spread-out latent space.
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, beta=4.0):
        super(BetaCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder layers
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim + 1, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(xy))
        h = F.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        logvar = self.fc3_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = F.relu(self.fc4(zy))
        h = F.relu(self.fc5(h))
        x_recon = self.fc6(h)
        return x_recon

    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mean, logvar


# =========================
# 3. Transformer Detector
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :].to(x.device)


class TransformerDetector(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerDetector, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(1)


# =========================
# 4. Mixture of Experts
# =========================
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, num_experts, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.1, gating_hidden_size=64):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            TransformerDetector(input_size=input_size, d_model=d_model,
                                nhead=nhead, num_layers=num_layers,
                                dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_experts)
        ])

        self.gating_network = nn.Sequential(
            nn.Linear(input_size, gating_hidden_size),
            nn.ReLU(),
            nn.Linear(gating_hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        gating_weights = self.gating_network(x)
        expert_outputs = torch.cat([expert(x).unsqueeze(1) for expert in self.experts], dim=1)
        output = torch.sum(expert_outputs * gating_weights, dim=1)
        return output


# =========================
# 5. PPO Components
# =========================
class PolicyNetwork(nn.Module):
    """
    Gaussian policy that outputs (mu, log_std) for continuous actions.
    Action = delta vector in input space, applied as x_adv = x_orig + delta.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_log_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_std = torch.clamp(self.fc_log_std(h), -5, 2)
        return mu, log_std

    def sample_action(self, x):
        """Reparameterized sample with log_prob and entropy."""
        mu, log_std = self(x)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        action = mu + std * eps
        log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2
                    + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def log_prob_of(self, x, action):
        """Compute log_prob of a given action under the current policy."""
        mu, log_std = self(x)
        std = torch.exp(log_std)
        log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2
                    + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(dim=-1, keepdim=True)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOTrainer:
    """
    PPO trainer for the cooperative evolution between generator and detector.
    
    The policy generates delta vectors in input space. The reward is designed
    to balance diversity (entropy) and adversarial quality (deceiving detector),
    following Swift Hydra's formulation with gamma decay.
    """
    def __init__(
            self,
            policy_net: PolicyNetwork,
            value_net: ValueNetwork,
            policy_lr=1e-4,
            value_lr=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            device="cpu"
    ):
        self.device = device
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=value_lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

    def ppo_update(self, states, actions, old_log_probs, returns, advantages, n_epochs=4):
        """
        PPO clipped objective update.
        
        Args:
            states: (N, input_dim) — the x_orig used to generate each sample
            actions: (N, output_dim) — the delta vectors that were applied
            old_log_probs: (N, 1) — log_prob at the time of generation
            returns: (N, 1) — reward (used as return in single-step setting)
            advantages: (N, 1) — advantage estimates
            n_epochs: number of PPO update epochs per call
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        for _ in range(n_epochs):
            # Recompute log_prob of the SAME actions under current policy
            new_log_probs, entropy = self.policy_net.log_prob_of(states, actions)

            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            adv = advantages.detach()
            # Normalize advantages for stability
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obj1 = ratio * adv
            obj2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                               1.0 + self.clip_epsilon) * adv
            policy_loss = -torch.mean(torch.min(obj1, obj2))

            # Value loss
            values_pred = self.value_net(states)
            value_loss = F.mse_loss(values_pred, returns)

            # Entropy bonus
            entropy_bonus = entropy.mean()

            total_loss = (policy_loss
                          + self.value_coefficient * value_loss
                          - self.entropy_coefficient * entropy_bonus)

            # Update policy
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
