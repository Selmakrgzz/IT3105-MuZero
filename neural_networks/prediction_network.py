import torch
import torch.nn as nn


class PredictionNetwork(nn.Module):
    def __init__(self, latent_dim: int, action_size: int, hidden_dim: int = 64):
        super().__init__()
        # learns features useful for choosing actions and estimating value
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        #[2.1, 0.3, -1.0] after softmax probabilties
        self.policy_head = nn.Linear(hidden_dim, action_size)
        # evaluate situation How good is this state
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor):
        """
        latent: [B, latent_dim]
        returns:
            policy_logits: [B, action_size]
            value:         [B]
        """
        h = self.shared(latent)
        policy_logits = self.policy_head(h)   # raw logits, no softmax here
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value