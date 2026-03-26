import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsNetwork(nn.Module):
    def __init__(self, latent_dim: int, action_size: int, hidden_dim: int = 64):
        super().__init__()
        self.action_size = action_size

        self.backbone = nn.Sequential(
            nn.Linear(latent_dim + action_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.next_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor):
        """
        latent: [B, latent_dim]
        action: [B] integer action ids OR [B, action_size] one-hot
        returns:
            next_latent: [B, latent_dim]
            reward:      [B]
        """
        if action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=self.action_size).float()
        print("a ", action.dim(), "l ", latent.dim())
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size).float()
        x = torch.cat([latent, action_onehot], dim=-1)
        #x = torch.cat([latent, action], dim=-1)
        h = self.backbone(x)

        next_latent = torch.relu(self.next_latent_head(h))
        reward = self.reward_head(h).squeeze(-1)

        return next_latent, reward
