import torch.nn as nn
import torch


class RepresentationNetwork(nn.Module):
    def __init__(self, state_dimension, history_len, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.state_dimension = state_dimension
        self.history_len = history_len
        print(hidden_dim)

        # linear network to start with
        self.net = nn.Sequential(
            nn.Linear(self.state_dimension * self.history_len, hidden_dim),
            nn.ReLU(),
            # (64,32)
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, state_list: torch.Tensor) -> torch.Tensor:
        """
        state_stack: [B, history_len, state_dim]
        returns:    [B, latent_dim]
        
        """

        if state_list.dim() == 2:
            # single sample: [history_len, state_dim] adds a dim of 1
            state_list = state_list.unsqueeze(0)

        x = state_list.float().reshape(state_list.size(0), -1)
        # take state(s), flatten them, and convert into a hidden vector z
        z = self.net(x)
        return z
    
    
    def represent(self, state_tensor):
            """Wrapper for MCTS"""
            return self.forward(state_tensor)

