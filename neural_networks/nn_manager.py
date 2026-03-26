from .prediction_network import PredictionNetwork
from .dynamics_network import DynamicsNetwork
from .representation_network import RepresentationNetwork
from state_managers.gsm import GameStateManager

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkManager(nn.Module):
    def __init__(
        self,
        representation: RepresentationNetwork,
        dynamics: DynamicsNetwork,
        prediction: PredictionNetwork,
        learning_rate=0.01,
        device: str | None = None,
    ):
        super().__init__()
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.params = (
            list(self.representation.parameters())
            + list(self.dynamics.parameters())
            + list(self.prediction.parameters())
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.training_steps = 0

    def initial_inference(self, state_stack):
        """Real states -> latent -> policy/value"""
        z = self.representation.forward(state_stack)
        policy_logits, value = self.prediction.forward(z)
        return z, policy_logits, value

    def recurrent_inference(self, latent, action):
        """Latent + action -> next latent/reward -> policy/value"""
        next_latent, reward = self.dynamics.forward(latent, action)
        policy_logits, value = self.prediction.forward(next_latent)
        return next_latent, reward, policy_logits, value

    def unroll(self, state_stack, action_seq):
        """
        state_stack: [B, H, state_dim]
        action_seq : [B, W]

        returns:
            policy_logits: [B, W+1, action_size]
            values:        [B, W+1]
            rewards:       [B, W]

        It does 3 things:
        Convert real state → latent (z)
        Predict action probabilities
        Predict value
        """
        if state_stack.dim() == 2:
            state_stack = state_stack.unsqueeze(0)
        if action_seq.dim() == 1:
            action_seq = action_seq.unsqueeze(0)

        z, p0, v0 = self.initial_inference(state_stack)

        policies = [p0]
        values = [v0]
        rewards = []

        for t in range(action_seq.size(1)):
            a = action_seq[:, t]  # shape [B]
            z, r, p, v = self.recurrent_inference(z, a)
            rewards.append(r)
            policies.append(p)
            values.append(v)

        return {
            "policies": policies,  # length w+1
            "values": values,  # length w+1
            "rewards": rewards,  # length w
        }

    def loss(self, batch):
        pred = self.unroll(batch["states"], batch["actions"])

        target_policy = batch["target_policy"].float()
        target_value = batch["target_value"].float()
        target_reward = batch["target_reward"].float()

        log_probs = F.log_softmax(pred["policy_logits"], dim=-1)
        policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

        value_loss = F.mse_loss(pred["values"], target_value)
        reward_loss = F.mse_loss(pred["rewards"], target_reward)

        total_loss = policy_loss + value_loss + reward_loss

        return total_loss, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "reward_loss": reward_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()

        total_loss, metrics = self.loss(batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.optimizer.step()

        self.training_steps += 1
        return metrics

    @torch.no_grad()
    def predict_root(self, state_stack):
        """
        Used by MCTS root:
        state_stack: [B, H, state_dim] or [H, state_dim]
        returns:
            latent, policy_probs, value
        """
        self.eval()

        if state_stack.dim() == 2:
            state_stack = state_stack.unsqueeze(0)

        state_stack = state_stack.to(self.device).float()
        z, logits, value = self.initial_inference(state_stack)
        policy_probs = torch.softmax(logits, dim=-1)
        return z, policy_probs, value

    def get_nn(self):
        return self.dynamics, self.representation, self.prediction
