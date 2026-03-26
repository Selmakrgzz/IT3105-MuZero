import torch
import torch.nn as nn

from dynamics_network import DynamicsNetwork
from prediction_network import PredictionNetwork
from representation_network import RepresentationNetwork


class NeuralNetworkManager:
    def __init__(
        self,
        representation: RepresentationNetwork,
        dynamics: DynamicsNetwork,
        prediction: PredictionNetwork,
        learning_rate=0.01,
    ):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
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
        z, p0, v0 = self.initial_inference(state_stack)

        policies = [p0]
        values = [v0]
        rewards = []

        for a in action_seq:
            z, r, p, v = self.recurrent_inference(z, a)
            rewards.append(r)
            policies.append(p)
            values.append(v)

        return {
            "policies": policies,   # length w+1
            "values": values,       # length w+1
            "rewards": rewards,     # length w
        }
        
    def loss(self, batch):
        pred = self.unroll(batch["states"], batch["actions"])
        # policy loss + value loss + reward loss
        total_loss = policy_loss + value_loss + reward_loss


        
