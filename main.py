from state_managers.gsm import GameStateManager
from mcts.u_mcts import MCTS
import torch
from neural_networks.nn_manager import NeuralNetworkManager
# gsm   = GameStateManager()
# mcts  = MCTS(gsm, num_searches=50)
# state = gsm.get_initial_state()

# # Actions
# #LEFT = 0
# #STAY = 1
# #RIGHT = 2

# action, policy, value = mcts.search(state)
# print("Beste handling:", ["LEFT", "STAY", "RIGHT"][action])
# print("Policy:", policy)
# print("Verdi:", value)

from neural_networks.dynamics_network import DynamicsNetwork
from neural_networks.prediction_network import PredictionNetwork
from neural_networks.representation_network import RepresentationNetwork
history_len = 1   # q = 0
rollout_len = 1   # w = 1
action_size = 3
latent_dim = 32

rep = RepresentationNetwork(3, history_len=history_len, latent_dim=latent_dim)
dyn = DynamicsNetwork(latent_dim=latent_dim, action_size=action_size)
pred = PredictionNetwork(latent_dim=latent_dim, action_size=action_size)

nnm = NeuralNetworkManager(rep, dyn, pred)

dummy_batch = {
    "states": torch.randn(8, history_len, 3),
    "actions": torch.randint(0, action_size, (8, rollout_len)),
    "target_policy": torch.softmax(torch.randn(8, rollout_len + 1, action_size), dim=-1),
    "target_value": torch.randn(8, rollout_len + 1),
    "target_reward": torch.randn(8, rollout_len),
}

print(nnm.train_step(dummy_batch))