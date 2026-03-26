# main.py
from state_managers.gsm import GameStateManager
from mcts.u_mcts import MCTS
from rl.episode_buffer import EpisodeBuffer
from rl.rl_manager import RLManager
from neural_networks.representation_network import RepresentationNetwork
from neural_networks.dynamics_network import DynamicsNetwork
from neural_networks.prediction_network import PredictionNetwork
from neural_networks.nn_manager import NeuralNetworkManager

# --- Parametre ---
NUM_EPISODES = 100
TRAINING_INTERVAL = 10
NUM_SEARCHES = 50
ROLLOUT_DEPTH = 5
Q = 3   # look-back

# --- Sett opp systemet ---
gsm  = GameStateManager()
nnr  = RepresentationNetwork()
nnd  = DynamicsNetwork()
nnp  = PredictionNetwork()
nnm  = NeuralNetworkManager(nnr, nnd, nnp)
mcts = MCTS(num_searches=NUM_SEARCHES, rollout_depth=ROLLOUT_DEPTH)
eb   = EpisodeBuffer()
rlm  = RLManager(gsm, mcts, eb, nnr, nnd, nnp, q=Q)

# --- Kjør ---
print("Starter trening...")
rlm.run(num_episodes=NUM_EPISODES, nnm=nnm, training_interval=TRAINING_INTERVAL)
print("Ferdig!")