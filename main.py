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
NUM_EPISODES = 50
TRAINING_INTERVAL = 1
NUM_SEARCHES = 20
ROLLOUT_DEPTH = 3
Q = 3   # look-back


STATE_DIM   = 3        # (ball_x, ball_y, basket_x)
ACTION_SIZE = 3        # LEFT, STAY, RIGHT
HISTORY_LEN = Q + 1   # antall states NNr mottar (look-back + current)
LATENT_DIM  = 32

# --- Sett opp systemet ---
gsm  = GameStateManager()
nnr = RepresentationNetwork(state_dimension=STATE_DIM, history_len=HISTORY_LEN, latent_dim=LATENT_DIM)
nnd = DynamicsNetwork(latent_dim=LATENT_DIM, action_size=ACTION_SIZE)
nnp = PredictionNetwork(latent_dim=LATENT_DIM, action_size=ACTION_SIZE)
nnm  = NeuralNetworkManager(nnr, nnd, nnp)
mcts = MCTS(num_searches=NUM_SEARCHES, rollout_depth=ROLLOUT_DEPTH)
eb   = EpisodeBuffer()
rlm  = RLManager(gsm, mcts, eb, nnr, nnd, nnp, q=Q)

# --- Kjør ---
print("Starter trening...")
rlm.run(num_episodes=NUM_EPISODES, nnm=nnm, training_interval=TRAINING_INTERVAL)
print("Ferdig!")