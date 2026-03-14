from state_managers.gsm import GameStateManager
from mcts.u_mcts import MCTS

gsm   = GameStateManager()
mcts  = MCTS(gsm, num_searches=50)
state = gsm.get_initial_state()

# Actions
#LEFT = 0
#STAY = 1
#RIGHT = 2

action, policy, value = mcts.search(state)
print("Beste handling:", ["LEFT", "STAY", "RIGHT"][action])
print("Policy:", policy)
print("Verdi:", value)