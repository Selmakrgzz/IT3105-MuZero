from vgs import VideoGameSimulator, LEFT, STAY, RIGHT, ACTIONS, GRID_WIDTH, BASKET_WIDTH, BASKET_Y

class GameStateManager:
    """
    Key feature: caches (state, action) -> (next_state, reward) results so
    that MCTS can check many times without re-running the simulator.
    """

    def __init__(self):
        self._sim   = VideoGameSimulator()
        self._cache = {}  # (state, action) -> (next_state, reward)

    # --- episode control ---

    def get_initial_state(self):
        return self._sim.reset()

    def get_current_state(self):
        return self._sim.get_state()

    def apply_action(self, action):
        return self._sim.step(action)

    # --- state queries ---

    def get_next_state_and_reward(self, state, action):
        """
        Given any state and action, return (next_state, reward).
        Uses a cache so repeated MCTS queries are instant.
        """
        key = (state, action)
        if key in self._cache:
            return self._cache[key]

        # Spin up a temporary simulator, force it to the given state, step it
        temp = VideoGameSimulator()
        temp.reset()
        temp.ball_x, temp.ball_y, temp.basket_x = state
        next_state, reward, _ = temp.step(action)

        self._cache[key] = (next_state, reward)
        return next_state, reward

    def get_legal_actions(self, state):
        """
        Return all legal actions for a given state.
        Basket can't move left if already at the left wall and vice versa.
        """
        ball_x, ball_y, basket_x = state
        legal = [STAY]
        if basket_x > 0:
            legal.append(LEFT)
        if basket_x < GRID_WIDTH - BASKET_WIDTH:
            legal.append(RIGHT)
        return legal

    def is_terminal(self, state):
        """
        This game has no true terminal state since
        the episode ends after a fixed number of steps.
        Returns False always, but included so MCTS has a consistent interface.
        """
        return False

    def evaluate_state(self, state):
        """
        Simple heuristic value for a state - how good is this position?
        Used as a fast estimate when MCTS doesn't have a neural network yet.

        Logic: reward the basket for being directly under the ball.
        Returns a value between -1.0 and +1.0.
        """
        ball_x, ball_y, basket_x = state

        # How many columns is the ball from the nearest basket edge?
        basket_center = basket_x + BASKET_WIDTH // 2
        distance = abs(ball_x - basket_center)

        # Closer = better. Normalise to [-1, 1]
        max_distance = GRID_WIDTH
        value = 1.0 - (2.0 * distance / max_distance)
        return value

    def get_state_size(self):
        return self._sim.get_state_size()

    def get_action_size(self):
        return self._sim.get_action_size()

    def clear_cache(self):
        """Empty the cache. Call this between episodes to keep memory tidy"""
        self._cache = {}

