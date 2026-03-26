import random
import torch

# Actions
LEFT = 0
STAY = 1
RIGHT = 2
ACTIONS = [LEFT, STAY, RIGHT]

# Grid settings
GRID_WIDTH = 10
GRID_HEIGHT = 10
BASKET_WIDTH = 3
BASKET_Y = GRID_HEIGHT - 2  # row the basket always sits on


class VideoGameSimulator:
    """
    Ball catcher game - only logic

    Each episode:
        state = sim.reset()
        while not done:
            state, reward, done = sim.step(action)  # action = LEFT / STAY / RIGHT
    """

    def reset(self):
        """Reset to a new episode. Returns the initial state."""
        self.ball_x   = random.randint(0, GRID_WIDTH - 1)
        self.ball_y   = 0
        self.basket_x = GRID_WIDTH // 2 - BASKET_WIDTH // 2
        self.score    = 0
        self.steps    = 0
        return self.get_state()

    def get_state(self):
        """Current state as a tuple. This is what the neural networks will receive."""
        return torch.tensor(self.ball_x, self.ball_y, self.basket_x)

    def step(self, action):
        """
        Args:
            action: LEFT (0), STAY (1), RIGHT (2)

        Returns:
            next_state : (ball_x, ball_y, basket_x)
            reward     : +1.0 caught, -1.0 missed, 0.0 otherwise
            done       : True when episode ends (after 200 steps)
        """
        # Move basket
        if action == LEFT:
            self.basket_x = max(0, self.basket_x - 1)
        elif action == RIGHT:
            self.basket_x = min(GRID_WIDTH - BASKET_WIDTH, self.basket_x + 1)

        # Drop ball one row
        self.ball_y += 1

        reward = 0.0

        if self.ball_y == BASKET_Y:
            if self.basket_x <= self.ball_x < self.basket_x + BASKET_WIDTH:
                reward = 1.0
                self.score += 1
            else:
                reward = -1.0
            # Spawn a new ball at the top
            self.ball_x = random.randint(0, GRID_WIDTH - 1)
            self.ball_y = 0

        self.steps += 1
        done = self.steps >= 200

        return self.get_state(), reward, done

    def get_legal_actions(self, basket_x):
        """Actions that are valid right now (Used by the MCTS to build the tree 0_0)"""
        legal = [STAY]
        if basket_x > 0:
            legal.append(LEFT)
        if basket_x < GRID_WIDTH - BASKET_WIDTH:
            legal.append(RIGHT)
        return legal

    def get_state_size(self):
        """Input size for the neural networks."""
        return 3  # (ball_x, ball_y, basket_x)

    def get_action_size(self):
        """Output size for the neural networks."""
        return len(ACTIONS)  # 3