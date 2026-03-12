import numpy as np
import random

class CatchEnv:
    def __init__(self, width=20, height=15):
        # Grid dimensions
        self.width = width
        self.height = height

        # Basket settings
        self.basket_width = 3
        self.action_space = 3  # 0 = left, 1 = stay, 2 = right

        # Episode length
        self.max_steps = 200

        self.reset()

    def reset(self):
        # Reset episode state
        self.steps = 0
        
        # Basket at center bottom
        self.basket_x = self.width // 2
        self.basket_y = self.height - 1

        # Ball spawns at random x at top
        self.ball_x = random.randint(0, self.width - 1)
        self.ball_y = 0

        return self._get_state()

    def step(self, action):
        """
        Action:
            0 = left
            1 = stay
            2 = right
        """
        # Move basket
        if action == 0:
            self.basket_x = max(0, self.basket_x - 1)
        elif action == 2:
            self.basket_x = min(self.width - self.basket_width, self.basket_x + 1)

        # Move ball
        self.ball_y += 1

        # Check reward
        reward = 0
        done = False

        # Catch
        if self.ball_y == self.basket_y:
            if self.basket_x <= self.ball_x < self.basket_x + self.basket_width:
                reward = 1
                done = True  # end episode after each catch
            else:
                reward = 0
                done = True  # ball missed = end episode

        # Or ball fell off screen
        if self.ball_y >= self.height:
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        """
        Returns a grid (height x width) with:
            1 = ball
            2 = basket
            0 = empty
        """
        grid = np.zeros((self.height, self.width), dtype=np.float32)

        # Ball
        if 0 <= self.ball_y < self.height:
            grid[self.ball_y][self.ball_x] = 1.0

        # Basket
        for i in range(self.basket_width):
            grid[self.basket_y][self.basket_x + i] = 2.0

        return grid
    
    
