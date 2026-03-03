# Source code from: https://github.com/jmaira123/Ball-Catcher-Game.git
import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (135, 206, 235)
GRID_COLOR = (200, 200, 200)  # light gray for grid lines

# Grid setting
GRID_SIZE = 20         # pixels between grid lines
GRID_THICKNESS = 1     # line thickness

# Ball settings
BALL_RADIUS = 15
BALL_FALL_SPEED = 5

# Basket settings
BASKET_WIDTH = 100
BASKET_HEIGHT = 20

# Timer setting
START_TIME = 30  # seconds

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Catch the Ball")

# Game variables
basket_x = SCREEN_WIDTH // 2 - BASKET_WIDTH // 2
basket_y = SCREEN_HEIGHT - BASKET_HEIGHT - 10
ball_x = random.randint(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)  # keep inside screen
ball_y = 0
score = 0
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Timer variable
start_ticks = pygame.time.get_ticks()

# Grid visibility toggle
show_grid = True

def draw_basket(x, y):
    pygame.draw.rect(screen, GREEN, (x, y, BASKET_WIDTH, BASKET_HEIGHT))

def draw_ball(x, y):
    pygame.draw.circle(screen, RED, (x, y), BALL_RADIUS)

def display_score(current_score):
    score_text = font.render(f"Score: {current_score}", True, BLACK)
    screen.blit(score_text, (10, 10))

def display_time(seconds):
    time_text = font.render(f"Time: {seconds}", True, BLACK)
    screen.blit(time_text, (SCREEN_WIDTH - 150, 10))

def draw_grid():
    """Draws a grid overlay across the entire screen."""
    # Vertical lines
    for x in range(0, SCREEN_WIDTH + 1, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), GRID_THICKNESS)
    # Horizontal lines
    for y in range(0, SCREEN_HEIGHT + 1, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y), GRID_THICKNESS)

# Main game loop
while True:
    screen.fill(BACKGROUND_COLOR)

    # Calculate elapsed time
    seconds = START_TIME - (pygame.time.get_ticks() - start_ticks) // 1000
    if seconds <= 0:
        print(f"Time's up! Final Score: {score}")
        pygame.quit()
        sys.exit()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Toggle grid with G
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                show_grid = not show_grid

    # Move the basket
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and basket_x > 0:
        basket_x -= 10
    if keys[pygame.K_RIGHT] and basket_x < SCREEN_WIDTH - BASKET_WIDTH:
        basket_x += 10

    # Update ball position
    ball_y += BALL_FALL_SPEED

    # Gradually increase difficulty
    if score > 0 and score % 5 == 0:
        BALL_FALL_SPEED = min(10, BALL_FALL_SPEED + 0.1)
        BASKET_WIDTH = max(50, BASKET_WIDTH - 0.1)

    # Check if the ball is caught
    # (Simple check: if the ball's center passes the basket y-level and x is within basket)
    if ball_y >= basket_y and basket_x < ball_x < basket_x + BASKET_WIDTH:
        score += 1
        ball_x = random.randint(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)
        ball_y = 0
    elif ball_y > SCREEN_HEIGHT:
        # Ball missed, reset its position
        ball_x = random.randint(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)
        ball_y = 0

    # Optionally draw grid first (so basket/ball appear on top)
    if show_grid:
        draw_grid()

    # Draw the basket and ball
    draw_basket(basket_x, basket_y)
    draw_ball(ball_x, ball_y)

    # Display the score and time
    display_score(score)
    display_time(seconds)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(30)