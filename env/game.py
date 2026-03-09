# Source code from: https://github.com/jmaira123/Ball-Catcher-Game.git

# Grid Based Catch The Ball
import pygame
import random
import sys

pygame.init()

# Screen
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300

# Grid settings
CELL_SIZE = 30
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

# Colors
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)
BACKGROUND_COLOR = (135,206,235)

# Basket settings (in grid cells)
BASKET_WIDTH = 3
BASKET_HEIGHT = 1

# Ball
BALL_FALL_SPEED = 1

# Timer
START_TIME = 30

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Grid Catch The Ball")

clock = pygame.time.Clock()
font = pygame.font.Font(None,36)

# Basket position (grid)
basket_x = GRID_WIDTH//2 - BASKET_WIDTH//2
basket_y = GRID_HEIGHT - 2

# Ball position (grid)
ball_x = random.randint(0, GRID_WIDTH-1)
ball_y = 0

score = 0
start_ticks = pygame.time.get_ticks()

def draw_grid():
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT), 1)

    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y), 1)

def draw_basket(x,y):
    rect = pygame.Rect(
        x*CELL_SIZE,
        y*CELL_SIZE,
        BASKET_WIDTH*CELL_SIZE,
        BASKET_HEIGHT*CELL_SIZE
    )
    pygame.draw.rect(screen,GREEN,rect)

def draw_ball(x,y):
    rect = pygame.Rect(
        x*CELL_SIZE,
        y*CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE
    )
    pygame.draw.rect(screen,RED,rect)

def display_score(score):
    text = font.render(f"Score: {score}",True,BLACK)
    screen.blit(text,(10,10))

def display_time(seconds):
    text = font.render(f"Time: {seconds}",True,BLACK)
    screen.blit(text,(SCREEN_WIDTH-150,10))

while True:

    screen.fill(BACKGROUND_COLOR)

    draw_grid()

    seconds = START_TIME - (pygame.time.get_ticks() - start_ticks)//1000
    if seconds <= 0:
        print("Time's up! Final Score:",score)
        pygame.quit()
        sys.exit()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Move basket one grid cell per key press
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and basket_x > 0:
                basket_x -= 1
            if event.key == pygame.K_RIGHT and basket_x < GRID_WIDTH - BASKET_WIDTH:
                basket_x += 1

    # Ball falls one grid cell
    ball_y += BALL_FALL_SPEED

    # Catch check
    if ball_y == basket_y and basket_x <= ball_x < basket_x + BASKET_WIDTH:
        score += 1
        ball_x = random.randint(0, GRID_WIDTH-1)
        ball_y = 0

    elif ball_y > GRID_HEIGHT:
        ball_x = random.randint(0, GRID_WIDTH-1)
        ball_y = 0

    draw_basket(basket_x,basket_y)
    draw_ball(ball_x,ball_y)

    display_score(score)
    display_time(seconds)

    pygame.display.flip()
    clock.tick(10)