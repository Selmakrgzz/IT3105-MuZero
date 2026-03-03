#!/usr/bin/env python3
"""
Basket Catcher - A grid-based terminal game using curses.
Use LEFT/RIGHT arrow keys (or A/D) to move the basket.
Catch falling items to score points. Miss 3 and it's game over!
Press Q to quit.
"""

import curses
import random
import time

# Game settings
GRID_WIDTH = 30
GRID_HEIGHT = 20
BASKET_WIDTH = 5
INITIAL_SPEED = 0.6   # seconds between frames
MIN_SPEED = 0.15
SPEED_INCREMENT = 0.03  # speed up every 5 points
MAX_MISSES = 3

ITEMS = ["*", "o", "+", "@", "#", "$"]
ITEM_COLORS = [1, 2, 3, 4, 5, 6]  # curses color pair indices


def draw_border(win, height, width):
    win.attron(curses.color_pair(7))
    for x in range(width + 2):
        win.addch(0, x, '-')
        win.addch(height + 1, x, '-')
    for y in range(height + 2):
        win.addch(y, 0, '|')
        win.addch(y, width + 1, '|')
    win.addch(0, 0, '+')
    win.addch(0, width + 1, '+')
    win.addch(height + 1, 0, '+')
    win.addch(height + 1, width + 1, '+')
    win.attroff(curses.color_pair(7))


def draw_basket(win, bx, by):
    basket = "[" + "=" * BASKET_WIDTH + "]"
    win.attron(curses.color_pair(8) | curses.A_BOLD)
    try:
        win.addstr(by + 1, bx + 1, basket)
    except curses.error:
        pass
    win.attroff(curses.color_pair(8) | curses.A_BOLD)


def draw_item(win, item):
    x, y, ch, color = item
    win.attron(curses.color_pair(color) | curses.A_BOLD)
    try:
        win.addch(y + 1, x + 1, ch)
    except curses.error:
        pass
    win.attroff(curses.color_pair(color) | curses.A_BOLD)


def draw_hud(win, score, misses, level, height, width):
    hud_y = height + 2
    hud = f" Score:{score:<4} Lvl:{level:<2} Miss:{'X'*misses}{'.'*(MAX_MISSES-misses)} [A/D]Move [Q]Quit"
    hud = hud[:width]  # truncate to fit
    win.attron(curses.color_pair(7))
    try:
        win.addstr(hud_y, 1, hud)
    except curses.error:
        pass
    win.attroff(curses.color_pair(7))


def game_over_screen(stdscr, score, level):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    cx, cy = w // 2, h // 2
    lines = [
        "╔══════════════════════╗",
        "║      GAME  OVER      ║",
        "╠══════════════════════╣",
        f"║  Final Score: {score:<7} ║",
        f"║  Level Reached: {level:<5} ║",
        "╠══════════════════════╣",
        "║  Press R to restart  ║",
        "║  Press Q to quit     ║",
        "╚══════════════════════╝",
    ]
    for i, line in enumerate(lines):
        try:
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(cy - 4 + i, cx - len(line) // 2, line)
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
        except curses.error:
            pass
    stdscr.refresh()
    while True:
        key = stdscr.getch()
        if key in (ord('q'), ord('Q')):
            return False
        if key in (ord('r'), ord('R')):
            return True


def main(stdscr):
    # Color setup
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_BLUE, -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    curses.init_pair(6, curses.COLOR_CYAN, -1)
    curses.init_pair(7, curses.COLOR_WHITE, -1)
    curses.init_pair(8, curses.COLOR_YELLOW, -1)  # basket

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    while True:
        # Init game state
        basket_x = GRID_WIDTH // 2 - BASKET_WIDTH // 2
        basket_y = GRID_HEIGHT - 1
        items = []
        score = 0
        misses = 0
        level = 1
        speed = INITIAL_SPEED
        last_fall = time.time()
        last_spawn = time.time()
        spawn_interval = 1.2

        # Create a sub-window for the grid
        sh, sw = stdscr.getmaxyx()
        win_h = min(GRID_HEIGHT + 4, sh)
        win_w = min(GRID_WIDTH + 2, sw)
        start_y = max(0, (sh - win_h) // 2)
        start_x = max(0, (sw - win_w) // 2)
        win = stdscr.subwin(win_h, win_w, start_y, start_x)
        win.keypad(True)

        running = True
        while running:
            now = time.time()

            # Input
            key = stdscr.getch()
            if key in (curses.KEY_LEFT, ord('a'), ord('A')):
                basket_x = max(0, basket_x - 1)
            elif key in (curses.KEY_RIGHT, ord('d'), ord('D')):
                basket_x = min(GRID_WIDTH - BASKET_WIDTH - 1, basket_x + 1)
            elif key in (ord('q'), ord('Q')):
                return

            # Spawn items
            if now - last_spawn > spawn_interval:
                x = random.randint(0, GRID_WIDTH - 1)
                ch = random.choice(ITEMS)
                color = random.randint(1, 6)
                items.append([x, 0, ch, color])
                last_spawn = now
                # Vary spawn rate with level
                spawn_interval = max(0.4, 1.2 - level * 0.05)

            # Move items down
            if now - last_fall > speed:
                new_items = []
                for item in items:
                    item[1] += 1
                    bw_end = basket_x + BASKET_WIDTH + 1
                    if item[1] == basket_y and basket_x <= item[0] <= bw_end:
                        # Caught!
                        score += 1
                        level = score // 5 + 1
                        speed = max(MIN_SPEED, INITIAL_SPEED - (level - 1) * SPEED_INCREMENT)
                    elif item[1] >= GRID_HEIGHT:
                        # Missed
                        misses += 1
                    else:
                        new_items.append(item)
                items = new_items
                last_fall = now

            if misses >= MAX_MISSES:
                running = False
                break

            # Draw
            win.erase()
            draw_border(win, GRID_HEIGHT, GRID_WIDTH)
            for item in items:
                draw_item(win, item)
            draw_basket(win, basket_x, basket_y)
            draw_hud(win, score, misses, level, GRID_HEIGHT, GRID_WIDTH)

            # Title
            title = " 🧺 BASKET CATCHER "
            try:
                win.attron(curses.color_pair(3) | curses.A_BOLD)
                win.addstr(0, (GRID_WIDTH + 2 - len(title)) // 2, title)
                win.attroff(curses.color_pair(3) | curses.A_BOLD)
            except curses.error:
                pass

            win.refresh()
            time.sleep(0.03)

        # Game over
        restart = game_over_screen(stdscr, score, level)
        if not restart:
            break


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass