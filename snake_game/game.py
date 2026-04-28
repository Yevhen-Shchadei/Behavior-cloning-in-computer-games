import csv
import math
import os
import random
import sys

import pygame

from .config import DATA_FILE

pygame.init()

# ── Constants ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 700
GRID_SIZE = 20
COLS = WIDTH // GRID_SIZE
ROWS = (HEIGHT - 120) // GRID_SIZE
GRID_OFFSET_Y = 80

FPS = 8

# ── Dataset recording ──────────────────────────────────────────────────────
RECORD_DATASET = True

# ── Palette ────────────────────────────────────────────────────────────────
BG_OUTER_TOP = (5, 8, 18)
BG_OUTER_BOTTOM = (8, 12, 24)

FIELD_TOP = (10, 18, 38)
FIELD_BOTTOM = (5, 10, 20)

HUD_TOP = (14, 18, 34)
HUD_BOTTOM = (10, 13, 24)

GRID_LINE = (26, 35, 70)
GRID_LINE_SOFT = (18, 24, 48)

NEON_CYAN = (0, 255, 170)
NEON_CYAN_SOFT = (0, 200, 140)
NEON_CYAN_DARK = (0, 120, 90)

SNAKE_HEAD = (0, 255, 170)
SNAKE_BODY = (0, 215, 145)
SNAKE_TAIL = (0, 125, 88)

FOOD_COLOR = (255, 70, 105)
FOOD_INNER = (255, 190, 205)
FOOD_OUTLINE = (255, 120, 145)

TEXT_MAIN = (220, 235, 255)
TEXT_SOFT = (150, 175, 210)
TEXT_DIM = (85, 105, 145)

PANEL_BG = (18, 24, 40)
PANEL_BORDER = (45, 70, 120)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ── Helpers ────────────────────────────────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_vertical_gradient(surface, rect, top_color, bottom_color):
    x, y, w, h = rect
    if h <= 0:
        return

    for i in range(h):
        t = i / max(1, h - 1)
        color = lerp_color(top_color, bottom_color, t)
        pygame.draw.line(surface, color, (x, y + i), (x + w, y + i))


def draw_glow(surface, color, center, radius, layers=6, alpha_scale=1.0):
    if radius <= 0:
        return

    glow_surf = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        alpha = int(85 * (i / layers) ** 2 * alpha_scale)
        r = int(radius * i / layers)
        pygame.draw.circle(glow_surf, (*color[:3], alpha), (radius + 1, radius + 1), r)

    surface.blit(glow_surf, (center[0] - radius - 1, center[1] - radius - 1))


def draw_rounded_rect(surface, color, rect, radius=6, width=0):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)


def draw_text_shadow(surface, font, text, color, pos, shadow=(0, 0, 0), offset=(2, 2)):
    shadow_surf = font.render(text, True, shadow)
    text_surf = font.render(text, True, color)
    surface.blit(shadow_surf, (pos[0] + offset[0], pos[1] + offset[1]))
    surface.blit(text_surf, pos)


def cell_rect(col, row):
    x = col * GRID_SIZE
    y = row * GRID_SIZE + GRID_OFFSET_Y
    return pygame.Rect(x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2)


def cell_center(col, row):
    r = cell_rect(col, row)
    return r.centerx, r.centery


# ── Particles ──────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, speed_scale=1.0):
        self.x = x
        self.y = y

        angle = random.uniform(0, math.tau)
        speed = random.uniform(1.3, 4.8) * speed_scale
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        self.life = 1.0
        self.decay = random.uniform(0.03, 0.075)
        self.size = random.uniform(2.0, 6.0)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.10
        self.vx *= 0.975
        self.life -= self.decay

    def draw(self, surface):
        if self.life <= 0:
            return

        alpha = int(self.life * 255)
        r = max(1, int(self.size * self.life))
        s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color[:3], alpha), (r + 1, r + 1), r)
        surface.blit(s, (int(self.x) - r - 1, int(self.y) - r - 1))


# ── Main game ──────────────────────────────────────────────────────────────
class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SNAKE · neon deluxe")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_huge = pygame.font.SysFont("consolas", 62, bold=True)
        self.font_big = pygame.font.SysFont("consolas", 44, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 18)
        self.font_tiny = pygame.font.SysFont("consolas", 14)

        self.high_score = 0
        self.state = "menu"  # menu | playing | dead

        self.particles = []
        self.food_anim = 0.0
        self.trail = []
        self.ambient_t = 0.0

        self.bg_orbs = self._create_bg_orbs()
        self.stars = self._create_stars()

        self.shake_timer = 0
        self.shake_strength = 0
        self.score_flash_timer = 0

        # Dataset
        self.recording = RECORD_DATASET
        self.data_file = str(DATA_FILE)
        self.pending_action = 0  # 0=straight, 1=left, 2=right

        self._init_dataset_file()
        self._reset()

    # ── Dataset helpers ────────────────────────────────────────────────────
    def _init_dataset_file(self):
        if self.recording and not os.path.exists(self.data_file):
            with open(self.data_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "dir_up",
                        "dir_down",
                        "dir_left",
                        "dir_right",
                        "danger_straight",
                        "danger_left",
                        "danger_right",
                        "dx_food",
                        "dy_food",
                    ]
                )

    def _log_sample(self, action):
        if not self.recording or self.state != "playing":
            return

        features = self._get_features()
        row = features + [action]

        with open(self.data_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _remove_last_n_rows(self, n=40):
        """Видаляє останні n рядків з CSV датасету."""
        if not os.path.exists(self.data_file):
            print(f"Файл {self.data_file} не існує")
            return

        try:
            # Читаємо весь файл
            with open(self.data_file, "r", newline="", encoding="utf-8") as f:
                lines = f.readlines()

            # Залишаємо заголовок + всі рядки крім останніх n
            if len(lines) > n + 1:  # +1 для заголовка
                new_lines = [lines[0]] + lines[1:-n]
            else:
                # Якщо менше n рядків, залишаємо тільки заголовок
                new_lines = [lines[0]]

            # Записуємо назад
            with open(self.data_file, "w", newline="", encoding="utf-8") as f:
                f.writelines(new_lines)

            rows_removed = min(n, len(lines) - 1)
            print(f"✓ delated last {rows_removed} rows")
        except Exception as e:
            print(f"✗ delate rows error: {e}")

    # ── Background ─────────────────────────────────────────────────────────
    def _create_bg_orbs(self):
        orbs = []
        for _ in range(10):
            orbs.append(
                {
                    "x": random.randint(0, WIDTH),
                    "y": random.randint(GRID_OFFSET_Y, HEIGHT),
                    "r": random.randint(60, 180),
                    "speed": random.uniform(0.03, 0.10),
                    "phase": random.uniform(0, math.tau),
                    "alpha": random.randint(12, 28),
                    "color": random.choice(
                        [
                            (0, 255, 170),
                            (0, 180, 255),
                            (120, 80, 255),
                        ]
                    ),
                }
            )
        return orbs

    def _create_stars(self):
        stars = []
        for _ in range(65):
            stars.append(
                {
                    "x": random.uniform(0, WIDTH),
                    "y": random.uniform(0, HEIGHT),
                    "r": random.uniform(0.8, 2.2),
                    "speed": random.uniform(0.05, 0.25),
                    "alpha": random.randint(30, 120),
                    "phase": random.uniform(0, math.tau),
                }
            )
        return stars

    def _update_bg(self):
        self.ambient_t += 0.025

        for orb in self.bg_orbs:
            orb["phase"] += orb["speed"]

        for s in self.stars:
            s["y"] += s["speed"]
            if s["y"] > HEIGHT:
                s["y"] = 0
                s["x"] = random.uniform(0, WIDTH)

        if self.shake_timer > 0:
            self.shake_timer -= 1

        if self.score_flash_timer > 0:
            self.score_flash_timer -= 1

    def _get_shake_offset(self):
        if self.shake_timer <= 0:
            return 0, 0

        strength = max(1, int(self.shake_strength * (self.shake_timer / 10)))
        return random.randint(-strength, strength), random.randint(-strength, strength)

    # ── Feature engineering ────────────────────────────────────────────────
    def _turn_left(self, direction):
        dx, dy = direction
        return (dy, -dx)

    def _turn_right(self, direction):
        dx, dy = direction
        return (-dy, dx)

    def _next_pos(self, pos, direction):
        x, y = pos
        dx, dy = direction
        return ((x + dx) % COLS, (y + dy) % ROWS)

    def _direction_to_relative_action(self, new_dir):
        if new_dir == self.direction:
            return 0
        if new_dir == self._turn_left(self.direction):
            return 1
        if new_dir == self._turn_right(self.direction):
            return 2
        return None

    # def _get_features(self):
    #     hx, hy = self.snake[0]
    #     fx, fy = self.food
    #     dx, dy = self.direction

    #     dir_up = 1 if (dx, dy) == (0, -1) else 0
    #     dir_down = 1 if (dx, dy) == (0, 1) else 0
    #     dir_left = 1 if (dx, dy) == (-1, 0) else 0
    #     dir_right = 1 if (dx, dy) == (1, 0) else 0

    #     food_up = 1 if fy < hy else 0
    #     food_down = 1 if fy > hy else 0
    #     food_left = 1 if fx < hx else 0
    #     food_right = 1 if fx > hx else 0

    #     straight_dir = self.direction
    #     left_dir = self._turn_left(self.direction)
    #     right_dir = self._turn_right(self.direction)

    #     straight_pos = self._next_pos((hx, hy), straight_dir)
    #     left_pos = self._next_pos((hx, hy), left_dir)
    #     right_pos = self._next_pos((hx, hy), right_dir)

    #     body = self.snake[1:]

    #     danger_straight = 1 if straight_pos in body else 0
    #     danger_left = 1 if left_pos in body else 0
    #     danger_right = 1 if right_pos in body else 0

    #     dx_food = (fx - hx) / COLS
    #     dy_food = (fy - hy) / ROWS

    #     return [
    #         dir_up, dir_down, dir_left, dir_right,
    #         food_up, food_down, food_left, food_right,
    #         danger_straight, danger_left, danger_right,
    #         dx_food, dy_food
    #     ]

    def _get_features(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        dx, dy = self.direction

        # 1. Поточний напрямок руху (One-hot encoding)
        dir_up = 1 if (dx, dy) == (0, -1) else 0
        dir_down = 1 if (dx, dy) == (0, 1) else 0
        dir_left = 1 if (dx, dy) == (-1, 0) else 0
        dir_right = 1 if (dx, dy) == (1, 0) else 0

        # Визначаємо вектори для перевірки перешкод
        straight_dir = self.direction
        left_dir = self._turn_left(self.direction)
        right_dir = self._turn_right(self.direction)

        # Позиції, куди змійка потрапить на наступному кроці
        # Використовуємо залишок від ділення (%), щоб врахувати відсутність стін (телепортацію)
        def get_wrapped_pos(pos):
            px, py = pos
            return (px % COLS, py % ROWS)

        straight_pos = get_wrapped_pos(self._next_pos((hx, hy), straight_dir))
        left_pos = get_wrapped_pos(self._next_pos((hx, hy), left_dir))
        right_pos = get_wrapped_pos(self._next_pos((hx, hy), right_dir))

        body = self.snake[1:]

        # 2. Перевірка небезпеки (тільки власне тіло)
        danger_straight = 1 if straight_pos in body else 0
        danger_left = 1 if left_pos in body else 0
        danger_right = 1 if right_pos in body else 0

        # 3. Відносні координати їжі (Нормалізовані від -1 до 1)
        # В безмежному світі "відстань" до їжі може бути хитрою (через край ближче),
        # але стандартна дельта dx/dy зазвичай достатня для навчання.
        dx_food = (fx - hx) / COLS
        dy_food = (fy - hy) / ROWS

        return [
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            danger_straight,
            danger_left,
            danger_right,
            dx_food,
            dy_food,
        ]

    # ── Init / reset ───────────────────────────────────────────────────────
    def _reset(self):
        mid_col = COLS // 2
        mid_row = ROWS // 2
        self.snake = [
            (mid_col, mid_row),
            (mid_col - 1, mid_row),
            (mid_col - 2, mid_row),
        ]
        self.direction = (1, 0)
        self.next_direction = (1, 0)
        self.score = 0
        self.particles = []
        self.trail = []
        self.pending_action = 0
        self.food_anim = 0.0
        self.shake_timer = 0
        self.shake_strength = 0
        self.score_flash_timer = 0
        self._place_food()

    def _place_food(self):
        empty = [
            (c, r) for c in range(COLS) for r in range(ROWS) if (c, r) not in self.snake
        ]
        self.food = random.choice(empty) if empty else (0, 0)

    # ── Main loop ──────────────────────────────────────────────────────────
    def run(self):
        while True:
            self.clock.tick(FPS)
            self._handle_events()
            self._update_bg()

            if self.state == "playing":
                self._update()

            self._draw()

    # ── Input ──────────────────────────────────────────────────────────────
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # Ctrl+Z - видалити останні 100 рядків з датасету
                if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self._remove_last_n_rows(100)
                    return

                if self.state == "menu":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.state = "playing"
                        self._reset()

                elif self.state == "playing":
                    # Тільки ESC обробляємо в KEYDOWN під час гри
                    # Напрямки обробляються в _update() через pygame.key.get_pressed()
                    if event.key == pygame.K_ESCAPE:
                        self.state = "menu"

                elif self.state == "dead":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_r):
                        self.state = "playing"
                        self._reset()
                    if event.key == pygame.K_ESCAPE:
                        self.state = "menu"

    # ── Logic ──────────────────────────────────────────────────────────────
    def _update(self):
        self.food_anim = (self.food_anim + 0.16) % math.tau

        # Перевіряємо вже натиснуті клавіші (не чекаючи на KEYDOWN)
        keys = pygame.key.get_pressed()
        dir_map = {
            pygame.K_UP: (0, -1),
            pygame.K_w: (0, -1),
            pygame.K_DOWN: (0, 1),
            pygame.K_s: (0, 1),
            pygame.K_LEFT: (-1, 0),
            pygame.K_a: (-1, 0),
            pygame.K_RIGHT: (1, 0),
            pygame.K_d: (1, 0),
        }

        for key, nd in dir_map.items():
            if keys[key]:
                # Перевіряємо що напрямок не протилежний
                if nd[0] != -self.direction[0] or nd[1] != -self.direction[1]:
                    action = self._direction_to_relative_action(nd)
                    if action is not None:
                        self.pending_action = action
                    self.next_direction = nd
                break  # Обробляємо тільки першу натиснуту клавішу

        self._log_sample(self.pending_action)
        self.pending_action = 0

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        self.direction = self.next_direction
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = ((hx + dx) % COLS, (hy + dy) % ROWS)

        if new_head in self.snake:
            self._spawn_death_particles()
            self.shake_timer = 12
            self.shake_strength = 8

            if self.score > self.high_score:
                self.high_score = self.score

            self.state = "dead"
            return

        tail_pos = self.snake[-1]
        self.trail.append((*cell_center(*tail_pos), 0.68))

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 10
            self.score_flash_timer = 16
            self.shake_timer = 6
            self.shake_strength = 3
            self._spawn_eat_particles(new_head)
            self._place_food()
        else:
            self.snake.pop()

    def _spawn_eat_particles(self, cell):
        cx, cy = cell_center(*cell)
        for _ in range(22):
            self.particles.append(Particle(cx, cy, FOOD_COLOR, speed_scale=1.0))
        for _ in range(10):
            self.particles.append(Particle(cx, cy, NEON_CYAN, speed_scale=0.8))

    def _spawn_death_particles(self):
        for seg in self.snake:
            cx, cy = cell_center(*seg)
            for _ in range(5):
                self.particles.append(Particle(cx, cy, SNAKE_HEAD, speed_scale=1.2))

    # ── Drawing ────────────────────────────────────────────────────────────
    def _draw(self):
        world = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        self._draw_outer_background(world)
        self._draw_ambient_background(world)
        self._draw_hud_background(world)
        self._draw_grid(world)

        if self.state == "menu":
            self._draw_menu(world)
        elif self.state in ("playing", "dead"):
            self._draw_trail(world)
            self._draw_food(world)
            self._draw_snake(world)
            self._draw_particles(world)
            self._draw_hud(world)

            if self.state == "dead":
                self._draw_death_overlay(world)

        self._draw_scanlines(world)
        self._draw_vignette(world)

        shake_x, shake_y = self._get_shake_offset()
        self.screen.fill(BLACK)
        self.screen.blit(world, (shake_x, shake_y))
        pygame.display.flip()

    def _draw_outer_background(self, surface):
        draw_vertical_gradient(
            surface, (0, 0, WIDTH, HEIGHT), BG_OUTER_TOP, BG_OUTER_BOTTOM
        )

    def _draw_ambient_background(self, surface):
        for orb in self.bg_orbs:
            y_float = orb["y"] + math.sin(self.ambient_t + orb["phase"]) * 15
            alpha_scale = 0.6 + 0.4 * math.sin(self.ambient_t + orb["phase"])
            draw_glow(
                surface,
                orb["color"],
                (int(orb["x"]), int(y_float)),
                orb["r"],
                layers=8,
                alpha_scale=max(0.2, alpha_scale * 0.25),
            )

        for s in self.stars:
            pulse = 0.55 + 0.45 * math.sin(self.ambient_t * 2.2 + s["phase"])
            alpha = int(s["alpha"] * pulse)
            star_surf = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(star_surf, (200, 240, 255, alpha), (4, 4), int(s["r"]))
            surface.blit(star_surf, (int(s["x"]), int(s["y"])))

    def _draw_hud_background(self, surface):
        hud_rect = pygame.Rect(0, 0, WIDTH, GRID_OFFSET_Y)
        draw_vertical_gradient(surface, hud_rect, HUD_TOP, HUD_BOTTOM)
        pygame.draw.line(
            surface, PANEL_BORDER, (0, GRID_OFFSET_Y - 1), (WIDTH, GRID_OFFSET_Y - 1), 2
        )

    def _draw_grid(self, surface):
        field_rect = pygame.Rect(0, GRID_OFFSET_Y, WIDTH, ROWS * GRID_SIZE)
        draw_vertical_gradient(surface, field_rect, FIELD_TOP, FIELD_BOTTOM)

        grid_surface = pygame.Surface((WIDTH, ROWS * GRID_SIZE), pygame.SRCALPHA)
        pulse = 0.65 + 0.35 * math.sin(self.ambient_t * 2.0)

        for c in range(COLS + 1):
            x = c * GRID_SIZE
            pygame.draw.line(
                grid_surface,
                (*GRID_LINE, int(42 * pulse)),
                (x, 0),
                (x, ROWS * GRID_SIZE),
            )

        for r in range(ROWS + 1):
            y = r * GRID_SIZE
            pygame.draw.line(
                grid_surface, (*GRID_LINE_SOFT, int(65 * pulse)), (0, y), (WIDTH, y)
            )

        surface.blit(grid_surface, (0, GRID_OFFSET_Y))

        pygame.draw.rect(surface, (35, 55, 100), field_rect, 2, border_radius=5)

        corner_len = 18
        color = (65, 110, 190)
        x0, y0, w, h = field_rect
        corners = [
            (
                (x0 + 6, y0 + 6),
                (x0 + 6 + corner_len, y0 + 6),
                (x0 + 6, y0 + 6 + corner_len),
            ),
            (
                (x0 + w - 6, y0 + 6),
                (x0 + w - 6 - corner_len, y0 + 6),
                (x0 + w - 6, y0 + 6 + corner_len),
            ),
            (
                (x0 + 6, y0 + h - 6),
                (x0 + 6 + corner_len, y0 + h - 6),
                (x0 + 6, y0 + h - 6 - corner_len),
            ),
            (
                (x0 + w - 6, y0 + h - 6),
                (x0 + w - 6 - corner_len, y0 + h - 6),
                (x0 + w - 6, y0 + h - 6 - corner_len),
            ),
        ]
        for origin, hx, hy in corners:
            pygame.draw.line(surface, color, origin, hx, 2)
            pygame.draw.line(surface, color, origin, hy, 2)

    def _draw_snake(self, surface):
        n = len(self.snake)

        for i, (col, row) in enumerate(self.snake):
            t = i / max(n - 1, 1)
            base_color = lerp_color(SNAKE_HEAD, SNAKE_TAIL, t)
            rect = cell_rect(col, row)

            if i == 0:
                draw_glow(
                    surface,
                    SNAKE_HEAD,
                    rect.center,
                    GRID_SIZE + 8,
                    layers=6,
                    alpha_scale=0.9,
                )

            inner = rect.inflate(-2, -2)
            draw_rounded_rect(surface, base_color, rect, radius=6)
            draw_rounded_rect(
                surface, lerp_color(base_color, WHITE, 0.18), inner, radius=5
            )

            shine = pygame.Rect(rect.x + 3, rect.y + 3, rect.w // 3, rect.h // 3)
            pygame.draw.ellipse(surface, (255, 255, 255, 120), shine)

            outline_color = lerp_color(base_color, BLACK, 0.45)
            draw_rounded_rect(surface, outline_color, rect, radius=6, width=2)

            if i == 0:
                dx, dy = self.direction
                cx, cy = rect.centerx, rect.centery
                e1x = cx + dx * 4 + dy * 4
                e1y = cy + dy * 4 - dx * 4
                e2x = cx + dx * 4 - dy * 4
                e2y = cy + dy * 4 + dx * 4

                pygame.draw.circle(surface, WHITE, (e1x, e1y), 4)
                pygame.draw.circle(surface, WHITE, (e2x, e2y), 4)
                pygame.draw.circle(surface, BLACK, (e1x, e1y), 2)
                pygame.draw.circle(surface, BLACK, (e2x, e2y), 2)

                tongue_len = 7 + int(2 * math.sin(self.ambient_t * 7))
                tx = cx + dx * (GRID_SIZE // 2 + tongue_len)
                ty = cy + dy * (GRID_SIZE // 2 + tongue_len)
                pygame.draw.line(
                    surface, (255, 80, 110), (cx + dx * 8, cy + dy * 8), (tx, ty), 2
                )

    def _draw_food(self, surface):
        col, row = self.food
        cx, cy = cell_center(col, row)

        pulse = 0.78 + 0.22 * math.sin(self.food_anim)
        r_outer = int(GRID_SIZE * 0.44 * pulse)
        r_mid = max(3, int(r_outer * 0.72))
        r_inner = max(2, int(r_outer * 0.38))

        draw_glow(
            surface, FOOD_COLOR, (cx, cy), r_outer + 10, layers=7, alpha_scale=0.95
        )
        pygame.draw.circle(surface, FOOD_OUTLINE, (cx, cy), r_outer + 1)
        pygame.draw.circle(surface, FOOD_COLOR, (cx, cy), r_mid)
        pygame.draw.circle(surface, FOOD_INNER, (cx, cy), r_inner)

        shine_x = cx - r_outer // 3
        shine_y = cy - r_outer // 3
        pygame.draw.circle(
            surface, (255, 235, 240), (shine_x, shine_y), max(1, r_inner // 2)
        )

    def _draw_trail(self, surface):
        self.trail = [(x, y, a - 0.08) for x, y, a in self.trail if a > 0]
        for x, y, alpha in self.trail:
            if alpha <= 0:
                continue
            r = max(1, int(5 * alpha))
            s = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*SNAKE_TAIL, int(alpha * 70)), (r * 2, r * 2), r * 2)
            pygame.draw.circle(s, (*SNAKE_HEAD, int(alpha * 120)), (r * 2, r * 2), r)
            surface.blit(s, (int(x) - r * 2, int(y) - r * 2))

    def _draw_particles(self, surface):
        for p in self.particles:
            p.draw(surface)

    def _draw_hud(self, surface):
        left_card = pygame.Rect(14, 10, 210, 58)
        draw_rounded_rect(surface, PANEL_BG, left_card, radius=12)
        draw_rounded_rect(surface, PANEL_BORDER, left_card, radius=12, width=2)

        title = self.font_med.render("SNAKE", True, NEON_CYAN)
        surface.blit(title, (26, 18))
        pygame.draw.line(surface, NEON_CYAN, (26, 48), (108, 48), 2)

        mode_text = "RECORD ON" if self.recording else "RECORD OFF"
        mode_color = (255, 90, 110) if self.recording else TEXT_DIM
        mode_surf = self.font_tiny.render(mode_text, True, mode_color)
        surface.blit(mode_surf, (130, 20))

        score_card = pygame.Rect(WIDTH // 2 - 95, 10, 190, 58)
        draw_rounded_rect(surface, PANEL_BG, score_card, radius=12)
        draw_rounded_rect(surface, PANEL_BORDER, score_card, radius=12, width=2)

        score_label = self.font_tiny.render("SCORE", True, TEXT_DIM)
        surface.blit(score_label, (WIDTH // 2 - 28, 16))

        pulse = (
            1.0 + 0.12 * math.sin((16 - self.score_flash_timer) * 0.6)
            if self.score_flash_timer > 0
            else 1.0
        )
        score_font = pygame.font.SysFont("consolas", int(28 * pulse), bold=True)
        score_surf = score_font.render(
            f"{self.score:05d}",
            True,
            NEON_CYAN if self.score_flash_timer > 0 else TEXT_MAIN,
        )
        surface.blit(score_surf, (WIDTH // 2 - score_surf.get_width() // 2, 32))

        right_card = pygame.Rect(WIDTH - 160, 10, 146, 58)
        draw_rounded_rect(surface, PANEL_BG, right_card, radius=12)
        draw_rounded_rect(surface, PANEL_BORDER, right_card, radius=12, width=2)

        hi_label = self.font_tiny.render("BEST", True, TEXT_DIM)
        hi_val = self.font_med.render(f"{self.high_score:05d}", True, TEXT_MAIN)
        surface.blit(hi_label, (WIDTH - 128, 16))
        surface.blit(hi_val, (WIDTH - 128, 31))

        ctrl = self.font_small.render("WASD / ← ↑ → ↓    ESC — menu", True, TEXT_DIM)
        y_bot = GRID_OFFSET_Y + ROWS * GRID_SIZE + 10
        surface.blit(ctrl, (WIDTH // 2 - ctrl.get_width() // 2, y_bot))

    def _draw_death_overlay(self, surface):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 18, 185))
        surface.blit(overlay, (0, 0))

        card = pygame.Rect(WIDTH // 2 - 190, HEIGHT // 2 - 120, 380, 220)
        draw_rounded_rect(surface, (16, 22, 40), card, radius=18)
        draw_rounded_rect(surface, (70, 95, 150), card, radius=18, width=2)

        draw_glow(
            surface,
            FOOD_COLOR,
            (WIDTH // 2, HEIGHT // 2 - 55),
            90,
            layers=8,
            alpha_scale=0.75,
        )

        title = self.font_big.render("GAME OVER", True, FOOD_COLOR)
        surface.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 62)))

        score_text = self.font_med.render(f"Score: {self.score}", True, TEXT_MAIN)
        surface.blit(
            score_text, score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 5))
        )

        if self.score >= self.high_score and self.score > 0:
            best_text = self.font_med.render("NEW BEST!", True, NEON_CYAN)
            surface.blit(
                best_text, best_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 42))
            )

        info = self.font_small.render(
            "SPACE / R — restart    ESC — menu", True, TEXT_SOFT
        )
        surface.blit(info, info.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 88)))

    def _draw_menu(self, surface):
        center_y = HEIGHT // 2 - 80

        draw_glow(
            surface, NEON_CYAN, (WIDTH // 2, center_y), 135, layers=10, alpha_scale=0.65
        )

        title = self.font_huge.render("SNAKE", True, NEON_CYAN)
        sub = self.font_small.render("n e o n   d e l u x e", True, TEXT_DIM)

        surface.blit(title, title.get_rect(center=(WIDTH // 2, center_y)))
        surface.blit(sub, sub.get_rect(center=(WIDTH // 2, center_y + 54)))

        pygame.draw.line(
            surface,
            (40, 80, 150),
            (WIDTH // 2 - 175, center_y + 78),
            (WIDTH // 2 + 175, center_y + 78),
            1,
        )

        pulse = 0.72 + 0.28 * math.sin(self.ambient_t * 3.2)
        start_color = lerp_color((70, 120, 200), NEON_CYAN, pulse)
        start = self.font_med.render("[ PRESS SPACE ]", True, start_color)
        surface.blit(start, start.get_rect(center=(WIDTH // 2, center_y + 130)))

        if self.high_score:
            hi = self.font_small.render(f"Best: {self.high_score}", True, TEXT_SOFT)
            surface.blit(hi, hi.get_rect(center=(WIDTH // 2, center_y + 172)))

        info_card = pygame.Rect(WIDTH // 2 - 170, HEIGHT - 120, 340, 62)
        draw_rounded_rect(surface, (16, 22, 40), info_card, radius=14)
        draw_rounded_rect(surface, (55, 80, 130), info_card, radius=14, width=2)

        ctrl1 = self.font_small.render("WASD / Arrow keys", True, TEXT_MAIN)
        ctrl2 = self.font_tiny.render(
            "Dataset recording works as before", True, TEXT_DIM
        )
        surface.blit(ctrl1, ctrl1.get_rect(center=(WIDTH // 2, HEIGHT - 96)))
        surface.blit(ctrl2, ctrl2.get_rect(center=(WIDTH // 2, HEIGHT - 74)))

    def _draw_scanlines(self, surface):
        scan = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 4):
            pygame.draw.line(scan, (0, 0, 0, 18), (0, y), (WIDTH, y))
        surface.blit(scan, (0, 0))

    def _draw_vignette(self, surface):
        vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for i in range(18):
            alpha = int(6 + i * 2.2)
            rect = pygame.Rect(i * 2, i * 2, WIDTH - i * 4, HEIGHT - i * 4)
            pygame.draw.rect(
                vignette, (0, 0, 0, alpha), rect, width=4, border_radius=20
            )
        surface.blit(vignette, (0, 0))


if __name__ == "__main__":
    SnakeGame().run()
