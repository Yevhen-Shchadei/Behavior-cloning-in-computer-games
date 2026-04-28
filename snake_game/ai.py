from pathlib import Path
import math
import random
import sys

import pygame
import torch
import torch.nn as nn

from config import BIG_MODEL_PATH, SMALL_MODEL_PATH, RES_MODEL_PATH

pygame.init()

# ── Constants ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 760
GRID_SIZE = 20
COLS = WIDTH // GRID_SIZE
ROWS = (HEIGHT - 180) // GRID_SIZE
GRID_OFFSET_Y = 80

FPS = 12

# ── AI settings ────────────────────────────────────────────────────────────
MODEL_PATH = BIG_MODEL_PATH   # or BIG_MODEL_PATH
AUTO_RESTART = True
RESTART_DELAY_MS = 1200

EXPLORATION_RATE = 0.50
AVOID_DEATH = True

# ── Palette ────────────────────────────────────────────────────────────────
BG_TOP = (6, 10, 20)
BG_BOTTOM = (10, 14, 28)

FIELD_TOP = (10, 18, 38)
FIELD_BOTTOM = (5, 10, 20)

HUD_TOP = (14, 18, 34)
HUD_BOTTOM = (10, 13, 24)

GRID_LINE = (28, 36, 72)
GRID_LINE_SOFT = (18, 24, 48)

NEON_CYAN = (0, 255, 170)
NEON_BLUE = (80, 170, 255)
NEON_PURPLE = (125, 90, 255)

SNAKE_HEAD = (0, 255, 170)
SNAKE_TAIL = (0, 140, 80)

FOOD_COLOR = (255, 70, 105)
FOOD_INNER = (255, 190, 205)
FOOD_OUTLINE = (255, 125, 150)

SCORE_COLOR = (0, 255, 170)
TEXT_COLOR = (220, 235, 255)
TEXT_SOFT = (160, 185, 220)
DIM_COLOR = (85, 105, 145)

PANEL_BG = (18, 24, 40)
PANEL_BORDER = (48, 74, 125)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ── Helpers ────────────────────────────────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


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
        speed = random.uniform(1.5, 5) * speed_scale
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0
        self.decay = random.uniform(0.03, 0.07)
        self.size = random.uniform(2, 6)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12
        self.vx *= 0.97
        self.life -= self.decay

    def draw(self, surface):
        if self.life <= 0:
            return
        alpha = int(self.life * 255)
        r = max(1, int(self.size * self.life))
        s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color[:3], alpha), (r + 1, r + 1), r)
        surface.blit(s, (int(self.x) - r - 1, int(self.y) - r - 1))


# ── Models ─────────────────────────────────────────────────────────────────
class SmallNet(nn.Module):
    def __init__(self, input_dim=9, hidden=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class BiggerNet(nn.Module):
    def __init__(self, input_dim=9, h1=32, h2=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Перший шар
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Другий шар (має повертати таку ж розмірність, як вхід)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Зберігаємо вхідні дані ("пам'ять")
        identity = x

        # Проходимо крізь шари
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        # Головна фішка: додаємо вхід до виходу
        # Це дозволяє сигналу "пролітати" крізь мережу без затухання
        out += identity

        return self.relu(out)


class ResidualNet(nn.Module):
    def __init__(self, input_dim=9, hidden=32, num_classes=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden)
        self.res_block = ResidualBlock(hidden, hidden)
        self.output_layer = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.res_block(x)  
        x = self.output_layer(x)
        return x


# ── Main game ──────────────────────────────────────────────────────────────
class SnakeGameAI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SNAKE AI · neon showcase")
        self.clock = pygame.time.Clock()

        self.font_huge = pygame.font.SysFont("consolas", 50, bold=True)
        self.font_big = pygame.font.SysFont("consolas", 36, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 18)
        self.font_tiny = pygame.font.SysFont("consolas", 14)

        self.high_score = 0
        self.state = "playing"
        self.particles = []
        self.food_anim = 0.0
        self.trail = []
        self.death_time = None

        self.last_action_name = "none"
        self.last_probs = [0.0, 0.0, 0.0]
        self.debug_features = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(MODEL_PATH)

        self.ambient_t = 0.0
        self.bg_orbs = self._create_bg_orbs()
        self.stars = self._create_stars()

        self.shake_timer = 0
        self.shake_strength = 0
        self.score_flash_timer = 0

        self._reset()

    # ── Ambient background ────────────────────────────────────────────────
    def _create_bg_orbs(self):
        orbs = []
        for _ in range(10):
            orbs.append(
                {
                    "x": random.randint(0, WIDTH),
                    "y": random.randint(GRID_OFFSET_Y, HEIGHT),
                    "r": random.randint(70, 170),
                    "speed": random.uniform(0.03, 0.10),
                    "phase": random.uniform(0, math.tau),
                    "color": random.choice([NEON_CYAN, NEON_BLUE, NEON_PURPLE]),
                }
            )
        return orbs

    def _create_stars(self):
        stars = []
        for _ in range(55):
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

    # ── AI helpers ─────────────────────────────────────────────────────────
    def _load_model(self, path):
        path = Path(path)
        path_str = path.name.lower()

        # Визначаємо, який клас створити
        if "res" in path_str:
            model = ResidualNet(input_dim=9) # Стара модель має 9 входів
        elif "bigger" in path_str:
            model = BiggerNet(input_dim=9)
        else:
            model = SmallNet(input_dim=9)

        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        print(f"Loaded model: {path}")
        return model

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

    def _action_to_direction(self, action):
        if action == 0:
            return self.direction
        if action == 1:
            return self._turn_left(self.direction)
        if action == 2:
            return self._turn_right(self.direction)
        return self.direction

    def _is_action_safe(self, action):
        direction = self._action_to_direction(action)
        next_pos = self._next_pos(self.snake[0], direction)
        body = self.snake[1:]
        return next_pos not in body

    def _get_safe_actions(self):
        safe_actions = []
        for action in [0, 1, 2]:
            if self._is_action_safe(action):
                safe_actions.append(action)
        return safe_actions

    def _distance_to_food_after_action(self, action):
        direction = self._action_to_direction(action)
        nx, ny = self._next_pos(self.snake[0], direction)
        fx, fy = self.food
        return abs(fx - nx) + abs(fy - ny)

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
    
    def _predict_action(self):
        features = self._get_features()
        # Створюємо тензор для нейронки (очікує 9 вхідних значень)
        x = torch.tensor([features], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
            model_action = torch.argmax(logits, dim=1).item()

        safe_actions = self._get_safe_actions()

        if safe_actions and random.random() < EXPLORATION_RATE:
            if random.random() < 0.5:
                action = random.choice(safe_actions)
                source = "random"
            else:
                action = min(safe_actions, key=self._distance_to_food_after_action)
                source = "food_seek"
        else:
            action = model_action
            source = "model"

            if AVOID_DEATH and not self._is_action_safe(action):
                if safe_actions:
                    # Обираємо найбезпечнішу дію з тих, що порадила модель
                    action = max(safe_actions, key=lambda a: probs[a])
                    source = "safe_override"

        action_names = {0: "straight", 1: "left", 2: "right"}

        self.last_action_name = f"{action_names.get(action, 'unknown')} [{source}]"
        self.last_probs = probs

        # Оновлені індекси для дебагу (всього 9 значень: від 0 до 8)
        self.debug_features = {
            "dir_up": features[0],
            "dir_down": features[1],
            "dir_left": features[2],
            "dir_right": features[3],
            "danger_straight": features[4],
            "danger_left": features[5],
            "danger_right": features[6],
            "dx_food": round(features[7], 3),
            "dy_food": round(features[8], 3),
        }

        return action

    # ── Reset ──────────────────────────────────────────────────────────────
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
        self.death_time = None
        self.food_anim = 0.0
        self.shake_timer = 0
        self.shake_strength = 0
        self.score_flash_timer = 0
        self._place_food()

        self.last_action_name = "reset"
        self.last_probs = [0.0, 0.0, 0.0]
        self.debug_features = {}

    def _place_food(self):
        empty = [(c, r) for c in range(COLS) for r in range(ROWS) if (c, r) not in self.snake]
        self.food = random.choice(empty) if empty else (0, 0)

    # ── Main loop ──────────────────────────────────────────────────────────
    def run(self):
        while True:
            self.clock.tick(FPS)
            self._handle_events()
            self._update_bg()

            if self.state == "playing":
                self._update()
            elif self.state == "dead" and AUTO_RESTART:
                now = pygame.time.get_ticks()
                if self.death_time is not None and now - self.death_time >= RESTART_DELAY_MS:
                    self.state = "playing"
                    self._reset()

            self._draw()

    # ── Input ──────────────────────────────────────────────────────────────
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    # ── Logic ──────────────────────────────────────────────────────────────
    def _update(self):
        self.food_anim = (self.food_anim + 0.14) % math.tau

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        action = self._predict_action()
        self.next_direction = self._action_to_direction(action)

        self.direction = self.next_direction
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = ((hx + dx) % COLS, (hy + dy) % ROWS)

        if new_head in self.snake:
            self._spawn_death_particles()
            self.high_score = max(self.high_score, self.score)
            self.state = "dead"
            self.death_time = pygame.time.get_ticks()
            self.shake_timer = 12
            self.shake_strength = 8
            return

        tail_pos = self.snake[-1]
        self.trail.append((*cell_center(*tail_pos), 0.68))

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 10
            self.score_flash_timer = 16
            self.shake_timer = 5
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

        self._draw_background(world)
        self._draw_hud_background(world)
        self._draw_grid(world)
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

    def _draw_background(self, surface):
        draw_vertical_gradient(surface, (0, 0, WIDTH, HEIGHT), BG_TOP, BG_BOTTOM)

        for orb in self.bg_orbs:
            y_float = orb["y"] + math.sin(self.ambient_t + orb["phase"]) * 15
            alpha_scale = 0.6 + 0.4 * math.sin(self.ambient_t + orb["phase"])
            draw_glow(
                surface,
                orb["color"],
                (int(orb["x"]), int(y_float)),
                orb["r"],
                layers=8,
                alpha_scale=max(0.2, alpha_scale * 0.22),
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
        pygame.draw.line(surface, PANEL_BORDER, (0, GRID_OFFSET_Y - 1), (WIDTH, GRID_OFFSET_Y - 1), 2)

    def _draw_grid(self, surface):
        field_rect = pygame.Rect(0, GRID_OFFSET_Y, WIDTH, ROWS * GRID_SIZE)
        draw_vertical_gradient(surface, field_rect, FIELD_TOP, FIELD_BOTTOM)

        grid_surface = pygame.Surface((WIDTH, ROWS * GRID_SIZE), pygame.SRCALPHA)
        pulse = 0.65 + 0.35 * math.sin(self.ambient_t * 2.0)

        for c in range(COLS + 1):
            x = c * GRID_SIZE
            pygame.draw.line(grid_surface, (*GRID_LINE, int(42 * pulse)), (x, 0), (x, ROWS * GRID_SIZE))

        for r in range(ROWS + 1):
            y = r * GRID_SIZE
            pygame.draw.line(grid_surface, (*GRID_LINE_SOFT, int(65 * pulse)), (0, y), (WIDTH, y))

        surface.blit(grid_surface, (0, GRID_OFFSET_Y))
        pygame.draw.rect(surface, (35, 55, 100), field_rect, 2, border_radius=5)

    def _draw_snake(self, surface):
        n = len(self.snake)
        for i, (col, row) in enumerate(self.snake):
            t = i / max(n - 1, 1)
            color = lerp_color(SNAKE_HEAD, SNAKE_TAIL, t)
            rect = cell_rect(col, row)

            if i == 0:
                draw_glow(surface, SNAKE_HEAD, rect.center, GRID_SIZE + 8, layers=6)

            inner = rect.inflate(-2, -2)
            draw_rounded_rect(surface, color, rect, radius=6)
            draw_rounded_rect(surface, lerp_color(color, WHITE, 0.18), inner, radius=5)
            draw_rounded_rect(surface, lerp_color(color, BLACK, 0.45), rect, radius=6, width=2)

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

    def _draw_food(self, surface):
        col, row = self.food
        cx, cy = cell_center(col, row)
        pulse = 0.75 + 0.25 * math.sin(self.food_anim)
        r_outer = int(GRID_SIZE * 0.44 * pulse)
        r_mid = max(3, int(r_outer * 0.72))
        r_inner = max(2, int(r_outer * 0.38))

        draw_glow(surface, FOOD_COLOR, (cx, cy), r_outer + 10, layers=7)
        pygame.draw.circle(surface, FOOD_OUTLINE, (cx, cy), r_outer + 1)
        pygame.draw.circle(surface, FOOD_COLOR, (cx, cy), r_mid)
        pygame.draw.circle(surface, FOOD_INNER, (cx, cy), r_inner)

        shine_x = cx - r_outer // 3
        shine_y = cy - r_outer // 3
        pygame.draw.circle(surface, (255, 235, 240), (shine_x, shine_y), max(1, r_inner // 2))

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
        left_card = pygame.Rect(14, 10, 230, 58)
        draw_rounded_rect(surface, PANEL_BG, left_card, radius=12)
        draw_rounded_rect(surface, PANEL_BORDER, left_card, radius=12, width=2)

        title = self.font_med.render("SNAKE AI", True, NEON_CYAN)
        surface.blit(title, (26, 18))
        pygame.draw.line(surface, NEON_CYAN, (26, 48), (118, 48), 2)

        mode_surf = self.font_tiny.render("BEHAVIOR CLONING", True, TEXT_SOFT)
        surface.blit(mode_surf, (130, 20))

        score_card = pygame.Rect(WIDTH // 2 - 95, 10, 190, 58)
        draw_rounded_rect(surface, PANEL_BG, score_card, radius=12)
        draw_rounded_rect(surface, PANEL_BORDER, score_card, radius=12, width=2)

        score_label = self.font_tiny.render("SCORE", True, DIM_COLOR)
        surface.blit(score_label, (WIDTH // 2 - 28, 16))

        pulse = 1.0 + 0.12 * math.sin((16 - self.score_flash_timer) * 0.6) if self.score_flash_timer > 0 else 1.0
        score_font = pygame.font.SysFont("consolas", int(26 * pulse), bold=True)
        score_surf = score_font.render(f"{self.score:05d}", True, SCORE_COLOR if self.score_flash_timer > 0 else TEXT_COLOR)
        surface.blit(score_surf, (WIDTH // 2 - score_surf.get_width() // 2, 33))

        right_card = pygame.Rect(WIDTH - 160, 10, 146, 58)
        draw_rounded_rect(surface, PANEL_BG, right_card, radius=12)
        draw_rounded_rect(surface, PANEL_BORDER, right_card, radius=12, width=2)

        hi_label = self.font_tiny.render("BEST", True, DIM_COLOR)
        hi_val = self.font_med.render(f"{self.high_score:05d}", True, TEXT_COLOR)
        surface.blit(hi_label, (WIDTH - 128, 16))
        surface.blit(hi_val, (WIDTH - 128, 32))

        debug_y = GRID_OFFSET_Y + ROWS * GRID_SIZE + 8
        debug_card = pygame.Rect(10, debug_y - 4, WIDTH - 20, 106)
        draw_rounded_rect(surface, PANEL_BG, debug_card, radius=14)
        draw_rounded_rect(surface, PANEL_BORDER, debug_card, radius=14, width=2)

        line1 = self.font_small.render(
            f"ACTION: {self.last_action_name} | "
            f"P(st)={self.last_probs[0]:.2f}  "
            f"P(l)={self.last_probs[1]:.2f}  "
            f"P(r)={self.last_probs[2]:.2f}",
            True, TEXT_COLOR
        )
        surface.blit(line1, (22, debug_y + 6))

        line2 = self.font_small.render(
            f"DANGER: S={self.debug_features.get('danger_straight', 0)} "
            f"L={self.debug_features.get('danger_left', 0)} "
            f"R={self.debug_features.get('danger_right', 0)}",
            True, TEXT_SOFT
        )
        surface.blit(line2, (22, debug_y + 28))

        line3 = self.font_small.render(
            f"FOOD: U={self.debug_features.get('food_up', 0)} "
            f"D={self.debug_features.get('food_down', 0)} "
            f"L={self.debug_features.get('food_left', 0)} "
            f"R={self.debug_features.get('food_right', 0)} "
            f"| dx={self.debug_features.get('dx_food', 0)} "
            f"dy={self.debug_features.get('dy_food', 0)}",
            True, TEXT_SOFT
        )
        surface.blit(line3, (22, debug_y + 50))

        line4 = self.font_small.render(
            f"DIR: U={self.debug_features.get('dir_up', 0)} "
            f"D={self.debug_features.get('dir_down', 0)} "
            f"L={self.debug_features.get('dir_left', 0)} "
            f"R={self.debug_features.get('dir_right', 0)}",
            True, TEXT_SOFT
        )
        surface.blit(line4, (22, debug_y + 72))

        line5 = self.font_tiny.render(
            f"MODEL: {Path(MODEL_PATH).name} | exploration={EXPLORATION_RATE:.2f} | avoid_death={AVOID_DEATH}",
            True, DIM_COLOR
        )
        surface.blit(line5, (22, debug_y + 92))

    def _draw_death_overlay(self, surface):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 18, 185))
        surface.blit(overlay, (0, 0))

        card = pygame.Rect(WIDTH // 2 - 180, HEIGHT // 2 - 110, 360, 200)
        draw_rounded_rect(surface, (16, 22, 40), card, radius=18)
        draw_rounded_rect(surface, (70, 95, 150), card, radius=18, width=2)

        draw_glow(surface, FOOD_COLOR, (WIDTH // 2, HEIGHT // 2 - 50), 90, layers=8, alpha_scale=0.75)

        t1 = self.font_big.render("AI DIED", True, FOOD_COLOR)
        surface.blit(t1, t1.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 55)))

        t2 = self.font_med.render(f"Score: {self.score}", True, TEXT_COLOR)
        surface.blit(t2, t2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 5)))

        t3_text = "Auto restart..." if AUTO_RESTART else "Restart disabled"
        t3 = self.font_small.render(t3_text, True, TEXT_SOFT)
        surface.blit(t3, t3.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 62)))

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
            pygame.draw.rect(vignette, (0, 0, 0, alpha), rect, width=4, border_radius=20)
        surface.blit(vignette, (0, 0))


if __name__ == "__main__":
    SnakeGameAI().run()