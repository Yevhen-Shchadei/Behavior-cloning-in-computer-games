import math
import random
import sys

import pygame
import torch
import torch.nn as nn

from .config import BIG_MODEL_PATH, SMALL_MODEL_PATH

pygame.init()

# ── Константы ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 760
GRID_SIZE     = 20
COLS          = WIDTH  // GRID_SIZE
ROWS          = (HEIGHT - 180) // GRID_SIZE
GRID_OFFSET_Y = 80

FPS = 12

# ── AI настройки ───────────────────────────────────────────────────────────
MODEL_PATH = SMALL_MODEL_PATH   # or BIG_MODEL_PATH
AUTO_RESTART = True
RESTART_DELAY_MS = 1200

# Exploration / heuristic control
EXPLORATION_RATE = 0.20   # 20% ходів будуть не чисто по моделі
AVOID_DEATH = True        # не обирати гарантовану смерть, якщо є альтернатива

# ── Цветовая палитра ────────────────────────────────────────────────────────
BG_DARK      = (8,   10,  18)
BG_GRID      = (14,  18,  32)
GRID_LINE    = (20,  26,  50)

SNAKE_HEAD   = (0,   255, 160)
SNAKE_TAIL   = (0,   140,  80)

FOOD_COLOR   = (255,  60,  80)
FOOD_INNER   = (255, 180, 190)

SCORE_COLOR  = (0,   255, 160)
TEXT_COLOR   = (180, 200, 230)
DIM_COLOR    = (60,   80, 120)


# ── Вспомогательные функции ─────────────────────────────────────────────────

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_glow(surface, color, center, radius, layers=6):
    glow_surf = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        alpha = int(80 * (i / layers) ** 2)
        r = int(radius * i / layers)
        pygame.draw.circle(glow_surf, (*color[:3], alpha), (radius + 1, radius + 1), r)
    surface.blit(glow_surf, (center[0] - radius - 1, center[1] - radius - 1))


def draw_rounded_rect(surface, color, rect, radius=6):
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def cell_rect(col, row):
    x = col * GRID_SIZE
    y = row * GRID_SIZE + GRID_OFFSET_Y
    return pygame.Rect(x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2)


def cell_center(col, row):
    r = cell_rect(col, row)
    return r.centerx, r.centery


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, math.tau)
        speed = random.uniform(1.5, 5)
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
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color[:3], alpha), (r, r), r)
        surface.blit(s, (int(self.x) - r, int(self.y) - r))


# ── Модели ─────────────────────────────────────────────────────────────────

class SmallNet(nn.Module):
    def __init__(self, input_dim=13, hidden=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class BiggerNet(nn.Module):
    def __init__(self, input_dim=13, h1=32, h2=16, num_classes=3):
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


# ── Основная игра ────────────────────────────────────────────────────────────

class SnakeGameAI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SNAKE AI · behavior cloning + exploration")
        self.clock = pygame.time.Clock()

        self.font_big   = pygame.font.SysFont("consolas", 56, bold=True)
        self.font_med   = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 18)
        self.font_title = pygame.font.SysFont("consolas", 30, bold=True)

        self.high_score = 0
        self.state = "playing"
        self.particles = []
        self.food_anim = 0.0
        self.trail = []
        self.death_time = None

        # debug state
        self.last_action_name = "none"
        self.last_probs = [0.0, 0.0, 0.0]
        self.debug_features = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(MODEL_PATH)

        self._reset()

    # ── AI helpers ─────────────────────────────────────────────────────────
    def _load_model(self, path):
        if "bigger" in path.lower():
            model = BiggerNet()
        else:
            model = SmallNet()

        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
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

        # Через wrap-around це не ідеальна метрика, але працює нормально
        return abs(fx - nx) + abs(fy - ny)

    def _get_features(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        dx, dy = self.direction

        dir_up    = 1 if (dx, dy) == (0, -1) else 0
        dir_down  = 1 if (dx, dy) == (0,  1) else 0
        dir_left  = 1 if (dx, dy) == (-1, 0) else 0
        dir_right = 1 if (dx, dy) == (1,  0) else 0

        food_up    = 1 if fy < hy else 0
        food_down  = 1 if fy > hy else 0
        food_left  = 1 if fx < hx else 0
        food_right = 1 if fx > hx else 0

        straight_dir = self.direction
        left_dir     = self._turn_left(self.direction)
        right_dir    = self._turn_right(self.direction)

        straight_pos = self._next_pos((hx, hy), straight_dir)
        left_pos     = self._next_pos((hx, hy), left_dir)
        right_pos    = self._next_pos((hx, hy), right_dir)

        body = self.snake[1:]

        danger_straight = 1 if straight_pos in body else 0
        danger_left     = 1 if left_pos in body else 0
        danger_right    = 1 if right_pos in body else 0

        dx_food = (fx - hx) / COLS
        dy_food = (fy - hy) / ROWS

        return [
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right,
            danger_straight, danger_left, danger_right,
            dx_food, dy_food
        ]

    def _predict_action(self):
        features = self._get_features()
        x = torch.tensor([features], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
            model_action = torch.argmax(logits, dim=1).item()

        safe_actions = self._get_safe_actions()

        # Exploration
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
                    action = max(safe_actions, key=lambda a: probs[a])
                    source = "safe_override"

        action_names = {
            0: "straight",
            1: "left",
            2: "right"
        }

        self.last_action_name = f"{action_names.get(action, 'unknown')} [{source}]"
        self.last_probs = probs

        self.debug_features = {
            "dir_up": features[0],
            "dir_down": features[1],
            "dir_left": features[2],
            "dir_right": features[3],
            "food_up": features[4],
            "food_down": features[5],
            "food_left": features[6],
            "food_right": features[7],
            "danger_straight": features[8],
            "danger_left": features[9],
            "danger_right": features[10],
            "dx_food": round(features[11], 3),
            "dy_food": round(features[12], 3),
        }

        return action

    # ── Инициализация/сброс ──────────────────────────────────────────────────
    def _reset(self):
        mid_col = COLS // 2
        mid_row = ROWS // 2
        self.snake = [
            (mid_col,     mid_row),
            (mid_col - 1, mid_row),
            (mid_col - 2, mid_row),
        ]
        self.direction      = (1, 0)
        self.next_direction = (1, 0)
        self.score          = 0
        self.particles      = []
        self.trail          = []
        self.death_time     = None
        self._place_food()

        self.last_action_name = "reset"
        self.last_probs = [0.0, 0.0, 0.0]
        self.debug_features = {}

    def _place_food(self):
        empty = [(c, r) for c in range(COLS) for r in range(ROWS)
                 if (c, r) not in self.snake]
        self.food = random.choice(empty) if empty else (0, 0)

    # ── Главный цикл ────────────────────────────────────────────────────────
    def run(self):
        while True:
            self.clock.tick(FPS)
            self._handle_events()

            if self.state == "playing":
                self._update()
            elif self.state == "dead" and AUTO_RESTART:
                now = pygame.time.get_ticks()
                if self.death_time is not None and now - self.death_time >= RESTART_DELAY_MS:
                    self.state = "playing"
                    self._reset()

            self._draw()

    # ── Ввод ────────────────────────────────────────────────────────────────
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    # ── Логика ──────────────────────────────────────────────────────────────
    def _update(self):
        self.food_anim = (self.food_anim + 0.12) % math.tau

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
            return

        tail_pos = self.snake[-1]
        self.trail.append((*cell_center(*tail_pos), 0.6))

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 10
            self._spawn_eat_particles(new_head)
            self._place_food()
        else:
            self.snake.pop()

    def _spawn_eat_particles(self, cell):
        cx, cy = cell_center(*cell)
        for _ in range(22):
            self.particles.append(Particle(cx, cy, FOOD_COLOR))

    def _spawn_death_particles(self):
        for seg in self.snake:
            cx, cy = cell_center(*seg)
            for _ in range(5):
                self.particles.append(Particle(cx, cy, SNAKE_HEAD))

    # ── Рисование ───────────────────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG_DARK)
        self._draw_grid()
        self._draw_trail()
        self._draw_food()
        self._draw_snake()
        self._draw_particles()
        self._draw_hud()

        if self.state == "dead":
            self._draw_death_overlay()

        pygame.display.flip()

    def _draw_grid(self):
        field_rect = pygame.Rect(0, GRID_OFFSET_Y, WIDTH, ROWS * GRID_SIZE)
        pygame.draw.rect(self.screen, BG_GRID, field_rect)

        for c in range(0, COLS + 1):
            x = c * GRID_SIZE
            pygame.draw.line(
                self.screen, GRID_LINE,
                (x, GRID_OFFSET_Y),
                (x, GRID_OFFSET_Y + ROWS * GRID_SIZE)
            )

        for r in range(0, ROWS + 1):
            y = r * GRID_SIZE + GRID_OFFSET_Y
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WIDTH, y))

        pygame.draw.rect(self.screen, (30, 40, 80), field_rect, 2, border_radius=4)

    def _draw_snake(self):
        n = len(self.snake)
        for i, (col, row) in enumerate(self.snake):
            t = i / max(n - 1, 1)
            color = lerp_color(SNAKE_HEAD, SNAKE_TAIL, t)
            rect = cell_rect(col, row)

            if i == 0:
                draw_glow(self.screen, SNAKE_HEAD, rect.center, GRID_SIZE, layers=5)

            draw_rounded_rect(self.screen, color, rect, radius=5)

            if i == 0:
                dx, dy = self.direction
                cx, cy = rect.centerx, rect.centery
                e1x = cx + dx * 4 + dy * 3
                e1y = cy + dy * 4 - dx * 3
                e2x = cx + dx * 4 - dy * 3
                e2y = cy + dy * 4 + dx * 3
                pygame.draw.circle(self.screen, (0, 0, 0), (e1x, e1y), 3)
                pygame.draw.circle(self.screen, (0, 0, 0), (e2x, e2y), 3)
                pygame.draw.circle(self.screen, (255, 255, 255), (e1x, e1y), 1)
                pygame.draw.circle(self.screen, (255, 255, 255), (e2x, e2y), 1)

    def _draw_food(self):
        col, row = self.food
        cx, cy = cell_center(col, row)
        pulse = 0.7 + 0.3 * math.sin(self.food_anim)
        r_outer = int(GRID_SIZE * 0.42 * pulse)
        r_inner = max(2, int(r_outer * 0.45))

        draw_glow(self.screen, FOOD_COLOR, (cx, cy), r_outer + 6, layers=7)
        pygame.draw.circle(self.screen, FOOD_COLOR, (cx, cy), r_outer)
        pygame.draw.circle(self.screen, FOOD_INNER, (cx, cy), r_inner)

        shine_x = cx - r_outer // 3
        shine_y = cy - r_outer // 3
        pygame.draw.circle(self.screen, (255, 220, 225), (shine_x, shine_y), max(1, r_inner // 2))

    def _draw_trail(self):
        self.trail = [(x, y, a - 0.08) for x, y, a in self.trail if a > 0]
        for x, y, alpha in self.trail:
            if alpha <= 0:
                continue
            r = max(1, int(4 * alpha))
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*SNAKE_TAIL, int(alpha * 120)), (r, r), r)
            self.screen.blit(s, (int(x) - r, int(y) - r))

    def _draw_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _draw_hud(self):
        hud_rect = pygame.Rect(0, 0, WIDTH, GRID_OFFSET_Y)
        pygame.draw.rect(self.screen, (10, 13, 26), hud_rect)
        pygame.draw.line(self.screen, (30, 50, 100), (0, GRID_OFFSET_Y - 1), (WIDTH, GRID_OFFSET_Y - 1))

        title_surf = self.font_title.render("SNAKE AI DEBUG", True, SNAKE_HEAD)
        self.screen.blit(title_surf, (20, 20))

        score_label = self.font_small.render("SCORE", True, DIM_COLOR)
        score_val = self.font_med.render(f"{self.score:05d}", True, SCORE_COLOR)
        self.screen.blit(score_label, (WIDTH // 2 - 40, 10))
        self.screen.blit(score_val, (WIDTH // 2 - 40, 28))

        hi_label = self.font_small.render("BEST", True, DIM_COLOR)
        hi_val = self.font_med.render(f"{self.high_score:05d}", True, TEXT_COLOR)
        self.screen.blit(hi_label, (WIDTH - 130, 10))
        self.screen.blit(hi_val, (WIDTH - 130, 28))

        debug_y = GRID_OFFSET_Y + ROWS * GRID_SIZE + 8

        line1 = self.font_small.render(
            f"ACT: {self.last_action_name} | "
            f"P(st)={self.last_probs[0]:.2f}  "
            f"P(l)={self.last_probs[1]:.2f}  "
            f"P(r)={self.last_probs[2]:.2f}",
            True, TEXT_COLOR
        )
        self.screen.blit(line1, (20, debug_y))

        line2 = self.font_small.render(
            f"DANGER: S={self.debug_features.get('danger_straight', 0)} "
            f"L={self.debug_features.get('danger_left', 0)} "
            f"R={self.debug_features.get('danger_right', 0)}",
            True, DIM_COLOR
        )
        self.screen.blit(line2, (20, debug_y + 22))

        line3 = self.font_small.render(
            f"FOOD: U={self.debug_features.get('food_up', 0)} "
            f"D={self.debug_features.get('food_down', 0)} "
            f"L={self.debug_features.get('food_left', 0)} "
            f"R={self.debug_features.get('food_right', 0)} "
            f"| dx={self.debug_features.get('dx_food', 0)} "
            f"dy={self.debug_features.get('dy_food', 0)}",
            True, DIM_COLOR
        )
        self.screen.blit(line3, (20, debug_y + 44))

        line4 = self.font_small.render(
            f"DIR: U={self.debug_features.get('dir_up', 0)} "
            f"D={self.debug_features.get('dir_down', 0)} "
            f"L={self.debug_features.get('dir_left', 0)} "
            f"R={self.debug_features.get('dir_right', 0)}",
            True, DIM_COLOR
        )
        self.screen.blit(line4, (20, debug_y + 66))

        line5 = self.font_small.render(
            f"MODEL: {MODEL_PATH.name} | exploration={EXPLORATION_RATE:.2f} | avoid_death={AVOID_DEATH}",
            True, DIM_COLOR
        )
        self.screen.blit(line5, (20, debug_y + 88))

    def _draw_death_overlay(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 18, 200))
        self.screen.blit(overlay, (0, 0))

        t1 = self.font_big.render("AI DIED", True, FOOD_COLOR)
        self.screen.blit(t1, t1.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))

        t2 = self.font_med.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(t2, t2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 10)))

        t3_text = "Auto restart..." if AUTO_RESTART else "Restart disabled"
        t3 = self.font_small.render(t3_text, True, DIM_COLOR)
        self.screen.blit(t3, t3.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100)))


if __name__ == "__main__":
    SnakeGameAI().run()