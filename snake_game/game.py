import csv
import math
import os
import random
import sys

import pygame

from .config import DATA_FILE

pygame.init()

# ── Константы ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 700
GRID_SIZE     = 20
COLS          = WIDTH  // GRID_SIZE
ROWS          = (HEIGHT - 120) // GRID_SIZE
GRID_OFFSET_Y = 80

FPS = 12

# ── Запись датасета ────────────────────────────────────────────────────────
RECORD_DATASET = True

# ── Цветовая палитра ────────────────────────────────────────────────────────
BG_DARK      = (8,   10,  18)
BG_GRID      = (14,  18,  32)
GRID_LINE    = (20,  26,  50)

SNAKE_HEAD   = (0,   255, 160)
SNAKE_BODY   = (0,   200, 120)
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
        pygame.draw.circle(glow_surf, (*color[:3], alpha),
                           (radius + 1, radius + 1), r)
    surface.blit(glow_surf,
                 (center[0] - radius - 1, center[1] - radius - 1))

def draw_rounded_rect(surface, color, rect, radius=6):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def cell_rect(col, row):
    x = col * GRID_SIZE
    y = row * GRID_SIZE + GRID_OFFSET_Y
    return pygame.Rect(x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2)

def cell_center(col, row):
    r = cell_rect(col, row)
    return r.centerx, r.centery


# ── Частицы ─────────────────────────────────────────────────────────────────

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


# ── Основная игра ────────────────────────────────────────────────────────────

class SnakeGame:
    def __init__(self):
        self.screen  = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SNAKE  ·  neon edition")
        self.clock   = pygame.time.Clock()

        # Шрифты
        self.font_big    = pygame.font.SysFont("consolas", 56, bold=True)
        self.font_med    = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_small  = pygame.font.SysFont("consolas", 18)
        self.font_title  = pygame.font.SysFont("consolas", 36, bold=True)

        self.high_score  = 0
        self.state       = "menu"    # menu | playing | dead
        self.particles   = []
        self.food_anim   = 0.0
        self.trail       = []

        # Для датасета
        self.recording = RECORD_DATASET
        self.data_file = str(DATA_FILE)
        self.pending_action = 0  # 0=straight, 1=left, 2=right

        self._init_dataset_file()
        self._reset()

    # ── Dataset helpers ─────────────────────────────────────────────────────
    def _init_dataset_file(self):
        if self.recording and not os.path.exists(self.data_file):
            with open(self.data_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "dir_up", "dir_down", "dir_left", "dir_right",
                    "food_up", "food_down", "food_left", "food_right",
                    "danger_straight", "danger_left", "danger_right",
                    "dx_food", "dy_food",
                    "action"
                ])

    def _log_sample(self, action):
        if not self.recording or self.state != "playing":
            return

        features = self._get_features()
        row = features + [action]

        with open(self.data_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ── Feature engineering ─────────────────────────────────────────────────
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
        self.pending_action = 0
        self._place_food()

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

            self._draw()

    # ── Ввод ────────────────────────────────────────────────────────────────
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if self.state == "menu":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.state = "playing"
                        self._reset()

                elif self.state == "playing":
                    DIR_MAP = {
                        pygame.K_UP:    (0, -1), pygame.K_w: (0, -1),
                        pygame.K_DOWN:  (0,  1), pygame.K_s: (0,  1),
                        pygame.K_LEFT:  (-1, 0), pygame.K_a: (-1, 0),
                        pygame.K_RIGHT: (1,  0), pygame.K_d: (1,  0),
                    }

                    if event.key in DIR_MAP:
                        nd = DIR_MAP[event.key]

                        if (nd[0] != -self.direction[0] or
                                nd[1] != -self.direction[1]):
                            action = self._direction_to_relative_action(nd)
                            if action is not None:
                                self.pending_action = action
                            self.next_direction = nd

                    if event.key == pygame.K_ESCAPE:
                        self.state = "menu"

                elif self.state == "dead":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_r):
                        self.state = "playing"
                        self._reset()
                    if event.key == pygame.K_ESCAPE:
                        self.state = "menu"

    # ── Логика ──────────────────────────────────────────────────────────────
    def _update(self):
        self.food_anim = (self.food_anim + 0.12) % math.tau

        # Логируем состояние + действие игрока ДО шага
        self._log_sample(self.pending_action)

        # Сбрасываем действие по умолчанию на straight
        self.pending_action = 0

        # Обновление частиц
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # Движение змейки
        self.direction = self.next_direction
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = ((hx + dx) % COLS, (hy + dy) % ROWS)

        # Самостолкновение
        if new_head in self.snake:
            self._spawn_death_particles()
            if self.score > self.high_score:
                self.high_score = self.score
            self.state = "dead"
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

        if self.state == "menu":
            self._draw_menu()
        elif self.state in ("playing", "dead"):
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
            pygame.draw.line(self.screen, GRID_LINE,
                             (x, GRID_OFFSET_Y),
                             (x, GRID_OFFSET_Y + ROWS * GRID_SIZE))
        for r in range(0, ROWS + 1):
            y = r * GRID_SIZE + GRID_OFFSET_Y
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WIDTH, y))

        pygame.draw.rect(self.screen, (30, 40, 80),
                         field_rect, 2, border_radius=4)

    def _draw_snake(self):
        n = len(self.snake)
        for i, (col, row) in enumerate(self.snake):
            t = i / max(n - 1, 1)
            color = lerp_color(SNAKE_HEAD, SNAKE_TAIL, t)
            rect  = cell_rect(col, row)

            if i == 0:
                draw_glow(self.screen, SNAKE_HEAD, rect.center, GRID_SIZE, layers=5)

            draw_rounded_rect(self.screen, color, rect, radius=5)

            if i == 0:
                shine = pygame.Rect(rect.x + 3, rect.y + 3, 5, 5)
                pygame.draw.ellipse(self.screen, (200, 255, 230), shine)

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
        cx, cy   = cell_center(col, row)
        pulse    = 0.7 + 0.3 * math.sin(self.food_anim)
        r_outer  = int(GRID_SIZE * 0.42 * pulse)
        r_inner  = max(2, int(r_outer * 0.45))

        draw_glow(self.screen, FOOD_COLOR, (cx, cy), r_outer + 6, layers=7)
        pygame.draw.circle(self.screen, FOOD_COLOR, (cx, cy), r_outer)
        pygame.draw.circle(self.screen, FOOD_INNER, (cx, cy), r_inner)

        shine_x = cx - r_outer // 3
        shine_y = cy - r_outer // 3
        pygame.draw.circle(self.screen, (255, 220, 225),
                           (shine_x, shine_y), max(1, r_inner // 2))

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
        pygame.draw.line(self.screen, (30, 50, 100), (0, GRID_OFFSET_Y - 1),
                         (WIDTH, GRID_OFFSET_Y - 1))

        title_surf = self.font_title.render("SNAKE", True, SNAKE_HEAD)
        self.screen.blit(title_surf, (20, 18))

        pygame.draw.line(self.screen, SNAKE_HEAD,
                         (20, 52), (20 + title_surf.get_width(), 52), 2)

        score_label = self.font_small.render("SCORE", True, DIM_COLOR)
        score_val   = self.font_med.render(f"{self.score:05d}", True, SCORE_COLOR)
        self.screen.blit(score_label, (WIDTH // 2 - 40, 12))
        self.screen.blit(score_val,   (WIDTH // 2 - 40, 30))

        hi_label = self.font_small.render("BEST", True, DIM_COLOR)
        hi_val   = self.font_med.render(f"{self.high_score:05d}", True, TEXT_COLOR)
        self.screen.blit(hi_label, (WIDTH - 130, 12))
        self.screen.blit(hi_val,   (WIDTH - 130, 30))

        ctrl = self.font_small.render("WASD / ← ↑ → ↓   ESC — меню", True, DIM_COLOR)
        y_bot = GRID_OFFSET_Y + ROWS * GRID_SIZE + 8
        self.screen.blit(ctrl, (WIDTH // 2 - ctrl.get_width() // 2, y_bot))

    def _draw_death_overlay(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 18, 200))
        self.screen.blit(overlay, (0, 0))

        t1 = self.font_big.render("GAME OVER", True, FOOD_COLOR)
        draw_glow(self.screen, FOOD_COLOR,
                  (WIDTH // 2, HEIGHT // 2 - 60), 80, layers=8)
        self.screen.blit(t1, t1.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))

        t2 = self.font_med.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(t2, t2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 10)))

        if self.score >= self.high_score and self.score > 0:
            t3 = self.font_med.render("New best!", True, SNAKE_HEAD)
            self.screen.blit(t3, t3.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 48)))

        t4 = self.font_small.render("SPACE / R — заново     ESC — меню",
                                    True, DIM_COLOR)
        self.screen.blit(t4, t4.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100)))

    def _draw_menu(self):
        t = pygame.time.get_ticks() / 1000
        for i in range(30):
            x = (i * 127 + int(t * 20 * (1 + i % 3))) % WIDTH
            y = (i * 83  + int(t * 12 * (1 + i % 5))) % HEIGHT
            a = int(40 + 30 * math.sin(t + i))
            r = 2 + i % 4
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*SNAKE_HEAD, a), (r, r), r)
            self.screen.blit(s, (x - r, y - r))

        cy = HEIGHT // 2 - 80
        draw_glow(self.screen, SNAKE_HEAD, (WIDTH // 2, cy), 120, layers=10)

        logo = self.font_big.render("SNAKE", True, SNAKE_HEAD)
        sub  = self.font_small.render("n e o n  e d i t i o n", True, DIM_COLOR)
        self.screen.blit(logo, logo.get_rect(center=(WIDTH // 2, cy)))
        self.screen.blit(sub,  sub.get_rect(center=(WIDTH // 2, cy + 54)))

        pygame.draw.line(self.screen, (30, 60, 100),
                         (WIDTH // 2 - 160, cy + 72),
                         (WIDTH // 2 + 160, cy + 72), 1)

        pulse = 0.7 + 0.3 * math.sin(t * 3)
        col   = tuple(int(c * pulse) for c in SNAKE_HEAD)
        start = self.font_med.render("[ PRESS SPACE ]", True, col)
        self.screen.blit(start, start.get_rect(center=(WIDTH // 2, cy + 120)))

        if self.high_score:
            hi = self.font_small.render(f"Best: {self.high_score}", True, DIM_COLOR)
            self.screen.blit(hi, hi.get_rect(center=(WIDTH // 2, cy + 165)))

        ctrl = self.font_small.render("WASD / Arrow keys", True, DIM_COLOR)
        self.screen.blit(ctrl, ctrl.get_rect(center=(WIDTH // 2, HEIGHT - 40)))


if __name__ == "__main__":
    SnakeGame().run()