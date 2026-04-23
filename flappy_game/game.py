import math
import random
from collections import deque

import numpy as np
import pygame
from pygame.locals import K_ESCAPE, K_r, K_s, K_SPACE, K_UP, K_z, KEYDOWN, QUIT

from .ai_agent import AIAgent
from .config import (
    BASE_HEIGHT,
    DATA_DIR,
    FPS,
    MAX_ABS_VEL,
    NORMALIZE_STATE,
    PIPE_DISTANCE,
    PIPE_GAP,
    PIPE_WIDTH,
    RANDOM_SEED,
    RECORD_DEFAULT,
    SCREEN_H,
    SCREEN_W,
)
from .entities import Base, Bird, Pipe
from .recorder import Recorder


class FlappyGame:
    def __init__(self):
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)

        pygame.init()
        pygame.display.set_caption("Flappy (enhanced + AI)")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("arial", 18)
        self.font = pygame.font.SysFont("arial", 24, bold=True)
        self.font_big = pygame.font.SysFont("arial", 44, bold=True)

        self.recorder = Recorder(DATA_DIR)
        self.bird = Bird()
        self.base = Base()
        self.pipes = deque()

        self.scroll_speed = 3
        self.score = 0
        self.best_score = 0
        self.next_pipe_x = SCREEN_W + 50

        self.recording = RECORD_DEFAULT
        self.episode_id = 0
        self.episode_timestep = 0

        self.frame_count = 0
        self.score_anim_timer = 0
        self.hit_flash_timer = 0
        self.shake_timer = 0
        self.shake_strength = 0
        self.game_over_timer = 0

        self.ai_enabled = False
        self.ai_agent = None
        self.ai_last_prob = 0.0
        self.ai_threshold = 0.15

        # hybrid AI tuning
        self.ai_exploration_rate = 0.12
        self.ai_emergency_margin = 0.08
        self.ai_target_offset = 0.02
        self.ai_last_source = "none"

        self.clouds = self.create_clouds()

        try:
            self.ai_agent = AIAgent(
                model_path="flappy_game/data/flappy_model.pt",
                meta_path="flappy_game/data/flappy_model_meta.json",
                threshold=self.ai_threshold,
            )
            print("AI model loaded")
        except Exception as e:
            print(f"AI model not loaded: {e}")

        for index in range(3):
            self.pipes.append(Pipe(self.next_pipe_x + index * PIPE_DISTANCE))

    def create_clouds(self):
        clouds = []
        for _ in range(6):
            clouds.append(
                {
                    "x": random.randint(0, SCREEN_W),
                    "y": random.randint(40, SCREEN_H // 2),
                    "w": random.randint(70, 140),
                    "h": random.randint(28, 55),
                    "speed": random.uniform(0.35, 0.95),
                }
            )
        return clouds

    def reset_round(self):
        self.bird = Bird()
        self.base = Base()
        self.pipes.clear()
        self.next_pipe_x = SCREEN_W + 50
        for index in range(3):
            self.pipes.append(Pipe(self.next_pipe_x + index * PIPE_DISTANCE))
        self.best_score = max(self.best_score, self.score)
        self.score = 0
        self.game_over_timer = 24
        self.ai_last_source = "reset"

    def flush_recording(self):
        saved_count = self.recorder.flush()
        if saved_count > 0:
            print(f"Saved {saved_count} rows to {self.recorder.csv_path}")

    def start_recording(self):
        self.recording = True
        print("Recording started")

    def save_recording(self):
        saved_count = self.recorder.flush()
        if saved_count > 0:
            print(f"Saved {saved_count} rows to {self.recorder.csv_path}")
        else:
            print("Nothing to save")

    def stop_recording(self):
        self.recording = False
        print("Recording stopped")

    def next_pipe_ahead(self):
        for pipe in self.pipes:
            if pipe.x + pipe.width >= self.bird.x:
                return pipe
        return self.pipes[0]

    def build_state(self, pipe):
        dist_to_pipe_x = pipe.x - self.bird.x
        pipe_top_y = pipe.gap_center - PIPE_GAP // 2
        pipe_bottom_y = pipe.gap_center + PIPE_GAP // 2
        return [
            float(self.bird.x),
            float(self.bird.y),
            float(self.bird.vel),
            float(dist_to_pipe_x),
            float(pipe_top_y),
            float(pipe_bottom_y),
            float(pipe.gap_center),
        ]

    def normalize_state(self, state):
        if not NORMALIZE_STATE:
            return state

        return [
            state[0] / SCREEN_W,
            state[1] / SCREEN_H,
            max(-1.0, min(1.0, state[2] / MAX_ABS_VEL)),
            state[3] / SCREEN_W,
            state[4] / SCREEN_H,
            state[5] / SCREEN_H,
            state[6] / SCREEN_H,
        ]

    def build_ai_features(self, pipe):
        bird_center_y = self.bird.y + self.bird.height / 2
        dist_x = pipe.x - self.bird.x
        pipe_top_y = pipe.gap_center - PIPE_GAP // 2
        pipe_bottom_y = pipe.gap_center + PIPE_GAP // 2

        return [
            bird_center_y / SCREEN_H,
            max(-1.0, min(1.0, self.bird.vel / MAX_ABS_VEL)),
            dist_x / SCREEN_W,
            pipe_top_y / SCREEN_H,
            pipe_bottom_y / SCREEN_H,
            pipe.gap_center / SCREEN_H,
            (bird_center_y - pipe.gap_center) / SCREEN_H,
            (bird_center_y - pipe_top_y) / SCREEN_H,
            (pipe_bottom_y - bird_center_y) / SCREEN_H,
        ]

    def handle_event(self, event):
        action = 0
        if event.type == QUIT:
            return False, action
        if event.type != KEYDOWN:
            return True, action

        if event.key in (K_SPACE, K_UP):
            if not self.ai_enabled:
                self.bird.flap()
                action = 1
        elif event.key == K_ESCAPE:
            return False, action
        elif event.key == K_r:
            if self.recording:
                self.save_recording()
                self.stop_recording()
            else:
                self.start_recording()
        elif event.key == K_s:
            self.save_recording()
        elif event.key == K_z:
            removed_from = self.recorder.undo_last()
            if removed_from == "buffer":
                print("Removed last pending record")
            else:
                print("No record to remove")
            self.stop_recording()
        elif event.key == pygame.K_a:
            if self.ai_agent is None:
                print("AI model is not loaded")
            else:
                self.ai_enabled = not self.ai_enabled
                print(f"AI mode: {'ON' if self.ai_enabled else 'OFF'}")
        elif event.key == pygame.K_q:
            self.ai_threshold = max(0.05, self.ai_threshold - 0.05)
            if self.ai_agent is not None:
                self.ai_agent.threshold = self.ai_threshold
            print(f"AI threshold: {self.ai_threshold:.2f}")
        elif event.key == pygame.K_w:
            self.ai_threshold = min(0.95, self.ai_threshold + 0.05)
            if self.ai_agent is not None:
                self.ai_agent.threshold = self.ai_threshold
            print(f"AI threshold: {self.ai_threshold:.2f}")

        return True, action

    def get_ai_action(self):
        if not self.ai_enabled or self.ai_agent is None:
            return 0

        pipe = self.next_pipe_ahead()
        features = self.build_ai_features(pipe)

        action, prob = self.ai_agent.predict_action(features)
        self.ai_last_prob = prob

        bird_center_y = self.bird.y + self.bird.height / 2
        bird_center_y_norm = bird_center_y / SCREEN_H

        pipe_top_y = (pipe.gap_center - PIPE_GAP // 2) / SCREEN_H
        pipe_bottom_y = (pipe.gap_center + PIPE_GAP // 2) / SCREEN_H
        gap_center_y = pipe.gap_center / SCREEN_H

        # 1. emergency rescue
        if bird_center_y_norm > pipe_bottom_y - self.ai_emergency_margin:
            self.ai_last_source = "emergency"
            return 1

        # 2. small exploration
        if random.random() < self.ai_exploration_rate:
            target_y = gap_center_y + self.ai_target_offset

            if bird_center_y_norm > target_y:
                self.ai_last_source = "explore_target"
                return 1

            random_action = 1 if random.random() < 0.25 else 0
            self.ai_last_source = "explore_random"
            return random_action

        # 3. model decision
        self.ai_last_source = "model"
        return action

    def trigger_hit_effects(self):
        self.hit_flash_timer = 10
        self.shake_timer = 14
        self.shake_strength = 8

    def update_clouds(self):
        for cloud in self.clouds:
            cloud["x"] -= cloud["speed"]
            if cloud["x"] + cloud["w"] < 0:
                cloud["x"] = SCREEN_W + random.randint(20, 120)
                cloud["y"] = random.randint(40, SCREEN_H // 2)
                cloud["w"] = random.randint(70, 140)
                cloud["h"] = random.randint(28, 55)
                cloud["speed"] = random.uniform(0.35, 0.95)

    def update_effects(self):
        if self.score_anim_timer > 0:
            self.score_anim_timer -= 1
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1
        if self.shake_timer > 0:
            self.shake_timer -= 1
        if self.game_over_timer > 0:
            self.game_over_timer -= 1

    def update(self):
        self.frame_count += 1
        self.update_clouds()
        self.update_effects()

        self.bird.update()
        self.base.update(self.scroll_speed)

        for pipe in list(self.pipes):
            pipe.update(self.scroll_speed)

        if self.pipes and self.pipes[-1].x < SCREEN_W:
            self.pipes.append(Pipe(self.pipes[-1].x + PIPE_DISTANCE))

        if self.pipes and self.pipes[0].x + PIPE_WIDTH < -50:
            self.pipes.popleft()

        bird_rect = self.bird.get_mask_rect()
        hit = False

        for pipe in self.pipes:
            upper_rect = pygame.Rect(int(pipe.x), 0, pipe.width, pipe.gap_center - PIPE_GAP // 2)
            lower_rect = pygame.Rect(int(pipe.x), pipe.gap_center + PIPE_GAP // 2, pipe.width, SCREEN_H)

            if bird_rect.colliderect(upper_rect) or bird_rect.colliderect(lower_rect):
                hit = True
                break

            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                self.score_anim_timer = 18

        if self.bird.y + self.bird.height >= self.base.y or self.bird.y <= -10:
            hit = True

        if hit:
            self.trigger_hit_effects()

        return hit

    def get_camera_offset(self):
        if self.shake_timer <= 0:
            return 0, 0

        strength = max(1, int(self.shake_strength * (self.shake_timer / 14)))
        return random.randint(-strength, strength), random.randint(-strength, strength)

    def draw_gradient_background(self, surface):
        top = (120, 200, 255)
        bottom = (235, 248, 255)

        for y in range(SCREEN_H):
            t = y / SCREEN_H
            r = int(top[0] + (bottom[0] - top[0]) * t)
            g = int(top[1] + (bottom[1] - top[1]) * t)
            b = int(top[2] + (bottom[2] - top[2]) * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (SCREEN_W, y))

    def draw_sun(self, surface):
        pulse = math.sin(self.frame_count * 0.03) * 3
        radius = int(42 + pulse)
        pygame.draw.circle(surface, (255, 235, 150), (SCREEN_W - 90, 90), radius)
        pygame.draw.circle(surface, (255, 245, 190), (SCREEN_W - 90, 90), max(10, radius - 10))

    def draw_cloud(self, surface, x, y, w, h):
        cloud_surface = pygame.Surface((w + 20, h + 20), pygame.SRCALPHA)
        color = (255, 255, 255, 175)
        pygame.draw.ellipse(cloud_surface, color, (0, h // 4, w // 2, h))
        pygame.draw.ellipse(cloud_surface, color, (w // 4, 0, w // 2, h))
        pygame.draw.ellipse(cloud_surface, color, (w // 2, h // 4, w // 2, h))
        surface.blit(cloud_surface, (x, y))

    def draw_clouds(self, surface):
        for cloud in self.clouds:
            self.draw_cloud(surface, int(cloud["x"]), int(cloud["y"]), cloud["w"], cloud["h"])

    def draw_hud_panel(self, surface):
        panel = pygame.Surface((SCREEN_W - 16, 92), pygame.SRCALPHA)
        pygame.draw.rect(panel, (255, 255, 255, 145), panel.get_rect(), border_radius=16)
        pygame.draw.rect(panel, (255, 255, 255, 220), panel.get_rect(), width=2, border_radius=16)
        surface.blit(panel, (8, 8))

    def draw_text_shadow(self, surface, text, font, color, pos, shadow=(0, 0, 0), offset=(2, 2)):
        shadow_surf = font.render(text, True, shadow)
        text_surf = font.render(text, True, color)
        surface.blit(shadow_surf, (pos[0] + offset[0], pos[1] + offset[1]))
        surface.blit(text_surf, pos)

    def draw_hud(self, surface, state):
        self.draw_hud_panel(surface)

        pulse = 1.0
        if self.score_anim_timer > 0:
            pulse = 1.0 + 0.28 * math.sin((18 - self.score_anim_timer) * 0.65)

        score_font = pygame.font.SysFont("arial", int(28 * pulse), bold=True)
        self.draw_text_shadow(surface, f"Score: {self.score}", score_font, (25, 25, 25), (20, 20))
        self.draw_text_shadow(surface, f"Best: {self.best_score}", self.font, (60, 60, 60), (170, 22))

        if self.recording:
            rec_color = (220, 30, 30)
            pygame.draw.circle(surface, (255, 60, 60), (SCREEN_W - 165, 33), 7)
            rec_text = self.font.render("REC", True, rec_color)
            surface.blit(rec_text, (SCREEN_W - 150, 20))
        else:
            idle_text = self.font_small.render("R = record", True, (70, 70, 70))
            surface.blit(idle_text, (SCREEN_W - 140, 16))

        ai_text = f"AI: {'ON' if self.ai_enabled else 'OFF'}"
        if self.ai_enabled:
            ai_text += (
                f" | flap_p={self.ai_last_prob:.2f}"
                f" | thr={self.ai_threshold:.2f}"
                f" | src={self.ai_last_source}"
            )
        ai_surf = self.font_small.render(ai_text, True, (40, 40, 40))
        surface.blit(ai_surf, (20, 52))

        help_text = "A=AI  Q/W=threshold  SPACE=flap  R=record  S=save  Z=undo"
        help_surf = self.font_small.render(help_text, True, (55, 55, 55))
        surface.blit(help_surf, (20, 72))

    def draw_game_over_overlay(self, surface):
        if self.game_over_timer <= 0:
            return

        alpha = int(150 * (self.game_over_timer / 24))
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((20, 20, 30, alpha))
        surface.blit(overlay, (0, 0))

        title_alpha = min(255, alpha + 60)
        title = self.font_big.render("CRASH!", True, (255, 255, 255))
        subtitle = self.font.render("Restarting round...", True, (235, 235, 235))

        title.set_alpha(title_alpha)
        subtitle.set_alpha(title_alpha)

        surface.blit(title, (SCREEN_W // 2 - title.get_width() // 2, SCREEN_H // 2 - 50))
        surface.blit(subtitle, (SCREEN_W // 2 - subtitle.get_width() // 2, SCREEN_H // 2 + 5))

    def draw_hit_flash(self, surface):
        if self.hit_flash_timer <= 0:
            return

        alpha = int(110 * (self.hit_flash_timer / 10))
        flash = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        flash.fill((255, 255, 255, alpha))
        surface.blit(flash, (0, 0))

    def draw(self, state):
        world = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self.draw_gradient_background(world)
        self.draw_sun(world)
        self.draw_clouds(world)

        for pipe in self.pipes:
            pipe.draw(world)

        self.base.draw(world)
        self.bird.draw(world)

        self.draw_hud(world, state)
        self.draw_game_over_overlay(world)
        self.draw_hit_flash(world)

        shake_x, shake_y = self.get_camera_offset()

        self.screen.fill((0, 0, 0))
        self.screen.blit(world, (shake_x, shake_y))
        pygame.display.flip()

    def run(self):
        running = True

        while running:
            self.clock.tick(FPS)
            action = 0

            for event in pygame.event.get():
                running, event_action = self.handle_event(event)
                action = max(action, event_action)
                if not running:
                    break

            if not running:
                break

            if self.ai_enabled:
                ai_action = self.get_ai_action()
                if ai_action == 1:
                    self.bird.flap()
                action = max(action, ai_action)

            hit = self.update()
            next_pipe = self.next_pipe_ahead()
            state = self.build_state(next_pipe)
            state_norm = self.normalize_state(state)

            if self.recording:
                self.recorder.record_frame(
                    state=state,
                    state_norm=state_norm,
                    action=action,
                    episode_id=self.episode_id,
                    timestep=self.episode_timestep,
                    done=hit,
                )

            self.draw(state)

            if hit:
                self.episode_id += 1
                self.episode_timestep = 0
                self.reset_round()
            else:
                self.episode_timestep += 1

        if self.recording:
            self.flush_recording()

        pygame.quit()
        print("Goodbye")


def main():
    FlappyGame().run()


if __name__ == "__main__":
    main()