import random
from collections import deque

import numpy as np
import pygame
from pygame.locals import K_ESCAPE, K_r, K_s, K_SPACE, K_UP, K_z, KEYDOWN, QUIT

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
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Flappy (recording ready)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.recorder = Recorder(DATA_DIR)
        self.bird = Bird()
        self.base = Base()
        self.pipes = deque()
        self.scroll_speed = 3
        self.score = 0
        self.next_pipe_x = SCREEN_W + 50
        self.recording = RECORD_DEFAULT
        self.episode_id = 0
        self.episode_timestep = 0

        for index in range(3):
            self.pipes.append(Pipe(self.next_pipe_x + index * PIPE_DISTANCE))

    def reset_round(self):
        self.bird = Bird()
        self.pipes.clear()
        self.next_pipe_x = SCREEN_W + 50
        for index in range(3):
            self.pipes.append(Pipe(self.next_pipe_x + index * PIPE_DISTANCE))
        self.score = 0

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

    def handle_event(self, event):
        action = 0
        if event.type == QUIT:
            return False, action
        if event.type != KEYDOWN:
            return True, action

        if event.key in (K_SPACE, K_UP):
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

        return True, action

    def update(self):
        self.bird.update()
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

        if self.bird.y + self.bird.height >= self.base.y or self.bird.y <= -10:
            hit = True

        return hit

    def draw(self, state):
        self.screen.fill((135, 206, 235))
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.base.draw(self.screen)
        self.bird.draw(self.screen)

        info = f"Score: {self.score} | {'REC' if self.recording else '----'}"
        self.screen.blit(self.font.render(info, True, (0, 0, 0)), (8, 8))
        if self.recording:
            self.screen.blit(self.font.render("RECORDING (press R to stop)", True, (255, 0, 0)), (8, 28))

        dist_to_pipe_x = state[3]
        pipe_top_y = state[4]
        pipe_bottom_y = state[5]
        pipe_info = self.font.render(
            f"Next pipe x:{int(dist_to_pipe_x)} top:{int(pipe_top_y)} bot:{int(pipe_bottom_y)}",
            True,
            (0, 0, 0),
        )
        self.screen.blit(pipe_info, (8, 48))
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
