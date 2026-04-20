import random

import pygame

from .config import BASE_HEIGHT, BIRD_START_X, FLAP_FORCE, GRAVITY, PIPE_GAP, PIPE_WIDTH, SCREEN_H, SCREEN_W


class Bird:
    def __init__(self):
        self.x = BIRD_START_X
        self.y = SCREEN_H // 2
        self.vel = 0.0
        self.radius = 13
        self.width = 26
        self.height = 26
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def flap(self):
        self.vel = FLAP_FORCE

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel
        self.rect.topleft = (int(self.x), int(self.y))

    def draw(self, surface):
        center = (int(self.x + self.width // 2), int(self.y + self.height // 2))
        pygame.draw.circle(surface, (255, 255, 0), center, self.radius)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.x + self.width * 0.62), int(self.y + self.height * 0.36)), 2)

    def get_mask_rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap_center = random.randint(80, SCREEN_H - BASE_HEIGHT - 80)
        self.top_y = self.gap_center - PIPE_GAP // 2 - SCREEN_H
        self.bottom_y = self.gap_center + PIPE_GAP // 2
        self.passed = False

    def update(self, dx):
        self.x -= dx

    def draw(self, surface):
        top_rect = pygame.Rect(int(self.x), int(self.top_y), self.width, SCREEN_H)
        bottom_rect = pygame.Rect(int(self.x), int(self.bottom_y), self.width, SCREEN_H)
        pygame.draw.rect(surface, (34, 139, 34), top_rect)
        pygame.draw.rect(surface, (34, 139, 34), bottom_rect)


class Base:
    def __init__(self):
        self.y = SCREEN_H - BASE_HEIGHT

    def draw(self, surface):
        pygame.draw.rect(surface, (222, 184, 135), (0, self.y, SCREEN_W, BASE_HEIGHT))
