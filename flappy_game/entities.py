import math
import random

import pygame

from .config import (
    BASE_HEIGHT,
    BIRD_START_X,
    FLAP_FORCE,
    GRAVITY,
    PIPE_GAP,
    PIPE_WIDTH,
    SCREEN_H,
    SCREEN_W,
)


class Bird:
    def __init__(self):
        self.x = float(BIRD_START_X)
        self.y = float(SCREEN_H // 2)
        self.vel = 0.0

        self.width = 30
        self.height = 28
        self.radius = 13

        self.rect = pygame.Rect(int(self.x), int(self.y), self.width, self.height)

        self.anim_time = 0.0
        self.flap_phase = 0.0
        self.bob_phase = 0.0
        self.visual_angle = 0.0
        self.last_flap_timer = 0

        self.body_color = (255, 220, 70)
        self.body_shadow = (230, 185, 50)
        self.belly_color = (255, 240, 170)
        self.wing_color = (245, 180, 55)
        self.beak_color = (255, 145, 45)
        self.outline_color = (120, 90, 20)
        self.eye_color = (25, 25, 25)

    def flap(self):
        self.vel = FLAP_FORCE
        self.last_flap_timer = 10
        self.flap_phase += 0.8

    def update(self):
        self.anim_time += 1.0
        self.bob_phase += 0.10

        self.vel += GRAVITY
        self.y += self.vel

        target_angle = max(-32, min(72, self.vel * 5.2))
        self.visual_angle += (target_angle - self.visual_angle) * 0.22

        if self.last_flap_timer > 0:
            self.last_flap_timer -= 1
            self.flap_phase += 0.40
        else:
            self.flap_phase += 0.18 if self.vel > 1 else 0.28

        self.rect.topleft = (int(self.x), int(self.y))

    def _draw_y(self):
        idle_bob = math.sin(self.bob_phase) * 1.5 if abs(self.vel) < 1.25 else 0.0
        return self.y + idle_bob

    def _build_surface(self):
        pad = 18
        surf_w = self.width + pad * 2
        surf_h = self.height + pad * 2
        surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)

        cx = surf_w // 2
        cy = surf_h // 2

        stretch = max(-0.05, min(0.08, self.vel * 0.012))
        body_w = int(self.width * (1.0 + stretch))
        body_h = int(self.height * (1.0 - stretch))

        shadow = pygame.Surface((body_w + 10, body_h + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 45), shadow.get_rect())
        surf.blit(shadow, (cx - shadow.get_width() // 2, cy - shadow.get_height() // 2 + 8))

        wing_up = math.sin(self.flap_phase)
        wing_angle = -35 * wing_up
        wing_surf = pygame.Surface((22, 16), pygame.SRCALPHA)
        pygame.draw.ellipse(wing_surf, self.wing_color, (0, 0, 22, 16))
        pygame.draw.ellipse(wing_surf, self.outline_color, (0, 0, 22, 16), 2)
        wing_rot = pygame.transform.rotate(wing_surf, wing_angle)
        surf.blit(wing_rot, (cx - 13 - wing_rot.get_width() // 2, cy - 1 - wing_rot.get_height() // 2))

        body_rect = pygame.Rect(0, 0, body_w, body_h)
        body_rect.center = (cx, cy)
        pygame.draw.ellipse(surf, self.body_color, body_rect)
        pygame.draw.ellipse(surf, self.outline_color, body_rect, 2)

        shade_rect = pygame.Rect(body_rect.x, body_rect.y + body_rect.h // 2, body_rect.w, body_rect.h // 2)
        pygame.draw.ellipse(surf, self.body_shadow, shade_rect)

        belly_rect = pygame.Rect(0, 0, int(body_w * 0.55), int(body_h * 0.42))
        belly_rect.center = (cx + 2, cy + 4)
        pygame.draw.ellipse(surf, self.belly_color, belly_rect)

        tail = [
            (cx - body_w // 2 - 7, cy + 1),
            (cx - body_w // 2 - 1, cy - 5),
            (cx - body_w // 2, cy + 7),
        ]
        pygame.draw.polygon(surf, (235, 170, 60), tail)
        pygame.draw.polygon(surf, self.outline_color, tail, 2)

        beak = [
            (cx + body_w // 2 - 1, cy),
            (cx + body_w // 2 + 10, cy - 3),
            (cx + body_w // 2 + 10, cy + 3),
        ]
        pygame.draw.polygon(surf, self.beak_color, beak)
        pygame.draw.polygon(surf, (140, 75, 20), beak, 1)

        eye_x = cx + body_w // 4
        eye_y = cy - body_h // 5
        pygame.draw.circle(surf, (255, 255, 255), (eye_x, eye_y), 4)
        pygame.draw.circle(surf, self.eye_color, (eye_x + 1, eye_y), 2)

        return surf

    def draw(self, surface):
        bird_surface = self._build_surface()
        rotated = pygame.transform.rotate(bird_surface, -self.visual_angle)
        rect = rotated.get_rect(center=(int(self.x + self.width / 2), int(self._draw_y() + self.height / 2)))
        surface.blit(rotated, rect.topleft)

    def get_mask_rect(self):
        collision_w = int(self.width * 0.82)
        collision_h = int(self.height * 0.78)
        offset_x = (self.width - collision_w) // 2
        offset_y = (self.height - collision_h) // 2
        return pygame.Rect(int(self.x + offset_x), int(self.y + offset_y), collision_w, collision_h)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap_center = random.randint(80, SCREEN_H - BASE_HEIGHT - 80)
        self.top_y = self.gap_center - PIPE_GAP // 2 - SCREEN_H
        self.bottom_y = self.gap_center + PIPE_GAP // 2
        self.passed = False

        self.body_color = (52, 168, 83)
        self.dark_color = (34, 110, 55)
        self.light_color = (92, 210, 120)
        self.cap_color = (42, 145, 70)

    def update(self, dx):
        self.x -= dx

    def _draw_pipe_body(self, surface, rect):
        pygame.draw.rect(surface, self.body_color, rect, border_radius=4)
        pygame.draw.rect(surface, self.dark_color, rect, 3, border_radius=4)

        shine = pygame.Rect(rect.x + 6, rect.y + 4, max(6, rect.w // 5), max(10, rect.h - 8))
        if shine.h > 0:
            pygame.draw.rect(surface, self.light_color, shine, border_radius=3)

    def _draw_pipe_cap(self, surface, x, y, w, h):
        cap_rect = pygame.Rect(x - 4, y, w + 8, h)
        pygame.draw.rect(surface, self.cap_color, cap_rect, border_radius=4)
        pygame.draw.rect(surface, self.dark_color, cap_rect, 3, border_radius=4)

    def draw(self, surface):
        top_rect = pygame.Rect(int(self.x), int(self.top_y), self.width, SCREEN_H)
        bottom_rect = pygame.Rect(int(self.x), int(self.bottom_y), self.width, SCREEN_H)

        self._draw_pipe_body(surface, top_rect)
        self._draw_pipe_body(surface, bottom_rect)

        top_cap_y = self.gap_center - PIPE_GAP // 2 - 18
        bottom_cap_y = self.gap_center + PIPE_GAP // 2

        self._draw_pipe_cap(surface, int(self.x), int(top_cap_y), self.width, 18)
        self._draw_pipe_cap(surface, int(self.x), int(bottom_cap_y), self.width, 18)


class Base:
    def __init__(self):
        self.y = SCREEN_H - BASE_HEIGHT
        self.offset = 0
        self.tile_w = 32

    def update(self, speed=3):
        self.offset = (self.offset + speed) % self.tile_w

    def draw(self, surface):
        base_rect = pygame.Rect(0, self.y, SCREEN_W, BASE_HEIGHT)
        pygame.draw.rect(surface, (222, 184, 135), base_rect)
        pygame.draw.line(surface, (190, 150, 105), (0, self.y), (SCREEN_W, self.y), 4)

        for x in range(-self.tile_w, SCREEN_W + self.tile_w, self.tile_w):
            draw_x = x - self.offset

            tile = pygame.Rect(draw_x, self.y + 6, self.tile_w, BASE_HEIGHT - 6)
            pygame.draw.rect(surface, (214, 176, 120), tile)

            stripe1 = pygame.Rect(draw_x + 4, self.y + 10, 8, BASE_HEIGHT - 14)
            stripe2 = pygame.Rect(draw_x + 18, self.y + 14, 7, BASE_HEIGHT - 18)

            pygame.draw.rect(surface, (196, 156, 102), stripe1, border_radius=2)
            pygame.draw.rect(surface, (234, 198, 145), stripe2, border_radius=2)