import math
import numpy as np
import pygame

black = (0, 0, 0)
white = (255, 255, 255)


class Pendulum:
    def __init__(self, pivot_x, pivot_y, radius, m=1, l=200, a=math.pi / 2, g=1, color='blue'):
        self.pivot = (pivot_x, pivot_y)
        self.radius = radius
        self.m = m
        self.l = l
        self.a = a
        self.g = g
        self.clr = color

        self.x = 0
        self.y = 0
        self.av = 0  # Angular Velocity
        self.trajectory = []

    def step(self):
        acc = (-self.g / self.l) * math.sin(self.a)
        self.av += acc
        self.av *= 0.99  # damping factor to simulate air resistance/friction
        self.a += self.av
        self.x = self.pivot[0] + self.l * math.sin(self.a)
        self.y = self.pivot[1] + self.l * math.cos(self.a)

    def draw(self, surface):
        pygame.draw.line(surface, white, self.pivot, (self.x, self.y))
        pygame.draw.circle(surface, self.clr, (self.x, self.y), self.radius)
