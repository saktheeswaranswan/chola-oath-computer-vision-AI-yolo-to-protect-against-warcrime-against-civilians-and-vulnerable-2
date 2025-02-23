import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math, random

# -------------------- Configuration --------------------
window_width = 1280
window_height = 720

inner_dome_radius = 300      # Inner dome radius (video-textured, transparent)
outer_dome_radius = 320      # Outer dome radius (extruded, patch-rendered, transparent)

# -------------------- Global Variables --------------------
cap = cv2.VideoCapture('videoplayback.mp4')
texture_id = None  # for video texture

dome_rotation = 0.0
rotating = False
prev_mouse_x = 0
zoom_factor = 3.0

camera_yaw = 0.0
camera_pitch = 0.0

projectile = None   # current projectile instance

# -------------------- Projectile Class --------------------
class Projectile:
    def __init__(self, pos, speed):
        self.pos = np.array(pos, dtype=float)
        # Constant vertical downward velocity
        self.vel = np.array([0, -speed, 0], dtype=float)
        self.active = True
        self.trail = [tuple(self.pos)]
    
    def update(self, dt):
        self.pos += self.vel * dt
        self.trail.append(tuple(self.pos))
        # Deactivate once the projectile reaches the base (y <= 0)
        if self.pos[1] <= 0:
            self.active = False

def spawn_projectile():
    # Spawn from a random position on the top of the outer dome.
    # Here the top is defined as y = outer_dome_radius + margin.
    margin = 10
    y = outer_dome_radius + margin
    # Choose a random horizontal position within the dome's top circle.
    r_spawn = random.uniform(0, outer_dome_radius)
    angle = random.uniform(0, 2 * math.pi)
    x = r_spawn * math.cos(angle)
    z = r_spawn * math.sin(angle)
    speed = 100  # constant downward speed (units per second)
    return Projectile([x, y, z], speed)

# -------------------- Texture Loading --------------------
def load_texture():
    global texture_id, cap
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return False
    frame = cv2.flip(frame, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_data = frame.tobytes()
    if texture_id is None:
        texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0],
                 0, GL_RGB, GL_UNSIGNED_BYTE, frame_data)
    return True

# -------------------- Dome Drawing Functions --------------------
def draw_inner_dome(radius, slices=100):
    """Draws the inner dome as a flat disc with the video texture and 50% transparency."""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(1, 1, 1, 0.5)  # white at 50% opacity
    glBegin(GL_TRIANGLE_FAN)
    glTexCoord2f(0.5, 0.5)
    glVertex3f(0, 0, 0)
    for i in range(slices + 1):
        angle = 2 * math.pi * i / slices
        tx, ty = 0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)
        glTexCoord2f(tx, ty)
        glVertex3f(radius * math.cos(angle), 0, radius * math.sin(angle))
    glEnd()
    glDisable(GL_BLEND)

def draw_outer_dome(radius, lat_steps=20, lon_steps=40):
    """Draws the outer dome as a transparent hemisphere composed of square patches."""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.7, 0.7, 0.7, 0.3)  # light grey with high transparency
    for i in range(lat_steps):
        theta1 = (i / lat_steps) * (math.pi / 2)
        theta2 = ((i + 1) / lat_steps) * (math.pi / 2)
        for j in range(lon_steps):
            phi1 = (j / lon_steps) * 2 * math.pi
            phi2 = ((j + 1) / lon_steps) * 2 * math.pi
            # Calculate quad vertices using spherical coordinates
            x1 = radius * math.sin(theta1) * math.cos(phi1)
            y1 = radius * math.cos(theta1)
            z1 = radius * math.sin(theta1) * math.sin(phi1)

            x2 = radius * math.sin(theta1) * math.cos(phi2)
            y2 = radius * math.cos(theta1)
            z2 = radius * math.sin(theta1) * math.sin(phi2)

            x3 = radius * math.sin(theta2) * math.cos(phi2)
            y3 = radius * math.cos(theta2)
            z3 = radius * math.sin(theta2) * math.sin(phi2)

            x4 = radius * math.sin(theta2) * math.cos(phi1)
            y4 = radius * math.cos(theta2)
            z4 = radius * math.sin(theta2) * math.sin(phi1)

            glBegin(GL_QUADS)
            glVertex3f(x1, y1, z1)
            glVertex3f(x2, y2, z2)
            glVertex3f(x3, y3, z3)
            glVertex3f(x4, y4, z4)
            glEnd()
    glDisable(GL_BLEND)

def draw_red_trail(trail):
    """Draws the thick red trail left by the projectile."""
    glColor3f(1, 0, 0)
    glLineWidth(8)  # thick line
    glBegin(GL_LINE_STRIP)
    for pt in trail:
        glVertex3f(pt[0], pt[1], pt[2])
    glEnd()

def draw_projectile(proj):
    """Draws the projectile as a small white point."""
    glPointSize(12)
    glColor3f(1, 1, 1)
    glBegin(GL_POINTS)
    glVertex3f(proj.pos[0], proj.pos[1], proj.pos[2])
    glEnd()

# -------------------- Scene Rendering --------------------
def render_scene():
    global dome_rotation, camera_yaw, camera_pitch
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Set up the camera (perspective is based on the inner dome radius)
    r = inner_dome_radius * zoom_factor
    center_y = inner_dome_radius / 2.0
    cam_x = r * math.cos(camera_pitch) * math.cos(camera_yaw)
    cam_y = center_y + r * math.sin(camera_pitch)
    cam_z = r * math.cos(camera_pitch) * math.sin(camera_yaw)
    gluLookAt(cam_x, cam_y, cam_z, 0, center_y, 0, 0, 1, 0)

    glPushMatrix()
    glRotatef(math.degrees(dome_rotation), 0, 1, 0)

    # Draw the outer transparent dome
    draw_outer_dome(outer_dome_radius)
    
    # Draw the inner dome with the video texture (50% transparent)
    if load_texture():
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        draw_inner_dome(inner_dome_radius)
        glDisable(GL_TEXTURE_2D)
    
    glPopMatrix()

    # Draw the projectile's thick red trail and the projectile itself
    if projectile is not None:
        draw_red_trail(projectile.trail)
        if projectile.active:
            draw_projectile(projectile)

# -------------------- Main Loop --------------------
def main():
    global dome_rotation, rotating, prev_mouse_x, zoom_factor, camera_yaw, camera_pitch, projectile
    pygame.init()
    pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Transparent Dome with Vertical Projectile Trajectory")
    glEnable(GL_DEPTH_TEST)
    glClearColor(0, 0, 0, 1)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, window_width/window_height, 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)

    clock = pygame.time.Clock()
    projectile = spawn_projectile()  # spawn the first projectile
    spawn_timer = 0
    running = True
    while running:
        dt = clock.tick(30) / 1000.0  # delta time in seconds
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                rotating = True
                prev_mouse_x = event.pos[0]
            elif event.type == MOUSEBUTTONUP:
                rotating = False
            elif event.type == MOUSEMOTION and rotating:
                dx = event.pos[0] - prev_mouse_x
                dome_rotation += dx * 0.005
                prev_mouse_x = event.pos[0]
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = max(0.5, min(5.0, zoom_factor - event.y * 0.1))
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    camera_yaw -= 0.05
                elif event.key == K_RIGHT:
                    camera_yaw += 0.05
                elif event.key == K_UP:
                    camera_pitch = min(camera_pitch + 0.05, math.pi / 2 - 0.1)
                elif event.key == K_DOWN:
                    camera_pitch = max(camera_pitch - 0.05, -math.pi / 2 + 0.1)

        # Update the projectile (move downward at a constant speed)
        if projectile is not None and projectile.active:
            projectile.update(dt)
        else:
            spawn_timer += dt
            if spawn_timer > 1.5:  # spawn a new projectile every 1.5 seconds
                projectile = spawn_projectile()
                spawn_timer = 0

        render_scene()
        pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()

