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

projectiles = []       # list to hold active projectiles
spawn_timer = 0

# Permanent markers for dome boundary entries
outer_entry_points = []  # Entry points when crossing into the outer dome
inner_entry_points = []  # Entry points when crossing into the inner dome

# -------------------- Projectile Class --------------------
class Projectile:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.active = True
        self.trail = [tuple(self.pos)]
        self.prev_distance = np.linalg.norm(self.pos)
    
    def update(self, dt):
        self.pos += self.vel * dt
        self.trail.append(tuple(self.pos))
        
        # Check for crossing dome boundaries (dome center is the origin)
        current_distance = np.linalg.norm(self.pos)
        # When crossing into the outer dome (first dome)
        if self.prev_distance > outer_dome_radius and current_distance <= outer_dome_radius:
            outer_entry_points.append(tuple(self.pos))
        # When crossing into the inner dome (second dome)
        if self.prev_distance > inner_dome_radius and current_distance <= inner_dome_radius:
            inner_entry_points.append(tuple(self.pos))
        self.prev_distance = current_distance
        
        # Deactivate when projectile falls to or below ground (y <= 0)
        if self.pos[1] <= 0:
            self.active = False

# -------------------- Utility: Zone Color --------------------
def get_zone_color(pos):
    """Return the color for a given position:
       - Green outside the outer dome,
       - Yellow in between,
       - Red inside the inner dome."""
    d = np.linalg.norm(np.array(pos))
    if d > outer_dome_radius:
        return (0, 1, 0)      # Green
    elif d > inner_dome_radius:
        return (1, 1, 0)      # Yellow
    else:
        return (1, 0, 0)      # Red

# -------------------- Projectile Spawning --------------------
def spawn_projectile():
    # Spawn from a spherical shell around the dome so trajectories come from every side.
    spawn_margin = 20
    spawn_radius = outer_dome_radius + spawn_margin
    # Sample random spherical coordinates
    theta = random.uniform(0, math.pi)         # polar angle [0, pi]
    phi = random.uniform(0, 2 * math.pi)         # azimuthal angle [0, 2pi]
    x = spawn_radius * math.sin(theta) * math.cos(phi)
    y = spawn_radius * math.cos(theta)
    z = spawn_radius * math.sin(theta) * math.sin(phi)
    pos = [x, y, z]
    # Aim toward the dome center (origin)
    direction = np.array([0, 0, 0]) - np.array(pos)
    direction = direction / np.linalg.norm(direction)
    speed = 100  # constant speed
    vel = direction * speed
    return Projectile(pos, vel)

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
    """Draw the inner dome as a flat disc with the video texture and 50% transparency."""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(1, 1, 1, 0.5)
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
    """Draw the outer dome as a transparent hemisphere composed of square patches."""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.7, 0.7, 0.7, 0.3)
    for i in range(lat_steps):
        theta1 = (i / lat_steps) * (math.pi / 2)
        theta2 = ((i + 1) / lat_steps) * (math.pi / 2)
        for j in range(lon_steps):
            phi1 = (j / lon_steps) * 2 * math.pi
            phi2 = ((j + 1) / lon_steps) * 2 * math.pi

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

# -------------------- Trail and Projectile Drawing --------------------
def draw_projectile_trail(proj):
    """Draw the projectile trail segment-by-segment with zone colors."""
    glLineWidth(8)
    glBegin(GL_LINES)
    for i in range(len(proj.trail) - 1):
        p1 = proj.trail[i]
        p2 = proj.trail[i+1]
        # Use the midpoint to determine the color zone
        mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
        color = get_zone_color(mid)
        glColor3f(color[0], color[1], color[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
    glEnd()

def draw_projectile(proj):
    """Draw the projectile as a small point using its current zone color."""
    color = get_zone_color(proj.pos)
    glPointSize(12)
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_POINTS)
    glVertex3f(proj.pos[0], proj.pos[1], proj.pos[2])
    glEnd()

def draw_permanent_markers():
    """Draw permanent yellow markers at the entry points into the domes."""
    glPointSize(10)
    glColor3f(1, 1, 0)  # Yellow for markers
    glBegin(GL_POINTS)
    for pt in outer_entry_points:
        glVertex3f(pt[0], pt[1], pt[2])
    glEnd()
    glBegin(GL_POINTS)
    for pt in inner_entry_points:
        glVertex3f(pt[0], pt[1], pt[2])
    glEnd()

# -------------------- Scene Rendering --------------------
def render_scene():
    global dome_rotation, camera_yaw, camera_pitch
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Set up camera to look at the origin (dome center)
    r = inner_dome_radius * zoom_factor
    cam_x = r * math.cos(camera_pitch) * math.cos(camera_yaw)
    cam_y = r * math.sin(camera_pitch)
    cam_z = r * math.cos(camera_pitch) * math.sin(camera_yaw)
    gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0)

    glPushMatrix()
    glRotatef(math.degrees(dome_rotation), 0, 1, 0)
    
    # Draw outer and inner domes
    draw_outer_dome(outer_dome_radius)
    if load_texture():
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        draw_inner_dome(inner_dome_radius)
        glDisable(GL_TEXTURE_2D)
    
    glPopMatrix()

    # Draw projectile trails and active projectiles
    for proj in projectiles:
        draw_projectile_trail(proj)
        if proj.active:
            draw_projectile(proj)
    
    # Draw permanent markers for dome entries
    draw_permanent_markers()

# -------------------- Main Loop --------------------
def main():
    global dome_rotation, rotating, prev_mouse_x, zoom_factor, camera_yaw, camera_pitch, spawn_timer
    pygame.init()
    pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Transparent Domes with Multi-Colored Trajectories & Permanent Markers")
    glEnable(GL_DEPTH_TEST)
    glClearColor(0, 0, 0, 1)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, window_width / window_height, 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(30) / 1000.0
        
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

        # Update projectiles and remove inactive ones
        for proj in projectiles[:]:
            if proj.active:
                proj.update(dt)
            else:
                projectiles.remove(proj)

        # Spawn new projectiles periodically
        spawn_timer += dt
        if spawn_timer > 0.5:
            projectiles.append(spawn_projectile())
            spawn_timer = 0

        render_scene()
        pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()

