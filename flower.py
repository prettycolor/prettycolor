"""
Generative Glowing Flower — Terminal Art
=========================================
Particle-based bioluminescent flower with bloom post-processing.
Inspired by @ciscoguypro's viral Threads post.

Engine lineage: "Lighter and Princess" drama → PhCtrlZ/heart particle engine
                → prettycolor/prettycolor (this fork)

Usage:
    python flower.py              # Animated window (breathing flower)
    python flower.py --static     # Single frame → terminal via imgcat
    python flower.py --wallpaper  # 5K render for desktop wallpaper
"""

import random
import os
import time
import argparse
from math import sin, cos, pi, log, sqrt, exp

import numpy as np
from PIL import Image, ImageFilter

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1200
CX = CANVAS_WIDTH // 2
CY = CANVAS_HEIGHT // 2

COLOR_CORE  = np.array([1.00, 0.92, 0.72])   # warm gold center (NOT white)
COLOR_INNER = np.array([1.00, 0.90, 0.65])   # rich cream
COLOR_GOLD  = np.array([1.00, 0.82, 0.38])   # deep gold
COLOR_AMBER = np.array([0.95, 0.65, 0.20])   # amber
COLOR_DEEP  = np.array([0.72, 0.38, 0.10])   # deep amber


# ────────────────────────────────────────────────────────────────
# Core Engine (from heart.py)
# ────────────────────────────────────────────────────────────────

def scatter_inside(x, y, cx, cy, beta=0.15):
    """Log-normal particle diffusion."""
    ratio_x = -beta * log(random.random())
    ratio_y = -beta * log(random.random())
    return x - ratio_x * (x - cx), y - ratio_y * (y - cy)


def curve(p):
    return 2 * (2 * sin(4 * p)) / (2 * pi)


def calc_position(x, y, cx, cy, ratio):
    force = 1 / (((x - cx) ** 2 + (y - cy) ** 2 + 1) ** 0.520)
    dx = ratio * force * (x - cx) + random.randint(-1, 1)
    dy = ratio * force * (y - cy) + random.randint(-1, 1)
    return x - dx, y - dy


def angle_dist(a, b):
    """Shortest angular distance between two angles."""
    d = (a - b + pi) % (2 * pi) - pi
    return abs(d)


# ────────────────────────────────────────────────────────────────
# Flower Generator — Sum-of-Gaussians Petal Modulation
#
# Each layer defines N petals at random angular positions.
# The radius modulation is the MAX of all petal gaussians,
# creating clear petal shapes where different layers overlap
# at independent angles — producing peony-like 3D depth.
# ────────────────────────────────────────────────────────────────

class Flower:

    def __init__(self, cx=CX, cy=CY, generate_frame=20):
        self.cx = cx
        self.cy = cy
        self.particles = []
        self.all_points = {}
        self.generate_frame = generate_frame

        self._setup_petals()
        self._build_mod_lut()
        self.build()
        for frame in range(generate_frame):
            self.calc(frame)

    def _setup_petals(self):
        """Pre-generate random petal layout for all layers.

        More petals in outer rings (like a real chrysanthemum/peony).
        Inner petals are WIDER (seen face-on in the dome) while
        outer petals are narrower (curving away from viewer).
        Total: ~50 individual petals across 8 concentric rings.
        """
        self._layers = []
        layer_defs = [
            # (n_petals, r_min, r_max, width_deg)
            # Outer rings: many narrow petals (seen edge-on, ~1.2x scale)
            (14,  85, 560,  6),
            (13,  72, 500,  7),
            (12,  60, 430,  8),
            # Middle rings: moderate count + width
            (10,  45, 350,  10),
            (9,   34, 275,  12),
            # Inner rings: fewer, wider petals (dome effect)
            (7,   22, 200,  16),
            (6,   12, 140,  20),
            (4,    6,  90,  28),
        ]
        for n_pet, r_min, r_max, w_deg in layer_defs:
            offset = random.uniform(0, 2 * pi / n_pet)
            angles = []
            widths = []
            for i in range(n_pet):
                a = 2 * pi * i / n_pet + offset + random.gauss(0, 0.05)
                w = w_deg * pi / 180 * random.uniform(0.82, 1.18)
                angles.append(a % (2 * pi))
                widths.append(w)
            self._layers.append((r_min, r_max, angles, widths))

    def _build_mod_lut(self):
        """Precompute petal modulation as a 2D lookup table (vectorized).
        Indexed by (angle_bin, radius_bin) — O(1) per particle.
        """
        n_angle = 720
        n_radius = 300
        r_max_lut = 620.0
        valley = 0.14

        lut = np.full((n_angle, n_radius), valley, dtype=np.float32)

        # Precompute angle and radius grids
        angle_grid = np.linspace(0, 2 * pi, n_angle, endpoint=False)  # (n_angle,)
        r_grid = np.linspace(0, r_max_lut, n_radius, endpoint=False)  # (n_radius,)

        for r_min, r_max, angles, widths in self._layers:
            r_center = (r_min + r_max) * 0.5
            r_spread = (r_max - r_min) * 0.42

            # Radial weight for entire radius grid at once
            r_weight = np.exp(-0.5 * ((r_grid - r_center) / r_spread) ** 2)  # (n_radius,)

            for pa, pw in zip(angles, widths):
                # Angular distance for entire angle grid at once
                d_angle = np.abs(((angle_grid - pa + pi) % (2 * pi)) - pi)  # (n_angle,)
                petal_score = np.exp(-0.5 * (d_angle / pw) ** 2)  # (n_angle,)

                # Outer product: (n_angle,) x (n_radius,) → (n_angle, n_radius)
                contribution = petal_score[:, np.newaxis] * r_weight[np.newaxis, :]
                vals = valley + (1.0 - valley) * contribution
                np.maximum(lut, vals, out=lut)

        self._mod_lut = lut
        self._lut_n_angle = n_angle
        self._lut_n_radius = n_radius
        self._lut_r_max = r_max_lut

    def _petal_mod(self, angle, base_r):
        """Fast O(1) lookup from precomputed table."""
        ai = int((angle % (2 * pi)) / (2 * pi) * self._lut_n_angle) % self._lut_n_angle
        ri = int(base_r / self._lut_r_max * self._lut_n_radius)
        if ri >= self._lut_n_radius:
            return 0.14
        return float(self._mod_lut[ai, ri])

    def build(self):
        """Generate particle field with petal-modulated density."""
        cx, cy = self.cx, self.cy

        # ── Main body ──
        n_body = 600000
        for _ in range(n_body):
            angle = random.uniform(0, 2 * pi)

            roll = random.random()
            if roll < 0.35:
                base_r = abs(random.gauss(130, 200))
            elif roll < 0.65:
                base_r = random.expovariate(1 / 270)
            else:
                base_r = random.uniform(140, 580)

            mod = self._petal_mod(angle, base_r)
            r = base_r * mod

            if r > 520 or r < 2:
                continue

            x = cx + r * cos(angle)
            y = cy + r * sin(angle)
            depth = min(r / 400, 1.0)
            self.particles.append((x, y, depth))

        # ── Core ──
        for _ in range(7000):
            a = random.uniform(0, 2 * pi)
            r = abs(random.gauss(0, 13))
            self.particles.append((cx + r * cos(a), cy + r * sin(a),
                                   min(r * 0.02, 0.08)))

        # ── Organic scatter (filamentary texture) ──
        n_scat = min(90000, len(self.particles) // 5)
        for x, y, depth in random.sample(self.particles, n_scat):
            sx, sy = scatter_inside(x, y, cx, cy, beta=0.05)
            self.particles.append((sx, sy, min(depth + 0.03, 1.0)))

        # ── Wispy filaments: high-beta scatter for hair-like strands ──
        outer_body = [p for p in self.particles if p[2] > 0.5]
        wisp_sample = random.sample(outer_body, min(25000, len(outer_body)))
        for x, y, depth in wisp_sample:
            sx, sy = scatter_inside(x, y, cx, cy, beta=0.10)
            self.particles.append((sx, sy, min(depth + 0.06, 1.0)))

        # ── Tip fill: extra density at petal tips to reduce grain ──
        tip_pts = [p for p in self.particles if 0.6 < p[2] < 0.95]
        tip_sample = random.sample(tip_pts, min(30000, len(tip_pts)))
        for x, y, depth in tip_sample:
            # Small local jitter — stays within petal, fills gaps
            jx = x + random.gauss(0, 3)
            jy = y + random.gauss(0, 3)
            self.particles.append((jx, jy, depth + random.uniform(-0.02, 0.02)))

        # ── Halo: extends petal shapes outward ──
        outer = [p for p in self.particles if p[2] > 0.65]
        for x, y, depth in random.sample(outer, min(15000, len(outer))):
            sx, sy = scatter_inside(x, y, cx, cy, beta=0.08)
            sx += random.gauss(0, 10)
            sy += random.gauss(0, 10)
            self.particles.append((sx, sy, min(depth + 0.05, 1.0)))

        print(f"  {len(self.particles):,} particles")

    def calc(self, frame):
        cx, cy = self.cx, self.cy
        ratio = 8 * curve(frame / 10 * pi)
        all_points = []
        for x, y, depth in self.particles:
            breath = 0.3 + 0.7 * depth
            nx, ny = calc_position(x, y, cx, cy, ratio * breath)
            # Larger particles everywhere — tips get 2-3 (not 1-2) for smoother fill
            size = random.choice((2, 3, 3, 4, 4)) if depth < 0.3 else \
                   random.choice((2, 2, 3, 3)) if depth < 0.7 else \
                   random.choice((1, 2, 2, 3))
            all_points.append((nx, ny, size, depth))
        self.all_points[frame] = all_points


# ────────────────────────────────────────────────────────────────
# Renderer with Bloom
# ────────────────────────────────────────────────────────────────

def color_for_depth(depth):
    if depth < 0.08:
        return COLOR_CORE
    elif depth < 0.25:
        t = (depth - 0.08) / 0.17
        return COLOR_CORE * (1 - t) + COLOR_INNER * t
    elif depth < 0.50:
        t = (depth - 0.25) / 0.25
        return COLOR_INNER * (1 - t) + COLOR_GOLD * t
    elif depth < 0.75:
        t = (depth - 0.50) / 0.25
        return COLOR_GOLD * (1 - t) + COLOR_AMBER * t
    else:
        t = min((depth - 0.75) / 0.25, 1.0)
        return COLOR_AMBER * (1 - t) + COLOR_DEEP * t


def render_frame_to_array(points, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    for x, y, size, depth in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width - size and 0 <= iy < height - size:
            color = color_for_depth(depth)
            brightness = 0.40 + 0.60 * max(0, 1 - depth) ** 0.55
            center_atten = 0.25 + 0.75 * min(depth * 3.5, 1.0)
            value = color * brightness * 0.13 * center_atten
            canvas[iy:iy + size, ix:ix + size] += value
    return canvas


def apply_bloom(canvas, passes=None):
    if passes is None:
        passes = [(2, 1.0), (7, 0.85), (20, 0.65), (55, 0.48), (140, 0.30), (250, 0.12)]
    result = canvas.copy()
    for sigma, weight in passes:
        bloom = np.zeros_like(canvas)
        for c in range(3):
            channel = np.clip(canvas[:, :, c], 0, 1)
            bright = np.where(channel > 0.02, channel, 0)
            img = Image.fromarray((bright * 255).astype(np.uint8), mode='L')
            blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            bloom[:, :, c] = np.array(blurred, dtype=np.float64) / 255.0
        result += bloom * weight
    return result


def compress_luminance_only(canvas, factor):
    """Reinhard compression in luminance space — preserves color saturation.
    Instead of compressing RGB equally (→ washes to white), we:
    1. Compute luminance L = 0.2126R + 0.7152G + 0.0722B
    2. Compress L_new = L / (1 + L * factor)
    3. Scale RGB by (L_new / L) to preserve hue and saturation
    """
    lum = 0.2126 * canvas[:, :, 0] + 0.7152 * canvas[:, :, 1] + 0.0722 * canvas[:, :, 2]
    lum_compressed = lum / (1.0 + lum * factor)
    # Scale factor per pixel (avoid div/0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(lum > 1e-6, lum_compressed / lum, 1.0)
    return canvas * scale[:, :, np.newaxis]


def tonemap(canvas, gamma=0.78, exposure=1.35):
    canvas = canvas * exposure
    canvas = np.clip(canvas, 0, 1)
    return np.power(canvas, gamma)


def render_image(flower, frame=0, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                 bloom_passes=None):
    points = flower.all_points[frame % flower.generate_frame]
    canvas = render_frame_to_array(points, width, height)
    # Luminance-only compression — preserves gold hue, only tames brightness
    canvas = compress_luminance_only(canvas, 3.0)
    canvas = apply_bloom(canvas, bloom_passes)
    canvas = compress_luminance_only(canvas, 0.6)  # Gentle post-bloom
    canvas = tonemap(canvas)
    return Image.fromarray((canvas * 255).astype(np.uint8))


# ────────────────────────────────────────────────────────────────
# Output Modes
# ────────────────────────────────────────────────────────────────

def mode_static(args):
    w, h = args.width, args.height
    print("Generating flower...")
    t0 = time.time()
    if args.seed is not None:
        random.seed(args.seed)
    flower = Flower(cx=w // 2, cy=h // 2, generate_frame=1)
    t1 = time.time()
    print("Rendering with bloom...")
    img = render_image(flower, frame=0, width=w, height=h)
    t2 = time.time()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'flower.png')
    img.save(path, quality=95)
    print(f"  Generate: {t1 - t0:.1f}s  |  Render: {t2 - t1:.1f}s")
    print(f"  Saved: {path}")
    if not args.no_imgcat:
        try:
            import subprocess
            subprocess.run(['imgcat', path], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            try:
                import imgcat as imgcat_lib
                with open(path, 'rb') as f:
                    imgcat_lib.imgcat(f.read())
            except Exception:
                print(f"  (open manually: {path})")
    return path


def mode_animated(args):
    import pygame
    w, h = args.width, args.height
    n_frames = args.frames
    print(f"Generating flower ({n_frames} frames)...")
    t0 = time.time()
    if args.seed is not None:
        random.seed(args.seed)
    flower = Flower(cx=w // 2, cy=h // 2, generate_frame=n_frames)
    t1 = time.time()
    print(f"  Particles in {t1 - t0:.1f}s")
    anim_bloom = [(2, 0.9), (8, 0.6), (25, 0.35), (60, 0.15)]
    print(f"Rendering {n_frames} frames...")
    surfaces = []
    pygame.init()
    for f in range(n_frames):
        img = render_image(flower, frame=f, width=w, height=h, bloom_passes=anim_bloom)
        raw = img.tobytes()
        surf = pygame.image.fromstring(raw, img.size, 'RGB')
        surfaces.append(surf)
        elapsed = time.time() - t1
        eta = elapsed / (f + 1) * (n_frames - f - 1)
        print(f"  Frame {f + 1}/{n_frames} — ETA {eta:.0f}s    ", end='\r')
    print(f"\n  All frames in {time.time() - t1:.1f}s")
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Generative Flower")
    clock = pygame.time.Clock()
    idx = 0
    running = True
    print("  Window open. Press Escape to close.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        screen.blit(surfaces[idx], (0, 0))
        pygame.display.flip()
        idx = (idx + 1) % n_frames
        clock.tick(12)
    pygame.quit()


def mode_wallpaper(args):
    w, h = 5120, 2880
    print(f"Rendering {w}x{h} wallpaper...")
    t0 = time.time()
    if args.seed is not None:
        random.seed(args.seed)
    flower = Flower(cx=w // 2, cy=h // 2, generate_frame=1)
    wp_bloom = [(3, 1.0), (12, 0.7), (35, 0.5), (90, 0.3), (180, 0.15)]
    img = render_image(flower, frame=0, width=w, height=h, bloom_passes=wp_bloom)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'flower_wallpaper.png')
    img.save(path, quality=95)
    print(f"  {time.time() - t0:.1f}s — saved: {path}")
    if args.set_wallpaper:
        import subprocess
        script = f'tell application "System Events" to tell every desktop to set picture to "{path}"'
        subprocess.run(['osascript', '-e', script], check=True)
        print("  Desktop wallpaper set!")
    return path


def mode_screensaver(args):
    n = args.screensaver_count
    w, h = args.width, args.height
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'output', 'screensaver')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {n} unique frames...")
    for i in range(n):
        random.seed((args.seed or 42) + i)
        flower = Flower(cx=w // 2, cy=h // 2, generate_frame=1)
        img = render_image(flower, frame=0, width=w, height=h)
        path = os.path.join(output_dir, f'flower_{i + 1:03d}.png')
        img.save(path, quality=95)
        print(f"  [{i + 1}/{n}] saved")
    print(f"\n  Point macOS Screen Saver to: {output_dir}")


def mode_gif(args):
    """Export animated breathing flower as GIF."""
    w, h = args.width, args.height
    n_frames = args.frames
    print(f"Generating {n_frames}-frame GIF at {w}x{h}...")
    t0 = time.time()
    if args.seed is not None:
        random.seed(args.seed)
    flower = Flower(cx=w // 2, cy=h // 2, generate_frame=n_frames)
    t1 = time.time()
    print(f"  Particles in {t1 - t0:.1f}s")

    anim_bloom = [(2, 0.9), (7, 0.65), (20, 0.40), (55, 0.18)]
    frames = []
    for f in range(n_frames):
        img = render_image(flower, frame=f, width=w, height=h, bloom_passes=anim_bloom)
        frames.append(img)
        elapsed = time.time() - t1
        eta = elapsed / (f + 1) * (n_frames - f - 1)
        print(f"  Frame {f + 1}/{n_frames} — ETA {eta:.0f}s    ", end='\r')

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'flower.gif')
    # Save as looping GIF; duration=80ms per frame ≈ 12fps
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=False)
    print(f"\n  {time.time() - t0:.1f}s — saved: {path} ({os.path.getsize(path) // 1024}KB)")
    return path


def mode_desktop(args):
    """Live animated wallpaper using pygame (no tkinter dependency).

    Opens a borderless fullscreen window with the breathing flower.
    Press Escape or Q to quit.
    """
    import pygame
    import subprocess

    # Detect screen resolution
    pygame.init()
    info = pygame.display.Info()
    display_w, display_h = info.current_w, info.current_h

    # Render at reasonable resolution, scale up for display
    render_w = min(display_w, 1920)
    render_h = min(display_h, 1080)
    n_frames = args.frames

    print(f"Desktop mode: {display_w}x{display_h} display, rendering at {render_w}x{render_h}")
    print(f"Generating {n_frames} animation frames...")
    t0 = time.time()
    if args.seed is not None:
        random.seed(args.seed)

    flower = Flower(cx=render_w // 2, cy=render_h // 2, generate_frame=n_frames)
    t1 = time.time()
    print(f"  Particles in {t1 - t0:.1f}s")

    anim_bloom = [(2, 0.85), (7, 0.55), (20, 0.32), (50, 0.15)]

    print(f"Rendering {n_frames} frames...")
    surfaces = []
    for f in range(n_frames):
        img = render_image(flower, frame=f, width=render_w, height=render_h,
                           bloom_passes=anim_bloom)
        if render_w != display_w or render_h != display_h:
            img = img.resize((display_w, display_h), Image.LANCZOS)
        # Convert PIL → pygame surface
        raw = img.tobytes()
        surf = pygame.image.fromstring(raw, img.size, 'RGB')
        surfaces.append(surf)
        elapsed = time.time() - t1
        eta = elapsed / (f + 1) * (n_frames - f - 1)
        print(f"  Frame {f + 1}/{n_frames} — ETA {eta:.0f}s    ", end='\r')
    print(f"\n  All frames in {time.time() - t1:.1f}s")

    # Open borderless fullscreen window
    screen = pygame.display.set_mode((display_w, display_h),
                                     pygame.NOFRAME | pygame.FULLSCREEN)
    pygame.display.set_caption("Generative Flower")
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()
    idx = 0
    running = True

    print("  Desktop wallpaper running. Press Escape or Q to quit.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        screen.blit(surfaces[idx], (0, 0))
        pygame.display.flip()
        idx = (idx + 1) % n_frames
        clock.tick(12)  # 12 fps

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description='Generative Glowing Flower — Terminal Art',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flower.py                  # Animated breathing window
  python flower.py --static         # Render → terminal (imgcat)
  python flower.py --wallpaper      # 5K desktop wallpaper
  python flower.py --gif            # Export animated GIF
  python flower.py --desktop        # Live animated desktop wallpaper
  python flower.py --screensaver 20 # 20 unique frames for screensaver
  python flower.py --seed 42        # Reproducible render
        """)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--static', action='store_true')
    mode.add_argument('--wallpaper', action='store_true')
    mode.add_argument('--gif', action='store_true', help='Export animated GIF')
    mode.add_argument('--desktop', action='store_true',
                      help='Live animated desktop wallpaper (fullscreen)')
    mode.add_argument('--screensaver', type=int, metavar='N', dest='screensaver_count')
    parser.add_argument('--width', type=int, default=CANVAS_WIDTH)
    parser.add_argument('--height', type=int, default=CANVAS_HEIGHT)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--set-wallpaper', action='store_true')
    parser.add_argument('--no-imgcat', action='store_true')
    args = parser.parse_args()
    if args.wallpaper:
        mode_wallpaper(args)
    elif args.gif:
        mode_gif(args)
    elif args.desktop:
        mode_desktop(args)
    elif args.screensaver_count:
        mode_screensaver(args)
    elif args.static:
        mode_static(args)
    else:
        mode_animated(args)


if __name__ == '__main__':
    main()
