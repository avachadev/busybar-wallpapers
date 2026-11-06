"""
Renders the Wallpapers themes into PNG frame sequences for the BUSY Bar .anim format.

Every theme has two sections:
  idle    a seamless loop, played forever
  action  what Start triggers; its tail blends into the end of idle, so idle can resume without a cut

Usage: python tools/render.py [theme ...]   (all themes by default)
Writes build/frames/<theme>/<n>.png and meta.json; tools/pack.py turns them into src/animations/<theme>.anim.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 72, 16
FPS = 30
IDLE_S = 8
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "frames"


# ---------------------------------------------------------------- helpers

def ramp(stops, v):
    """Maps values in [0, 1] through colour stops [(pos, (r, g, b)), ...]; returns (..., 3) floats in 0..1."""
    v = np.clip(v, 0, 1)
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], dtype=float) / 255
    return np.stack([np.interp(v, pos, cols[:, k]) for k in range(3)], axis=-1)


def rgb(*c):
    return np.array(c, dtype=float) / 255


def down(img, ss):
    """Box-filters a supersampled canvas back to screen size."""
    if ss == 1:
        return img
    h, w, _ = img.shape
    return img.reshape(h // ss, ss, w // ss, ss, 3).mean(axis=(1, 3))


def blur(img, r):
    """Separable box blur, run twice: cheap and close enough to a Gaussian for bloom."""
    out = img
    k = 2 * r + 1
    for _ in range(2):
        for axis in (0, 1):
            pad = [(0, 0)] * 3
            pad[axis] = (r, r)
            p = np.pad(out, pad, mode="edge")
            c = np.cumsum(p, axis=axis)
            c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
            n = out.shape[axis]
            out = (np.take(c, np.arange(k, k + n), axis=axis) - np.take(c, np.arange(0, n), axis=axis)) / k
    return out


def bloom(img, r=2, k=0.6):
    return img + blur(np.clip(img - 0.35, 0, None), r) * k


def splat(canvas, x, y, color, ss):
    """Adds a point at subpixel screen position (x, y) to a supersampled canvas, bilinearly."""
    X, Y = x * ss - 0.5, y * ss - 0.5
    x0, y0 = math.floor(X), math.floor(Y)
    fx, fy = X - x0, Y - y0
    h, w, _ = canvas.shape
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        if 0 <= xi < w and 0 <= yi < h:
            canvas[yi, xi] += color * wgt


def dot(canvas, x, y, color, ss, size=1.0):
    """A soft dot about `size` screen pixels wide."""
    r = max(1, int(round(size * ss / 2)))
    cx, cy = x * ss, y * ss
    h, w, _ = canvas.shape
    for yi in range(int(cy - r), int(cy + r) + 1):
        for xi in range(int(cx - r), int(cx + r) + 1):
            if 0 <= xi < w and 0 <= yi < h:
                d = math.hypot(xi + 0.5 - cx, yi + 0.5 - cy) / r
                if d < 1:
                    canvas[yi, xi] += color * (1 - d * d)


def line(canvas, x0, y0, x1, y1, color, ss, width=1.0):
    """Antialiased-ish line by stepping soft dots along it."""
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) * ss * 1.5))
    for i in range(n):
        t = i / (n - 1)
        dot(canvas, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, color / (ss * 0.75), ss, width)


def loop_crossfade(frames, k):
    """Takes len(frames) - k frames that loop seamlessly: the tail fades into the head."""
    n = len(frames) - k
    out = [f.copy() for f in frames[k:]]
    for j in range(k):
        t = (j + 1) / (k + 1)
        out[n - k + j] = out[n - k + j] * (1 - t) + frames[j] * t
    return out


def blend_tail(action, idle, k):
    """Fades the last k action frames into the last k idle frames, so idle's frame 0 follows seamlessly."""
    out = list(action)
    for j in range(k):
        t = (j + 1) / k
        i = len(out) - k + j
        out[i] = out[i] * (1 - t) + idle[len(idle) - k + j] * t
    return out


def envelope(t, attack, hold, release):
    """0 → 1 over `attack`, stays 1 for `hold`, back to 0 over `release` (seconds), smooth."""
    if t < attack:
        x = t / attack
    elif t < attack + hold:
        x = 1
    else:
        x = max(0.0, 1 - (t - attack - hold) / release)
    return x * x * (3 - 2 * x)


def gradient_v(top, bottom, ss=1, mid=None):
    ys = (np.arange(H * ss) + 0.5) / (H * ss)
    stops = [(0, top), (1, bottom)] if mid is None else [(0, top), (mid[0], mid[1]), (1, bottom)]
    col = ramp(stops, ys)
    return np.repeat(col[:, None, :], W * ss, axis=1)


# ---------------------------------------------------------------- 1. Inferno

FIRE = [(0, (0, 0, 0)), (0.18, (50, 4, 2)), (0.38, (170, 26, 6)), (0.58, (250, 100, 12)),
        (0.78, (255, 190, 50)), (0.92, (255, 240, 160)), (1, (255, 255, 235))]


class Fire:
    """Doom PSX fire at 2x resolution, burning in drifting tongues, with sparks that rise and fade."""

    SS = 2

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.heat = np.zeros((H * self.SS, W * self.SS))
        self.sparks = []
        self.t = 0.0

    def step(self, level, spark_rate, hot=0.0):
        rng, heat = self.rng, self.heat
        h, w = heat.shape
        xs = np.arange(w)
        t = self.t
        tongues = 0.62 + 0.22 * np.sin(xs / 9.0 + t * 1.9) + 0.16 * np.sin(xs / 4.1 - t * 3.3)
        heat[-1] = np.clip(level * tongues + rng.random(w) * 0.12, 0, 1)
        for y in range(1, h):
            r = rng.integers(0, 3, w)
            dst = np.clip(xs - r + 1, 0, w - 1)
            decay = rng.random(w) * 0.075 * (1 - 0.4 * hot)
            heat[y - 1, dst] = np.clip(heat[y, xs] - decay, 0, 1)
        n = rng.poisson(spark_rate)
        for _ in range(n):
            self.sparks.append([rng.uniform(2, W - 2), H - rng.uniform(2, 5), rng.uniform(-0.3, 0.3),
                                -rng.uniform(4, 9), rng.uniform(0.6, 1.4), rng.random() * 6])
        alive = []
        for s in self.sparks:
            s[0] += (s[2] + 0.8 * math.sin(t * 7 + s[5])) / FPS
            s[1] += s[3] / FPS
            s[4] -= 1 / FPS
            if s[4] > 0 and s[1] > -1:
                alive.append(s)
        self.sparks = alive
        self.t += 1 / FPS

    def frame(self, hot=0.0):
        img = down(ramp(FIRE, self.heat * (1 + 0.35 * hot)), self.SS)
        canvas = np.zeros((H * 4, W * 4, 3))
        for x, y, _, _, life, _ in self.sparks:
            splat(canvas, x, y, rgb(255, 210, 120) * min(1, life) * 2.2, 4)
        return bloom(img + down(canvas, 4) * 4, 1, 0.5)


def theme_inferno():
    fire = Fire(1)
    level = 0.78
    for _ in range(90):
        fire.step(level, 0.7)
    k = 30
    raw = []
    for _ in range(IDLE_S * FPS + k):
        fire.step(level, 0.7)
        raw.append(fire.frame())
    idle = loop_crossfade(raw, k)

    action = []
    dur = 3.5
    for i in range(int(dur * FPS)):
        t = i / FPS
        e = envelope(t, 0.35, 0.9, 1.6)
        fire.step(level + 0.35 * e, 0.7 + (40 if i == 3 else 0) + 6 * e, hot=e)
        action.append(fire.frame(hot=e))
    return idle, blend_tail(action, idle, 20)


# ---------------------------------------------------------------- 2. Hyperspace

def theme_hyperspace():
    ss = 4
    rng = np.random.default_rng(2)
    layers = [(26, 1, 0.30), (16, 2, 0.55), (9, 4, 1.0)]  # count, laps per loop, brightness
    stars = [(rng.uniform(0, W), rng.uniform(0, H), laps, b, rng.integers(1, 4), rng.uniform(0, 6.28),
              rng.choice([rgb(255, 255, 255), rgb(170, 200, 255), rgb(255, 220, 190)]))
             for count, laps, b in layers for _ in range(count)]
    yy, xx = np.mgrid[0:H * ss, 0:W * ss] / ss
    base = gradient_v((1, 1, 8), (6, 3, 18), ss)

    def nebula(phase):
        n = (0.5 + 0.5 * np.sin(xx / 9 + 2 * math.pi * phase + np.sin(yy / 3))) * \
            (0.5 + 0.5 * np.sin(yy / 4 - xx / 17 - 2 * math.pi * phase))
        m = (0.5 + 0.5 * np.sin(xx / 13 - 2 * math.pi * phase + 1.7)) * (0.5 + 0.5 * np.cos(yy / 5 + xx / 23))
        return n[..., None] * rgb(90, 30, 140) * 0.35 + m[..., None] * rgb(20, 90, 140) * 0.3

    def draw(offsets, speeds, flash=0.0, phase=0.0, blue=0.0):
        canvas = base + nebula(phase)
        for (x0, y, laps, b, tw, ph, col), off, sp in zip(stars, offsets, speeds):
            x = (x0 - off) % W
            twinkle = 0.75 + 0.25 * math.sin(2 * math.pi * tw * phase + ph)
            c = col * (1 - blue) + rgb(150, 190, 255) * blue
            length = max(0.0, sp)
            if length < 0.6:
                dot(canvas, x, y, c * b * twinkle * 1.6, ss, 1.0)
            else:
                line(canvas, x, y, x + length, y, c * b * 1.4, ss, 0.9)
        img = down(canvas, ss)
        return bloom(img, 1, 0.6) + flash * rgb(200, 220, 255)

    n = IDLE_S * FPS
    idle = []
    for i in range(n):
        phase = i / n
        offs = [laps * W * phase for (_, _, laps, *_r) in stars]
        idle.append(draw(offs, [laps * W / n for (_, _, laps, *_r) in stars], phase=phase))

    action = []
    dur = 3.2
    offs = [0.0 for _ in stars]
    for i in range(int(dur * FPS)):
        t = i / FPS
        e = envelope(t, 0.7, 0.8, 1.2)
        mult = 1 + 70 * e ** 2
        speeds = [laps * W / n * mult for (_, _, laps, *_r) in stars]
        offs = [o + s for o, s in zip(offs, speeds)]
        flash = max(0.0, 1 - abs(t - 1.55) * 6) * 0.9
        action.append(draw(offs, speeds, flash=flash, phase=t / IDLE_S, blue=e))
    return idle, blend_tail(action, idle, 18)


# ---------------------------------------------------------------- 3. Aurora

def theme_aurora():
    ss = 2
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H * ss, 0:W * ss] / ss
    sky = gradient_v((0, 1, 8), (4, 14, 30), ss)
    ridge = 12.5 + 1.8 * np.sin(np.arange(W * ss) / ss / 7.0 + 1) + 1.2 * np.sin(np.arange(W * ss) / ss / 3.1)
    mountains = (yy >= ridge[None, :])
    rim = (yy >= ridge[None, :]) & (yy < ridge[None, :] + 0.6)
    stars = [(rng.uniform(0, W), rng.uniform(0, 10), rng.integers(1, 5), rng.uniform(0, 6.28), rng.uniform(0.3, 1))
             for _ in range(28)]

    def draw(phase, storm=0.0, extra=0.0):
        p = 2 * math.pi * phase
        base = 8.5 + 2.2 * np.sin(xx / 8.0 + p) + 1.1 * np.sin(xx / 3.7 - 2 * p) - storm * 1.5
        height = 5 + 2 * np.sin(xx / 11 + 3 * p) + storm * 3
        rays = 0.45 + 0.35 * np.sin(xx * 1.3 + 5 * p + np.sin(xx / 5 + p) * 3) + 0.2 * np.sin(xx / 2.3 - 7 * p)
        rays *= 0.55 + 0.45 * np.sin(xx / 16 - p + extra)
        above = np.clip(base - yy, 0, None)
        curtain = np.where(yy <= base, np.exp(-above / height), np.exp(-(yy - base) * 2.5))
        inten = curtain * rays * (1 + storm * 1.2)
        hgt = np.clip(above / (height * 1.6), 0, 1)
        green = ramp([(0, (40, 255, 120)), (0.45, (20, 200, 170)), (1, (150, 50, 210))], hgt)
        pink = ramp([(0, (255, 80, 160)), (0.5, (200, 40, 220)), (1, (90, 30, 255))], hgt)
        col = green * (1 - storm) + pink * storm
        canvas = sky + inten[..., None] * col * 1.35
        for x, y, tw, ph, b in stars:
            dot(canvas, x, y, rgb(255, 255, 255) * b * (0.6 + 0.4 * math.sin(tw * p + ph)) * 0.9, ss, 0.8)
        canvas[mountains] = rgb(2, 3, 8)
        canvas[rim] += rgb(30, 80, 70) * (0.15 + 0.6 * float(storm))
        return bloom(down(canvas, ss), 1, 0.45)

    n = IDLE_S * FPS
    idle = [draw(i / n) for i in range(n)]
    action = []
    dur = 3.6
    for i in range(int(dur * FPS)):
        t = i / FPS
        e = envelope(t, 0.5, 1.4, 1.5)
        action.append(draw(t / IDLE_S + e * 0.25, storm=e, extra=t * 6 * e))
    return idle, blend_tail(action, idle, 20)


# ---------------------------------------------------------------- 4. Neon Rain

def theme_rain():
    ss = 2
    rng = np.random.default_rng(4)
    sky = gradient_v((10, 3, 26), (95, 30, 80), ss, mid=(0.55, (40, 10, 60)))
    buildings = []
    x = -1
    while x < W:
        bw = int(rng.integers(5, 10))
        bh = int(rng.integers(6, 13))
        buildings.append((x, bw, bh))
        x += bw + int(rng.integers(0, 2))
    windows = []
    for bx, bw, bh in buildings:
        for wy in range(H - bh + 1, H - 1, 2):
            for wx in range(bx + 1, bx + bw - 1, 2):
                if rng.random() < 0.5 and 0 <= wx < W:
                    warm = rng.random() < 0.7
                    windows.append((wx, wy, rgb(255, 180, 80) if warm else rgb(120, 220, 255),
                                    rng.uniform(0.7, 1.4), rng.integers(0, 4), rng.uniform(0, 6.28)))
    sign = buildings[len(buildings) // 2]
    drops = [(rng.uniform(0, W), rng.uniform(0, H + 6), int(rng.integers(2, 5)), rng.uniform(0.5, 1.0))
             for _ in range(80)]

    skyline = np.zeros((H * ss, W * ss), dtype=bool)
    for bx, bw, bh in buildings:
        skyline[(H - bh) * ss:, max(0, bx * ss):max(0, (bx + bw) * ss)] = True

    def draw(phase, flash=0.0, bolt=None, heavy=0.0):
        canvas = sky * (1 - flash) + gradient_v((150, 140, 230), (90, 70, 160), ss) * flash
        canvas[skyline] = rgb(7, 5, 14) * (1 - flash * 0.5)
        sx, sw, sh = sign
        pulse = 0.7 + 0.3 * math.sin(2 * math.pi * 4 * phase)
        for xi in range(max(0, sx + 1), min(W, sx + sw - 1)):
            dot(canvas, xi + 0.5, H - sh - 0.8, rgb(255, 40, 180) * pulse * 2.4, ss, 1.2)
        for wx, wy, col, b, fl, ph in windows:
            on = 1.0 if fl == 0 else (0.55 + 0.45 * math.sin(2 * math.pi * fl * phase + ph))
            dot(canvas, wx + 0.5, wy + 0.5, col * b * on * 1.8, ss, 1.1)
        for x0, y0, laps, b in drops:
            y = (y0 + laps * (H + 6) * phase * (1 + heavy * 2)) % (H + 6) - 3
            xx = (x0 - y * 0.35) % W
            line(canvas, xx, y, xx - 0.6, y + 1.8, rgb(150, 180, 255) * b * (1.6 + heavy), ss, 0.7)
        if bolt is not None:
            for (ax, ay), (bx2, by) in zip(bolt, bolt[1:]):
                line(canvas, ax, ay, bx2, by, rgb(255, 255, 255) * 3, ss, 1.0)
        return bloom(down(canvas, ss), 2, 0.5)

    n = IDLE_S * FPS
    idle = [draw(i / n) for i in range(n)]

    target = buildings[int(rng.integers(2, len(buildings) - 2))]
    bx = target[0] + target[1] / 2
    pts = [(bx + rng.uniform(-6, 6), -1)]
    y = -1
    while y < H - target[2]:
        y = min(H - target[2], y + rng.uniform(1.5, 3))
        pts.append((pts[-1][0] + rng.uniform(-2.5, 2.5), y))
    pts[-1] = (bx, H - target[2])
    action = []
    dur = 3.0
    for i in range(int(dur * FPS)):
        t = i / FPS
        flash = max(math.exp(-t * 7) if t < 0.5 else 0, 0.8 * math.exp(-(t - 0.22) * 9) if t > 0.22 else 0)
        bolt = pts if (t < 0.12 or 0.22 < t < 0.3) else None
        action.append(draw(t / IDLE_S, flash=min(1.0, flash), bolt=bolt, heavy=envelope(t, 0.2, 1.2, 1.2)))
    return idle, blend_tail(action, idle, 18)


# ---------------------------------------------------------------- 5. Synthwave

def theme_synthwave():
    ss = 4
    horizon = 8.5
    cx = 36.0
    yy, xx = np.mgrid[0:H * ss, 0:W * ss] / ss + 0.5 / ss
    is_floor = yy >= horizon
    sky = ramp([(0, (14, 0, 38)), (0.5, (90, 10, 95)), (0.85, (220, 50, 110)), (1, (255, 110, 90))],
               np.clip(yy / horizon, 0, 1))
    floor_bg = ramp([(0, (45, 0, 60)), (1, (6, 0, 16))], np.clip((yy - horizon) / (H - horizon), 0, 1))
    # The sun: big, half set, banded towards the bottom.
    cy, rad = horizon + 0.3, 7.0
    dsun = np.hypot(xx - cx, yy - cy)
    sun_col = ramp([(0, (255, 240, 120)), (0.5, (255, 140, 70)), (1, (255, 40, 140))],
                   np.clip((yy - (cy - rad)) / rad, 0, 1))
    # Jagged ridges on both sides, rising away from the sun.
    away = np.clip((np.abs(xx - cx) - 9) / 20, 0, 1)
    peaks = np.abs(np.sin(xx / 2.2)) * 0.6 + np.abs(np.sin(xx / 5.3 + 1)) * 0.9 + np.abs(np.sin(xx / 1.1)) * 0.25
    ridge = horizon - away * peaks * 3.2
    mountains = (yy >= ridge) & ~is_floor & (away > 0)
    mrim = mountains & (yy < ridge + 0.3)
    depth = np.clip(yy - horizon, 1e-3, None)

    def draw(phase, travel, pulse=0.0, glitch=0.0):
        canvas = np.where(is_floor[..., None], floor_bg, sky)
        band_pos = (yy - (cy - rad * 0.55)) / (rad * 0.55)
        bands = (np.sin((yy * 1.6 - 2 * math.pi * phase * 3)) > 0.6 - 0.9 * band_pos) & (band_pos > 0)
        sun = (dsun < rad) & ~is_floor & ~bands
        canvas = np.where(sun[..., None], sun_col * (1 + 0.5 * pulse), canvas)
        halo = np.exp(-np.clip(dsun - rad, 0, None) / 4.0) * (~is_floor) * (~sun)
        canvas = canvas + halo[..., None] * rgb(255, 50, 130) * (0.45 + 0.4 * pulse)
        canvas = np.where(mountains[..., None], rgb(22, 0, 40), canvas)
        canvas = np.where(mrim[..., None], rgb(90, 230, 255) * (1 + pulse), canvas)
        # Perspective grid: z = k / depth. Lines are kept about a quarter pixel thick on screen.
        z = 7.0 / depth
        dz = 7.0 / depth ** 2 / ss * 1.2
        fz = (z + travel) % 1.0
        hline = (fz < dz) | (fz > 1 - dz)
        wx = (xx - cx) / depth
        fx = (wx / 2.4 + 0.5) % 1.0
        dx = 0.32 / depth / 2.4
        vline = (np.abs(fx - 0.5) < dx)
        grid = (hline | vline) & is_floor
        fade = np.clip(depth / 4.0, 0.08, 1) ** 1.3
        canvas = canvas + grid[..., None] * rgb(255, 50, 210) * (1.6 * fade)[..., None] * (1 + pulse)
        canvas = canvas + np.where(is_floor, np.exp(-depth * 3), 0)[..., None] * rgb(255, 60, 160) * 0.2
        img = down(canvas, ss)
        if glitch > 0:
            r = np.roll(img[..., 0], -1, axis=1)
            b = np.roll(img[..., 2], 1, axis=1)
            img = img * (1 - glitch) + np.stack([r, img[..., 1], b], -1) * glitch
            img[int(phase * 997) % H] += glitch * 0.2
        return bloom(img, 1, 0.5)

    n = IDLE_S * FPS
    idle = [draw(i / n, i / n * 6) for i in range(n)]
    action = []
    dur = 3.2
    travel = 0.0
    for i in range(int(dur * FPS)):
        t = i / FPS
        e = envelope(t, 0.5, 1.2, 1.3)
        travel += (6 / n) * (1 + 12 * e)
        action.append(draw(t / IDLE_S, travel, pulse=e, glitch=e * (0.6 + 0.4 * math.sin(t * 40))))
    return idle, blend_tail(action, idle, 18)


# ---------------------------------------------------------------- 6. Lava Lamp

def theme_lava():
    ss = 2
    rng = np.random.default_rng(6)
    yy, xx = np.mgrid[0:H * ss, 0:W * ss] / ss
    bg = ramp([(0, (30, 0, 45)), (1, (70, 0, 40))], yy / H)
    blobs = [(5 + i * 10.3, rng.uniform(10, 18), int(rng.integers(1, 3)), int(rng.integers(1, 3)),
              rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(3.3, 4.8)) for i in range(7)]

    def draw(phase, heat=0.0, extra=()):
        f = np.zeros_like(xx)
        p = 2 * math.pi * phase
        for cx, ax, fx, fy, px, py, r in blobs:
            x = cx + ax * 0.5 * math.sin(fx * p + px)
            y = 8 + 6.5 * math.sin(fy * p + py)
            rr = r * (1 + 0.25 * heat)
            f += rr * rr / ((xx - x) ** 2 + ((yy - y) * 1.6) ** 2 + 0.3)
        for x, y, r in extra:
            f += r * r / ((xx - x) ** 2 + ((yy - y) * 1.6) ** 2 + 0.3)
        inside = np.clip((f - 1.0) * 0.7, 0, 1)
        edge = np.clip(1 - np.abs(f - 1.0) * 3, 0, 1)
        core = ramp([(0, (230, 30, 60)), (0.35, (255, 90, 40)), (0.75, (255, 160, 40)), (1, (255, 220, 110))],
                    inside * (0.8 + 0.25 * heat))
        canvas = bg * (1 - np.clip(f, 0, 1)[..., None] * 0.3)
        canvas = np.where((f > 1)[..., None], core, canvas)
        canvas += edge[..., None] * rgb(255, 40, 160) * 0.8
        canvas += np.clip(f, 0, 1)[..., None] * rgb(120, 0, 80) * 0.25
        return bloom(down(canvas, ss), 2, 0.3 + 0.15 * heat)

    n = IDLE_S * FPS
    idle = [draw(i / n) for i in range(n)]
    action = []
    dur = 3.6
    bubbles = [(rng.uniform(6, 66), rng.uniform(0.0, 0.8), rng.uniform(1.0, 1.8)) for _ in range(9)]
    for i in range(int(dur * FPS)):
        t = i / FPS
        e = envelope(t, 0.6, 1.4, 1.3)
        extra = [(x, H + 2 - (t - d) * 9, r * e) for x, d, r in bubbles if t > d]
        action.append(draw(t / IDLE_S + e * 0.08, heat=e, extra=extra))
    return idle, blend_tail(action, idle, 20)


# ---------------------------------------------------------------- 7. Matrix

GLYPHS = {
    "B": ["1111.", "1...1", "1...1", "1111.", "1...1", "1...1", "1111."],
    "U": ["1...1", "1...1", "1...1", "1...1", "1...1", "1...1", ".111."],
    "S": [".1111", "1....", "1....", ".111.", "....1", "....1", "1111."],
    "Y": ["1...1", "1...1", ".1.1.", "..1..", "..1..", "..1..", "..1.."],
}


def text_mask(text, x0, y0, scale=1):
    m = np.zeros((H, W))
    x = x0
    for ch in text:
        for r, rowbits in enumerate(GLYPHS[ch]):
            for c, bit in enumerate(rowbits):
                if bit == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            yi, xi = y0 + r * scale + dy, x + c * scale + dx
                            if 0 <= yi < H and 0 <= xi < W:
                                m[yi, xi] = 1
        x += 6 * scale
    return m


def theme_matrix():
    rng = np.random.default_rng(7)
    n = IDLE_S * FPS
    cols = []
    for x in range(W):
        if rng.random() < 0.82:
            for _ in range(int(rng.integers(1, 3))):
                cols.append((x, rng.uniform(0, 40), int(rng.integers(2, 6)), int(rng.integers(5, 14))))
    period = 40.0
    noise_frames = [rng.uniform(0.55, 1.0, (H, W)) for _ in range(12)]
    word = text_mask("BUSY", 9, 4)

    def draw(i, phase, speed=1.0, reveal=None, wordglow=0.0):
        g = np.zeros((H, W))
        head = np.zeros((H, W))
        for x, y0, laps, length in cols:
            yh = (y0 + laps * period * phase * speed) % period - 6
            for k in range(length):
                y = int(math.floor(yh)) - k
                if 0 <= y < H:
                    v = (1 - k / length) ** 1.4
                    if k == 0:
                        head[y, x] = max(head[y, x], 1.0)
                    g[y, x] = max(g[y, x], v)
        g *= noise_frames[(i // 3) % len(noise_frames)]
        img = g[..., None] * rgb(30, 255, 90) * 0.85 + head[..., None] * rgb(210, 255, 220)
        img += rgb(0, 6, 2)
        if reveal is not None:
            locked = word * reveal
            img = img * (1 - locked[..., None] * 0.85) + locked[..., None] * rgb(170, 255, 190) * (0.9 + wordglow)
        return bloom(img, 1, 0.55)

    idle = [draw(i, i / n) for i in range(n)]

    action = []
    dur = 4.0
    lock_at = rng.uniform(0.25, 1.1, (H, W))
    unlock_at = rng.uniform(2.6, 3.4, (H, W))
    ph = 0.0
    for i in range(int(dur * FPS)):
        t = i / FPS
        e = envelope(t, 0.3, 2.4, 1.0)
        ph += (1 + 1.5 * e) / n
        reveal = ((t > lock_at) & (t < unlock_at)).astype(float)
        glow = 0.4 * max(0.0, 1 - abs(t - 1.3) * 3)
        action.append(draw(i, ph, reveal=reveal, wordglow=glow))
    return idle, blend_tail(action, idle, 18)


# ---------------------------------------------------------------- 8. Fireworks

PALETTES = [rgb(255, 60, 90), rgb(80, 200, 255), rgb(255, 210, 60), rgb(140, 255, 120), rgb(220, 90, 255),
            rgb(255, 140, 40)]


def firework_events(rng, times, big=False):
    ev = []
    for t0 in times:
        ev.append({
            "t0": t0, "x": rng.uniform(10, 62), "y": rng.uniform(1.5, 5.5), "rise": rng.uniform(0.5, 0.8),
            "col": PALETTES[int(rng.integers(0, len(PALETTES)))],
            "col2": PALETTES[int(rng.integers(0, len(PALETTES)))],
            "n": int(rng.integers(30, 44)) if not big else 64, "speed": rng.uniform(9, 13) * (1.3 if big else 1),
            "life": rng.uniform(1.2, 1.7) * (1.5 if big else 1), "seed": int(rng.integers(0, 1 << 30)),
            "willow": big,
        })
    return ev


def draw_fireworks(canvas, events, t, ss):
    for e in events:
        dt = t - e["t0"]
        if dt < 0:
            continue
        x0, yb = e["x"], e["y"]
        if dt < e["rise"]:
            u = dt / e["rise"]
            y = H - (H - yb) * (1 - (1 - u) ** 2)
            dot(canvas, x0, y, rgb(255, 220, 160) * 1.4, ss, 0.9)
            for k in range(1, 5):
                dot(canvas, x0 + math.sin(k * 3 + dt * 20) * 0.2, y + k * 0.6, rgb(255, 150, 60) * (0.5 / k), ss, 0.8)
            continue
        db = dt - e["rise"]
        if db > e["life"]:
            continue
        if db < 0.15:
            dot(canvas, x0, yb, rgb(255, 250, 230) * (1 - db / 0.15) * 4, ss, 5.0)
        rng = np.random.default_rng(e["seed"])
        fade = (1 - db / e["life"]) ** 1.5
        for k in range(e["n"]):
            ang = 2 * math.pi * k / e["n"] + rng.uniform(-0.1, 0.1)
            sp = e["speed"] * (rng.uniform(0.82, 1.05) if k % 2 else rng.uniform(0.25, 0.8))
            drag = 1.7 if not e["willow"] else 1.3
            dist = sp * (1 - math.exp(-drag * db)) / drag
            gy = (3.0 if not e["willow"] else 4.5) * db * db
            px = x0 + math.cos(ang) * dist * 1.25
            py = yb + math.sin(ang) * dist * 0.7 + gy
            col = e["col"] if k % 3 else e["col2"]
            if e["willow"]:
                col = rgb(255, 190, 80)
            tw = 1.0 if db < e["life"] * 0.6 else (0.4 + 0.6 * (math.sin(db * 40 + k) > 0))
            # A splat lands in one of ss*ss subpixels, so it is scaled up to read as a full screen pixel.
            splat(canvas, px, py, col * fade * tw * 1.3 * ss * ss, ss)
            for trail in (1, 2):
                back = trail * 0.45 * min(1.0, sp * math.exp(-drag * db) / 6)
                splat(canvas, px - math.cos(ang) * back * 1.25, py - math.sin(ang) * back * 0.7 - gy * 0.08 * trail,
                      col * fade * (0.55 / trail) * ss * ss, ss)


def theme_fireworks():
    ss = 4
    rng = np.random.default_rng(8)
    horizon = 13
    sky = gradient_v((1, 1, 10), (14, 6, 34), ss)
    yy = (np.arange(H * ss) + 0.5) / ss
    water = yy >= horizon

    stars = [(rng.uniform(0, W), rng.uniform(0, 11), rng.uniform(0.2, 0.6), rng.uniform(0, 6.28)) for _ in range(22)]

    def compose(events, t):
        canvas = sky.copy()
        for sx, sy, b, ph in stars:
            dot(canvas, sx, sy, rgb(200, 210, 255) * b * (0.7 + 0.3 * math.sin(2 * math.pi * t / IDLE_S * 3 + ph)), ss, 0.8)
        draw_fireworks(canvas, events, t, ss)
        canvas[water] = rgb(2, 3, 12)
        img = down(canvas, ss)
        for r in range(horizon, H):
            src = 2 * horizon - r - 1
            shift = int(round(math.sin(t * 6 + r * 1.7) * 1.0))
            img[r] += np.roll(img[src], shift, axis=0) * (0.42 - 0.08 * (r - horizon))
        img[horizon] += rgb(10, 8, 30)
        return bloom(img, 2, 0.7)

    loop = IDLE_S
    ev = firework_events(rng, [0.2, 1.5, 2.9, 4.0, 5.3, 6.6])
    shifted = ev + [dict(e, t0=e["t0"] - loop) for e in ev]
    n = loop * FPS
    idle = [compose(shifted, i / FPS) for i in range(n)]

    finale = firework_events(rng, [0.0, 0.15, 0.35, 0.5, 0.7, 0.85, 1.05, 1.3]) + \
        firework_events(rng, [1.6], big=True)
    dur = 4.2
    action = [compose(finale, i / FPS) for i in range(int(dur * FPS))]
    return idle, blend_tail(action, idle, 20)


# ---------------------------------------------------------------- 9. Black Hole

DISK = [(0, (0, 0, 0)), (0.2, (90, 20, 5)), (0.45, (230, 100, 20)), (0.7, (255, 190, 90)), (0.9, (255, 240, 200)),
        (1, (255, 255, 255))]


def theme_blackhole():
    ss = 4
    rng = np.random.default_rng(9)
    cx, cy = 25.0, 8.0  # left of centre, clear of the clock
    tilt = 0.24  # sin of the viewing angle above the disk plane: thin, but the orbits read as ellipses
    yy, xx = np.mgrid[0:H * ss, 0:W * ss] / ss + 0.5 / ss
    dx, dy = xx - cx, yy - cy
    r_screen = np.hypot(dx, dy * 1.1)
    ang_screen = np.arctan2(dy, dx)
    shadow_r = 4.0
    r_in, r_out = 5.0, 34.0
    # Disk-plane coordinates of every pixel: the near half lies below the centre, the far half above.
    u, v = dx, dy / tilt
    rd = np.hypot(u, v)
    phi = np.arctan2(v, u)
    near = dy > 0
    profile = np.clip((rd - r_in) / 1.5, 0, 1) * (r_in / np.maximum(rd, r_in)) ** 1.2 * (rd < r_out)
    # Keplerian shear: inner rings orbit faster. Streak phases are fixed per ring so the texture winds up like a real disk.
    omega = 2.4 * (r_in / np.maximum(rd, r_in)) ** 1.5
    seeds = [(int(rng.integers(3, 11)), rng.uniform(0, 6.28), rng.uniform(0.04, 0.16), rng.uniform(0.12, 0.3))
             for _ in range(7)]
    lanes = 0.8 + 0.2 * np.sin(rd * 1.7 + 0.5) * np.sin(rd * 0.63)
    doppler = 1 + 0.55 * np.tanh(-dx / 7)  # the side spinning towards us is brighter
    stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.2, 0.9)) for _ in range(44)]

    def wound(tau, offset):
        tex = np.full_like(rd, 0.45)
        for k, ph, freq, amp in seeds:
            tex += amp * np.sin(k * (phi - omega * tau) + ph + offset + rd * freq)
        return tex

    flow_t = 4.0  # seconds before a layer resets; IDLE_S is a multiple, so the idle loop closes by itself

    def disk_texture(t):
        # Flow-map trick: two layers half a period apart, each faded out before the shear winds it into rings.
        ta, tb = t % flow_t, (t + flow_t / 2) % flow_t
        wa = math.sin(math.pi * ta / flow_t) ** 2
        tex = wa * wound(ta, 0.0) + (1 - wa) * wound(tb, 2.1)
        return np.clip(tex * lanes, 0, 1.2)

    def draw(t, flare=0.0, jets=0.0, star=None, tail=()):
        canvas = np.zeros((H * ss, W * ss, 3)) + rgb(1, 1, 4)
        for sx, sy, b in stars:
            ddx, ddy = sx - cx, sy - cy
            rr = math.hypot(ddx, ddy) + 1e-3
            k = 1 + 16.0 / (rr * rr)  # stars near the hole are thrown outward, as by a lens
            if rr * k > shadow_r + 0.8:
                dot(canvas, cx + ddx * k, cy + ddy * k, rgb(200, 210, 255) * b * (1 + flare), ss, 0.8)
        disk = profile * disk_texture(t) * doppler * (1 + 0.7 * flare)
        # Far half of the disk, light bent over the top of the shadow: the lensed arc.
        arc = np.exp(-((r_screen - 5.2) ** 2) / (2 * 0.7 ** 2)) * (dy < 0.5) * (0.5 + 0.5 * np.cos(ang_screen + math.pi / 2) ** 2)
        arc *= (0.7 + 0.3 * np.sin(ang_screen * 4 - t * 3)) * doppler * (1 + flare)
        ring = np.exp(-((r_screen - shadow_r - 0.3) ** 2) / (2 * 0.25 ** 2)) * (0.8 + flare)
        far_heat = np.where(~near, disk, 0) + arc + ring
        near_heat = np.where(near, disk, 0)
        hole = r_screen < shadow_r
        heat_behind = np.where(hole, 0, far_heat)
        heat = np.clip(heat_behind + near_heat, 0, 1.4)
        col = ramp(DISK, np.clip(heat, 0, 1)) * (1 + 0.35 * np.clip(heat - 1, 0, None))[..., None]
        col += np.clip(-dx / 22, 0, 1)[..., None] * heat[..., None] * rgb(30, 50, 110)  # approaching side shifts blue
        canvas = np.where((hole & (near_heat < 0.05))[..., None], rgb(0, 0, 0), canvas + col)
        if jets > 0:
            jet = np.exp(-(dx ** 2) / (2 * (0.45 + 0.22 * np.abs(dy)) ** 2)) * np.clip(np.abs(dy) - 3.0, 0, 1)
            canvas += (jet * jets)[..., None] * rgb(140, 120, 255) * 1.7
        if star is not None:
            dot(canvas, star[0], star[1], rgb(255, 255, 255) * 3, ss, 1.6)
        for tx, ty, b in tail:
            splat(canvas, tx, ty, rgb(255, 230, 200) * b * ss * ss, ss)
        return bloom(down(canvas, ss), 2, 0.6 + 0.4 * flare)

    idle = [draw(i / FPS) for i in range(IDLE_S * FPS)]
    t0 = 0.0

    def orbit_point(s):
        """A star falling in along the disk plane, projected like the disk."""
        rad = 30 * (1 - s) ** 1.5 + 5.2
        a = 1.2 + s * 9.0
        return cx + math.cos(a) * rad, cy + math.sin(a) * rad * tilt

    action = []
    dur = 4.2
    for i in range(int(dur * FPS)):
        t = i / FPS
        star, tail = None, []
        if t < 1.4:
            s = t / 1.4
            star = orbit_point(s)
            for j in range(1, 22):
                tx, ty = orbit_point(max(0.0, s - j * 0.01 * (0.3 + s)))
                tail.append((tx, ty, (1 - j / 22) * (0.3 + s) * 1.3))
        flare = envelope(t - 1.3, 0.15, 0.6, 1.8) if t > 1.3 else 0.0
        jets = envelope(t - 1.35, 0.2, 0.8, 1.2) if t > 1.35 else 0.0
        action.append(draw(t0 + t, flare=flare, jets=jets, star=star, tail=tail))
    return idle, blend_tail(action, idle, 20)


# ---------------------------------------------------------------- 10. Night Train

def periodic_profile(rng, xs, period, terms, amp):
    """A random silhouette that repeats every `period` pixels."""
    h = np.zeros_like(xs, dtype=float)
    for _ in range(terms):
        k = int(rng.integers(1, 9))
        h += rng.uniform(0.3, 1) * np.sin(2 * math.pi * k * xs / period + rng.uniform(0, 6.28)) / k ** 0.7
    return h / np.abs(h).max() * amp


TRAIN_S = 16  # twice the usual loop: the layers scroll at half speed and still tile seamlessly


def theme_train():
    ss = 2
    rng = np.random.default_rng(10)
    xs = np.arange(W * ss) / ss
    yy = (np.arange(H * ss) + 0.5) / ss
    sky = gradient_v((20, 28, 80), (255, 130, 110), ss, mid=(0.5, (110, 70, 150)))
    stars = [(rng.uniform(0, W), rng.uniform(0, 6), rng.uniform(0.3, 0.9)) for _ in range(30)]
    far, near = int(rng.integers(0, 1 << 30)), int(rng.integers(0, 1 << 30))
    town_cols = [rgb(255, 190, 90), rgb(255, 230, 160), rgb(140, 200, 255)]
    town = [(rng.uniform(0, W), rng.uniform(9.6, 11.2), town_cols[int(rng.integers(0, 3))], rng.uniform(0.5, 1.2))
            for _ in range(34)]
    signals = [(rng.uniform(0, W), [rgb(255, 40, 40), rgb(40, 255, 120)][k]) for k in range(2)]
    moon = np.exp(-((xs[None, :] - 59) ** 2 + (yy[:, None] - 3.2) ** 2) / 30)[..., None] * rgb(60, 60, 90) * 0.5

    def silhouette(seed, shift, base, amp, terms):
        prof = base + periodic_profile(np.random.default_rng(seed), (xs + shift) % W, W, terms, amp)
        return yy[:, None] >= prof[None, :]

    def draw(phase, tunnel=0.0, lamps=0.0, sparks=0, fr=None):
        canvas = sky + moon
        for sx, sy, b in stars:
            dot(canvas, sx, sy, rgb(220, 220, 255) * b, ss, 0.8)
        dot(canvas, 59, 3.2, rgb(255, 245, 220) * 1.4, ss, 3.2)
        canvas[silhouette(far, W * 1 * phase, 8.6, 1.6, 5)] = rgb(80, 50, 110)
        for tx, ty, col, b in town:
            dot(canvas, (tx - W * 2 * phase) % W, ty, col * b * 2.0 * (0.8 + 0.2 * math.sin(phase * 40 + tx)), ss, 0.9)
        canvas[silhouette(near, W * 4 * phase, 12.6, 1.6, 7)] = rgb(14, 8, 26)
        for sx, col in signals:
            dot(canvas, (sx - W * 4 * phase) % W, 11.6, col * 2.2, ss, 1.2)
        # Catenary poles and the sagging wire between them flash past.
        gap = W / 3
        for k in range(4):
            px = (k * gap - W * 9 * phase) % (W + 6) - 3
            canvas[:, max(0, int(px * ss)):max(0, int((px + 0.9) * ss))] = rgb(3, 3, 8)
            u = (xs - px) / gap
            on = (u >= 0) & (u <= 1)
            wy = ((1.6 + 6.4 * u * (1 - u)) * ss).astype(int)
            cols = np.nonzero(on & (wy >= 0) & (wy < H * ss))[0]
            canvas[wy[cols], cols] = rgb(4, 3, 10)
        if tunnel > 0:
            canvas = canvas * (1 - tunnel) + rgb(4, 2, 2) * tunnel
            for k in range(5):
                lx = (k * 17 - lamps * 900) % (W + 20) - 10
                line(canvas, lx, 3, lx + 9, 3, rgb(255, 150, 50) * 2.0 * tunnel, ss, 1.0)
                line(canvas, lx, 13.5, lx + 12, 13.5, rgb(255, 120, 40) * 0.6 * tunnel, ss, 0.8)
            for _ in range(sparks):
                sx, sy = fr.uniform(20, 52), fr.uniform(0, 2)
                line(canvas, sx, sy, sx - fr.uniform(1, 4), sy + fr.uniform(0.5, 2), rgb(170, 200, 255) * 3 * tunnel, ss, 0.7)
        return bloom(down(canvas, ss), 1, 0.5)

    n = TRAIN_S * FPS
    idle = [draw(i / n) for i in range(n)]
    action = []
    dur = 4.0
    fr = np.random.default_rng(100)
    for i in range(int(dur * FPS)):
        t = i / FPS
        tunnel = envelope(t, 0.25, 2.6, 0.5)
        sparks = int(fr.poisson(3)) if fr.random() < 0.35 and tunnel > 0.5 else 0
        action.append(draw(t / TRAIN_S, tunnel=tunnel, lamps=t * 0.6, sparks=sparks, fr=fr))
    return idle, blend_tail(action, idle, 14)


# ---------------------------------------------------------------- 11. Low Orbit

def theme_orbit():
    ss = 4
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:H * ss, 0:W * ss] / ss + 0.5 / ss
    # Seen from far enough away for the planet to be a dome, off-centre so the clock keeps clear sky.
    ecx, ecy, er = 30.0, 52.0, 42.0
    nx = (xx - ecx) / er
    ny = (yy - ecy) / er
    rho2 = nx * nx + ny * ny
    on_disk = rho2 < 1
    nz = np.sqrt(np.clip(1 - rho2, 0, 1))
    alt = (np.sqrt(rho2) - 1) * er  # screen pixels above the limb
    tilt = 0.85  # the north pole leans away from us: we look down onto the cap and the cities sweep across it

    # Twelve-fold symmetric city map: the globe turns 30 degrees per loop, slowly, and still loops seamlessly.
    base = []
    for _ in range(22):
        lon0, lat0 = rng.uniform(0, math.pi / 6), rng.uniform(-0.1, 1.15)
        for _ in range(int(rng.integers(3, 10))):
            base.append((lon0 + rng.normal(0, 0.03), lat0 + rng.normal(0, 0.03), rng.uniform(0.4, 1.2)))
    cities = [(lon + k * math.pi / 6, lat, b) for k in range(12) for lon, lat, b in base]
    storms = [(rng.uniform(0, math.pi / 6) + k * math.pi / 6, rng.uniform(0.0, 1.0), rng.uniform(0, 1))
              for k in range(12) for _ in range(1)]

    def project(lon, lat, rot):
        """Sphere point → screen, with the axis tilted towards the viewer. Returns None on the far side."""
        x = math.cos(lat) * math.sin(lon + rot)
        y = math.sin(lat)
        z = math.cos(lat) * math.cos(lon + rot)
        y, z = y * math.cos(tilt) + z * math.sin(tilt), -y * math.sin(tilt) + z * math.cos(tilt)
        if z <= 0.05:
            return None
        return ecx + x * er, ecy - y * er, z

    # Space: a Milky Way band of faint dust and dense tiny stars, plus brighter stars.
    band_d = np.abs((yy - 4) * 0.9 - (xx - 40) * 0.18)
    milky = np.exp(-band_d ** 2 / 10) * (0.6 + 0.4 * np.sin(xx * 0.7 + yy * 1.3) * np.sin(xx * 0.23))
    space = rgb(0, 0, 4) + milky[..., None] * rgb(30, 26, 50) * 0.9
    dust = []
    for _ in range(140):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        d = abs((y - 4) * 0.9 - (x - 40) * 0.18)
        if rng.random() < math.exp(-d * d / 12):
            dust.append((x, y, rng.uniform(0.1, 0.35), rng.uniform(0, 6.28)))
    stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.3, 1.0), rng.uniform(0, 6.28), int(rng.integers(1, 4)))
             for _ in range(24)]
    night = ramp([(0, (8, 14, 34)), (0.5, (3, 6, 16)), (1, (1, 2, 8))], np.clip(1 - nz, 0, 1) ** 0.5)
    night_full = np.where(on_disk[..., None], night, 0)

    def draw(phase, sunrise=0.0):
        p = 2 * math.pi * phase
        canvas = space.copy()
        for x, y, b, ph in dust:
            dot(canvas, x, y, rgb(200, 200, 255) * b * (0.8 + 0.2 * math.sin(p * 2 + ph)), ss, 0.6)
        for x, y, b, ph, tw in stars:
            dot(canvas, x, y, rgb(230, 235, 255) * b * (0.65 + 0.35 * math.sin(p * tw + ph)) * (1 - 0.6 * sunrise), ss, 0.9)
        # Two satellites cross the sky each loop, catching the sun; a meteor burns up in the atmosphere.
        for start, sy, dirx in ((0.1, 2.5, 1), (0.55, 6.0, -1)):
            u = (phase - start) % 1.0
            if u < 0.4:
                sxp = (u / 0.4) * (W + 10) - 5
                sxp = sxp if dirx > 0 else W - sxp
                dot(canvas, sxp, sy + u * 3, rgb(255, 255, 230) * (0.8 + 0.6 * (math.sin(u * 60) > 0.6)), ss, 0.9)
        mu = (phase - 0.3) % 1.0
        if mu < 0.08:
            s = mu / 0.08
            mx, my = 44 + s * 10, 4 + s * 7
            for j in range(8):
                splat(canvas, mx - j * 0.9, my - j * 0.6, rgb(180, 255, 200) * (1 - j / 8) * (1 - s * 0.5) * 0.8 * ss * ss, ss)
        canvas = np.where(on_disk[..., None], night_full, canvas)
        rot = p / 12
        for lon, lat, b in cities:
            pr = project(lon, lat, rot)
            if pr is not None:
                x, y, z = pr
                splat(canvas, x, y, rgb(255, 180, 80) * b * (0.3 + 0.7 * z) * 0.6 * ss * ss, ss)
        for lon, lat, t0 in storms:
            pr = project(lon, lat, rot)
            if pr is not None:
                local = (phase * 2 + t0) % 1.0
                flash = max(0.0, 1 - abs(local - 0.5) * 30) + 0.6 * max(0.0, 1 - abs(local - 0.56) * 40)
                if flash > 0:
                    dot(canvas, pr[0], pr[1], rgb(200, 220, 255) * flash * 2.2, ss, 2.4)
        # Atmosphere: a thin blue limb hugging the globe and the green airglow layer floating just above it.
        limb = np.exp(-np.clip(alt, 0, None) / 0.5) * (alt > -0.6)
        airglow = np.exp(-((alt - 0.9) ** 2) / (2 * 0.22 ** 2))
        canvas += limb[..., None] * rgb(40, 110, 255) * (0.6 + 1.2 * sunrise)
        canvas += airglow[..., None] * rgb(60, 255, 110) * 0.3 * (1 - sunrise * 0.5)
        if sunrise > 0:
            # The sun rises behind the right shoulder of the globe.
            sx, sy = 44.0, 12.0  # just above the limb, left of the clock
            dd = np.hypot(xx - sx, (yy - sy) * 1.4)
            band = np.exp(-np.abs(alt) / 0.7) * np.exp(-np.hypot(xx - sx, yy - sy) / 12)
            spectral = ramp([(0, (255, 60, 20)), (0.4, (255, 170, 60)), (0.7, (120, 200, 255)), (1, (40, 90, 255))],
                            np.clip((alt + 0.6) / 2.4, 0, 1))
            canvas += (band * sunrise * 2.4)[..., None] * spectral
            canvas += np.exp(-dd / 1.1)[..., None] * rgb(255, 250, 230) * sunrise * 4
            streak = np.exp(-((yy - sy) ** 2) / (2 * 0.3 ** 2)) * np.exp(-np.abs(xx - sx) / 26)
            canvas += (streak * sunrise * 1.3)[..., None] * rgb(255, 220, 180)
            day = np.exp(-np.hypot(xx - sx, yy - sy) / 9) * on_disk
            canvas += (day * sunrise * 0.6)[..., None] * rgb(40, 90, 170)
        return bloom(down(canvas, ss), 2, 0.6)

    n = IDLE_S * FPS
    idle = [draw(i / n) for i in range(n)]
    action = []
    dur = 4.5
    for i in range(int(dur * FPS)):
        t = i / FPS
        action.append(draw(t / IDLE_S, sunrise=envelope(t, 1.2, 1.5, 1.6)))
    return idle, blend_tail(action, idle, 20)


# ---------------------------------------------------------------- 12. Slime Mold

SLIME = [(0, (0, 0, 0)), (0.12, (6, 14, 40)), (0.35, (0, 110, 140)), (0.6, (60, 220, 170)), (0.82, (255, 210, 90)),
         (1, (255, 255, 230))]


class Physarum:
    """Jones' Physarum model: agents sense the trail ahead, turn towards it, move and deposit; the trail diffuses and decays.

    Runs on a 3x grid that wraps left-right and has walls top and bottom: on a short toroidal grid every agent
    ends up in a single loop, while walls and noisy turning keep it growing branching veins.
    """

    SCALE = 3

    def __init__(self, seed, n=3000):
        self.rng = np.random.default_rng(seed)
        self.w, self.h = W * self.SCALE, H * self.SCALE
        self.x = self.rng.uniform(0, self.w, n)
        self.y = self.rng.uniform(0, self.h, n)
        self.a = self.rng.uniform(0, 2 * math.pi, n)
        self.trail = np.zeros((self.h, self.w))
        yi, xi = np.mgrid[0:self.h, 0:self.w]
        self.grid = (xi, yi)
        self.norm = 1.0

    def sense(self, angle, dist=2.5):
        sx = ((self.x + np.cos(angle) * dist) % self.w).astype(int)
        sy = np.clip(self.y + np.sin(angle) * dist, 0, self.h - 1).astype(int)
        return self.trail[sy, sx]

    def step(self, food=()):
        sa, ra = 1.0, 0.6
        f, left, right = self.sense(self.a), self.sense(self.a - sa), self.sense(self.a + sa)
        turn = np.where((f >= left) & (f >= right), 0.0, np.where(left > right, -ra, ra))
        self.a += turn + self.rng.uniform(-0.4, 0.4, self.a.shape)
        self.x = (self.x + np.cos(self.a)) % self.w
        ny = self.y + np.sin(self.a)
        hit = (ny < 0) | (ny >= self.h)
        self.a = np.where(hit, -self.a, self.a)
        self.y = np.clip(ny, 0, self.h - 0.01)
        np.add.at(self.trail, (self.y.astype(int), self.x.astype(int)), 1.0)
        xi, yi = self.grid
        for fx, fy, amount in food:
            self.trail += amount * np.exp(-(((xi - fx) ** 2 + (yi - fy) ** 2) / 18.0))
        t = self.trail
        self.trail = (t + np.roll(t, 1, 0) + np.roll(t, -1, 0) + np.roll(t, 1, 1) + np.roll(t, -1, 1)) / 5 * 0.7

    def calibrate(self):
        """Fixes the brightness scale once, so frames don't pump."""
        self.norm = float(np.percentile(self.trail, 98)) + 1e-6

    def frame(self, glow=0.0):
        v = np.clip(self.trail / self.norm, 0, 1) ** 0.85
        return bloom(down(ramp(SLIME, np.clip(v * (1 + 0.4 * glow), 0, 1)), self.SCALE), 1, 0.5 + 0.4 * glow)


def theme_slime():
    sim = Physarum(12)
    for _ in range(500):
        sim.step()
    sim.calibrate()
    k = 40
    raw = []
    for _ in range(IDLE_S * FPS + k):
        sim.step()
        sim.step()
        raw.append(sim.frame())
    idle = loop_crossfade(raw, k)
    rng = np.random.default_rng(13)
    food = [(rng.uniform(15, 200), rng.uniform(10, 38)) for _ in range(3)]
    action = []
    dur = 4.5
    for i in range(int(dur * FPS)):
        t = i / FPS
        e = envelope(t, 0.2, 2.2, 1.2)
        sim.step(food=[(fx, fy, 6.0 * e) for fx, fy in food])
        sim.step(food=[(fx, fy, 6.0 * e) for fx, fy in food])
        action.append(sim.frame(glow=e))
    return idle, blend_tail(action, idle, 24)


# ---------------------------------------------------------------- 13. Plasma

def theme_plasma():
    ss = 4
    rng = np.random.default_rng(14)
    cx, cy = 36.0, 8.0
    yy, xx = np.mgrid[0:H * ss, 0:W * ss] / ss + 0.5 / ss
    glass = np.exp(-np.hypot(xx - cx, (yy - cy) * 2.5) / 30)[..., None] * rgb(40, 0, 60) + rgb(3, 0, 8)
    core = np.exp(-np.hypot(xx - cx, (yy - cy) * 1.2) / 1.3)[..., None] * rgb(255, 210, 255)
    tendrils = []
    for k in range(9):
        harmonics = [(int(rng.integers(2, 6)), int(rng.integers(1, 4)) * int(rng.choice([-1, 1])), rng.uniform(0, 6.28),
                      rng.uniform(0.4, 1.0)) for _ in range(4)]
        tendrils.append((2 * math.pi * k / 9 + rng.uniform(-0.25, 0.25), int(rng.integers(1, 3)), rng.uniform(0, 6.28),
                         harmonics))
    target = (56.0, -1.0)
    spot = np.exp(-np.hypot(xx - target[0], yy) / 1.5)[..., None] * rgb(255, 230, 255)

    def draw(phase, touch=0.0, crackle=None):
        canvas = glass.copy()
        p = 2 * math.pi * phase
        for base, wander, ph, harmonics in tendrils:
            ang = base + 0.35 * math.sin(wander * p + ph)
            # Free ends roam an ellipse beyond the wide screen; a touch pulls them all to one point.
            pull = touch * 0.92
            ex = (cx + math.cos(ang) * 40) * (1 - pull) + target[0] * pull
            ey = (cy + math.sin(ang) * 10.5) * (1 - pull) + target[1] * pull
            vx, vy = ex - cx, ey - cy
            length = math.hypot(vx, vy)
            nx, ny = -vy / length, vx / length
            steps = int(length * ss * 1.2)
            flicker = 0.75 + 0.25 * math.sin(p * 13 + ph * 3)
            for si in range(steps + 1):
                s = si / steps
                disp = sum(a * math.sin(2 * math.pi * f * s + 2 * math.pi * m * phase + q) for f, m, q, a in harmonics)
                disp *= math.sin(math.pi * min(1.0, s * 1.15)) * 1.8 * (1 - 0.35 * touch)
                if crackle is not None:
                    disp += crackle.normal(0, 0.35) * touch
                fade = (1 - s * 0.55) * flicker * (1 + 1.5 * touch)
                col = rgb(255, 120, 230) * (1 - s) + rgb(150, 90, 255) * s
                splat(canvas, cx + vx * s + nx * disp, cy + vy * s + ny * disp, col * fade * ss * 0.5, ss)
        canvas += core * (1.6 + touch) + spot * touch * 2.5
        return bloom(down(canvas, ss), 2, 0.7)

    n = IDLE_S * FPS
    idle = [draw(i / n) for i in range(n)]
    action = []
    dur = 3.4
    cr = np.random.default_rng(15)
    for i in range(int(dur * FPS)):
        t = i / FPS
        action.append(draw(t / IDLE_S, touch=envelope(t, 0.3, 1.6, 0.9), crackle=cr))
    return idle, blend_tail(action, idle, 16)


# ---------------------------------------------------------------- output

THEMES = {
    "inferno": theme_inferno,
    "hyperspace": theme_hyperspace,
    "aurora": theme_aurora,
    "rain": theme_rain,
    "synthwave": theme_synthwave,
    "lava": theme_lava,
    "matrix": theme_matrix,
    "fireworks": theme_fireworks,
    "blackhole": theme_blackhole,
    "train": theme_train,
    "orbit": theme_orbit,
    "slime": theme_slime,
    "plasma": theme_plasma,
}


def save(name, idle, action):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    frames = idle + action
    for i, f in enumerate(frames):
        Image.fromarray((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(out / f"{i}.png")
    meta = {
        "fps": FPS,
        "color_mode": "rgb888",
        "sections": [
            {"name": "idle", "start": 0, "end": len(idle) - 1},
            {"name": "action", "start": len(idle), "end": len(frames) - 1},
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta))
    # Contact sheet for review: every 15th frame of idle, then of action, 8x upscaled.
    picks = frames[0:len(idle):len(idle) // 6] + frames[len(idle)::max(1, len(action) // 6)]
    sheet = Image.new("RGB", (W * 8, (H * 8 + 6) * len(picks)), (50, 50, 50))
    for k, f in enumerate(picks):
        im = Image.fromarray((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").resize((W * 8, H * 8), Image.NEAREST)
        sheet.paste(im, (0, k * (H * 8 + 6)))
    sheet.save(OUT / f"{name}_sheet.png")
    print(f"{name}: idle {len(idle)} + action {len(action)} frames ({len(action) / FPS:.1f}s)")


def main():
    names = sys.argv[1:] or list(THEMES)
    for name in names:
        idle, action = THEMES[name]()
        save(name, idle, action)


if __name__ == "__main__":
    main()
