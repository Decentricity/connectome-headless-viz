"""Compose visualization buffers from reservoir activity."""
from __future__ import annotations

import math

import numpy as np

from .engine import FrameState, LivingEngine


def activity_grid(engine: LivingEngine, frame: FrameState, width: int, height: int) -> np.ndarray:
    """Rasterize neuron activity into a [H,W] float grid in [0,1]."""
    act = np.abs(frame.activity)
    if act.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    prev = np.abs(engine.prev_activity)
    trail = 0.72 * prev + 0.28 * act
    amp = trail / (np.percentile(trail, 95) + 1e-6)
    amp = np.clip(amp, 0, 1)

    if getattr(engine, "camera", "triad") == "orbit":
        grid = _orbit_grid(engine, frame, amp, width, height)
    else:
        grid = _triad_grid(engine, frame, amp, width, height)

    engine.prev_activity = act.astype(np.float32, copy=True)
    return np.clip(grid, 0, 1)


def _viz_xyz(engine: LivingEngine) -> np.ndarray:
    xyz = engine.viz.get("xyz")
    if xyz is not None and len(xyz) == engine.viz["n"]:
        return np.asarray(xyz, dtype=np.float32)
    xy = np.asarray(engine.viz["xy"], dtype=np.float32)
    z = np.full((len(xy), 1), 0.5, dtype=np.float32)
    return np.concatenate([xy, z], axis=1)


def project_plane(xyz: np.ndarray, plane: str) -> np.ndarray:
    """Orthographic plane → UV in [0,1]."""
    if plane == "xy":
        uv = xyz[:, [0, 1]]
    elif plane == "xz":
        uv = xyz[:, [0, 2]]
    elif plane == "yz":
        uv = xyz[:, [1, 2]]
    else:
        raise ValueError(plane)
    return _norm_uv(uv)


def project_orbit(xyz: np.ndarray, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Yaw (about Z) + pitch (about X) orthographic projection → (uv[0,1], depth)."""
    c = xyz.mean(axis=0)
    p = xyz - c
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    x1 = cy * x - sy * y
    y1 = sy * x + cy * y
    z1 = z
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1
    uv = _norm_uv(np.stack([x1, y2], axis=1))
    return uv, z2.astype(np.float32)


def _norm_uv(uv: np.ndarray) -> np.ndarray:
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return ((uv - lo) / span).astype(np.float32)


def _triad_grid(engine: LivingEngine, frame: FrameState, amp: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.float32)
    xyz = _viz_xyz(engine)
    # Layout: 2x2 — XY | XZ / YZ | (hint)
    mid_x = width // 2
    mid_y = height // 2
    panels = (
        ("xy", 0, 0, mid_x - 1, mid_y - 1),
        ("xz", mid_x + 1, 0, width - mid_x - 1, mid_y - 1),
        ("yz", 0, mid_y + 1, mid_x - 1, height - mid_y - 1),
    )
    for plane, x0, y0, pw, ph in panels:
        if pw < 8 or ph < 4:
            continue
        uv = project_plane(xyz, plane)
        _raster_into(grid, uv, amp, engine, frame, x0, y0, pw, ph)
        _label(grid, x0 + 1, y0, plane.upper())

    # Bottom-right: miniature orbit peek at current yaw/pitch
    ox0, oy0 = mid_x + 1, mid_y + 1
    ow, oh = width - mid_x - 1, height - mid_y - 1
    if ow >= 8 and oh >= 4:
        uv, depth = project_orbit(xyz, float(engine.yaw), float(engine.pitch))
        _raster_into(grid, uv, amp, engine, frame, ox0, oy0, ow, oh, depth=depth)
        _label(grid, ox0 + 1, oy0, "ORB")

    # Separators
    if 0 <= mid_x < width:
        grid[:, mid_x] = np.maximum(grid[:, mid_x], 0.35)
    if 0 <= mid_y < height:
        grid[mid_y, :] = np.maximum(grid[mid_y, :], 0.35)
    return grid


def _orbit_grid(engine: LivingEngine, frame: FrameState, amp: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.float32)
    xyz = _viz_xyz(engine)
    uv, depth = project_orbit(xyz, float(engine.yaw), float(engine.pitch))
    _raster_into(grid, uv, amp, engine, frame, 0, 0, width, height, depth=depth)
    deg_y = int(round(math.degrees(float(engine.yaw)))) % 360
    deg_p = int(round(math.degrees(float(engine.pitch))))
    _label(grid, 1, 0, f"Y{deg_y} P{deg_p}")
    return grid


def _raster_into(
    grid: np.ndarray,
    uv: np.ndarray,
    amp: np.ndarray,
    engine: LivingEngine,
    frame: FrameState,
    x0: int,
    y0: int,
    pw: int,
    ph: int,
    depth: np.ndarray | None = None,
):
    xs = np.clip((uv[:, 0] * (pw - 1)).astype(np.int32), 0, pw - 1) + x0
    ys = np.clip((uv[:, 1] * (ph - 1)).astype(np.int32), 0, ph - 1) + y0
    order = np.arange(len(amp))
    if depth is not None:
        order = np.argsort(depth)  # far → near

    if frame.view == "region":
        rid = engine.viz["region_id"].astype(np.float32)
        nreg = max(1, len(engine.viz.get("region_names", [""])))
        val = (rid / nreg) * 0.55 + 0.45 * amp
        if depth is None:
            np.maximum.at(grid, (ys, xs), val)
        else:
            for i in order:
                y, x = int(ys[i]), int(xs[i])
                if grid[y, x] < val[i]:
                    grid[y, x] = val[i]
    elif frame.view == "heatmap":
        sub = np.zeros((ph, pw), dtype=np.float32)
        np.add.at(sub, (ys - y0, xs - x0), amp)
        sub = _box_blur(sub, 1)
        m = sub.max() + 1e-6
        sub /= m
        np.maximum(grid[y0 : y0 + ph, x0 : x0 + pw], sub, out=grid[y0 : y0 + ph, x0 : x0 + pw])
    elif frame.view == "storm":
        prev = np.abs(engine.prev_activity)
        act = np.abs(frame.activity)
        delta = np.clip(act - prev, 0, None)
        if delta.max() > 0:
            d = np.clip(delta / (np.percentile(delta, 90) + 1e-6), 0, 1)
        else:
            d = amp
        if depth is None:
            np.maximum.at(grid, (ys, xs), d)
        else:
            for i in order:
                y, x = int(ys[i]), int(xs[i])
                if grid[y, x] < d[i]:
                    grid[y, x] = d[i]
        step_e = max(1, max(1, len(engine.edges)) // 800) if len(engine.edges) else 1
        for a, b in engine.edges[::step_e]:
            if amp[a] > 0.25 or amp[b] > 0.25:
                _draw_line(grid, xs[a], ys[a], xs[b], ys[b], 0.35 * max(amp[a], amp[b]))
    else:  # whole
        if depth is None:
            np.maximum.at(grid, (ys, xs), amp)
        else:
            for i in order:
                y, x = int(ys[i]), int(xs[i])
                if grid[y, x] < amp[i]:
                    grid[y, x] = amp[i]
        step_e = max(1, len(engine.edges) // 500) if len(engine.edges) else 1
        for a, b in engine.edges[::step_e]:
            strength = 0.15 * min(amp[a], amp[b])
            if strength > 0.02:
                _draw_line(grid, xs[a], ys[a], xs[b], ys[b], strength)


def _label(grid: np.ndarray, x: int, y: int, text: str):
    # Soft bright ticks so labels read on dark panels without fonts.
    for i, _ch in enumerate(text[:8]):
        xx = x + i
        if 0 <= y < grid.shape[0] and 0 <= xx < grid.shape[1]:
            grid[y, xx] = max(grid[y, xx], 0.85)


def _box_blur(g: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return g
    out = g.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            out += np.roll(np.roll(g, dy, axis=0), dx, axis=1)
    return out / float((2 * r + 1) ** 2)


def _draw_line(grid, x0, y0, x1, y1, val):
    n = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    for i in range(n + 1):
        t = i / n
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            if grid[y, x] < val:
                grid[y, x] = val
