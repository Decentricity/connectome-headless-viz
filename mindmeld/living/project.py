"""Shared projection + activity amplitude for caca and cinema renderers."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import FrameState, LivingEngine


@dataclass(frozen=True)
class Panel:
    name: str
    x0: int
    y0: int
    w: int
    h: int
    kind: str  # "xy" | "xz" | "yz" | "orbit"


def viz_xyz(engine: LivingEngine) -> np.ndarray:
    xyz = engine.viz.get("xyz")
    if xyz is not None and len(xyz) == engine.viz["n"]:
        return np.asarray(xyz, dtype=np.float32)
    xy = np.asarray(engine.viz["xy"], dtype=np.float32)
    z = np.full((len(xy), 1), 0.5, dtype=np.float32)
    return np.concatenate([xy, z], axis=1)


def project_plane(xyz: np.ndarray, plane: str) -> np.ndarray:
    if plane == "xy":
        uv = xyz[:, [0, 1]]
    elif plane == "xz":
        uv = xyz[:, [0, 2]]
    elif plane == "yz":
        uv = xyz[:, [1, 2]]
    else:
        raise ValueError(plane)
    return norm_uv(uv)


def project_orbit(xyz: np.ndarray, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    uv, depth, _, _ = project_orbit_framed(xyz, yaw, pitch)
    return uv, depth


def rotate_yaw_pitch(xyz: np.ndarray, yaw: float, pitch: float, center: np.ndarray | None = None) -> np.ndarray:
    """Return Nx3 rotated coordinates (yaw about Z, then pitch about X)."""
    c = xyz.mean(axis=0) if center is None else np.asarray(center, dtype=np.float32)
    p = xyz - c
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    x1 = cy * x - sy * y
    y1 = sy * x + cy * y
    z1 = z
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1
    return np.stack([x1, y2, z2], axis=1).astype(np.float32)


def project_orbit_framed(
    xyz: np.ndarray,
    yaw: float,
    pitch: float,
    *,
    center: np.ndarray | None = None,
    uv_lo: np.ndarray | None = None,
    uv_hi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orbit project with optional shared normalization frame.

    Returns (uv, depth, uv_lo, uv_hi). When lo/hi omitted, derived from xyz.
    """
    rot = rotate_yaw_pitch(xyz, yaw, pitch, center=center)
    uv_raw = rot[:, :2]
    if uv_lo is None or uv_hi is None:
        uv_lo = uv_raw.min(axis=0)
        uv_hi = uv_raw.max(axis=0)
    span = np.maximum(uv_hi - uv_lo, 1e-6)
    # slight padding so grid/box edges stay visible
    pad = 0.06 * span
    lo = uv_lo - pad
    hi = uv_hi + pad
    span2 = np.maximum(hi - lo, 1e-6)
    uv = ((uv_raw - lo) / span2).astype(np.float32)
    return uv, rot[:, 2].astype(np.float32), uv_lo.astype(np.float32), uv_hi.astype(np.float32)


def unit_cube_wire_segments(div: int = 6) -> np.ndarray:
    """3D grid on the unit cube [0,1]^3: cube edges + face lattice. Shape [S,2,3]."""
    div = max(2, int(div))
    segs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    def add(a, b):
        segs.append((a, b))

    # Outer cube edges
    corners = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        add(corners[i], corners[j])

    # Face grids (skip edges already drawn)
    for i in range(1, div):
        t = i / div
        # bottom z=0 and top z=1
        add((t, 0, 0), (t, 1, 0))
        add((0, t, 0), (1, t, 0))
        add((t, 0, 1), (t, 1, 1))
        add((0, t, 1), (1, t, 1))
        # sides
        add((t, 0, 0), (t, 0, 1))
        add((t, 1, 0), (t, 1, 1))
        add((0, t, 0), (0, t, 1))
        add((1, t, 0), (1, t, 1))
        add((0, 0, t), (1, 0, t))
        add((0, 1, t), (1, 1, t))
        add((0, 0, t), (0, 1, t))
        add((1, 0, t), (1, 1, t))

    return np.asarray(segs, dtype=np.float32)


def project_wire_segments(
    segments: np.ndarray,
    yaw: float,
    pitch: float,
    *,
    center: np.ndarray,
    uv_lo: np.ndarray,
    uv_hi: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Project 3D wire segments into shared orbit UV frame. Returns list of (uv_a, uv_b)."""
    out = []
    for a, b in segments:
        pts = np.stack([a, b], axis=0)
        uv, _, _, _ = project_orbit_framed(pts, yaw, pitch, center=center, uv_lo=uv_lo, uv_hi=uv_hi)
        out.append((uv[0], uv[1]))
    return out


def norm_uv(uv: np.ndarray) -> np.ndarray:
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return ((uv - lo) / span).astype(np.float32)


def triad_panels(width: int, height: int) -> list[Panel]:
    mid_x = width // 2
    mid_y = height // 2
    return [
        Panel("XY", 0, 0, max(1, mid_x - 1), max(1, mid_y - 1), "xy"),
        Panel("XZ", mid_x + 1, 0, max(1, width - mid_x - 1), max(1, mid_y - 1), "xz"),
        Panel("YZ", 0, mid_y + 1, max(1, mid_x - 1), max(1, height - mid_y - 1), "yz"),
        Panel("ORB", mid_x + 1, mid_y + 1, max(1, width - mid_x - 1), max(1, height - mid_y - 1), "orbit"),
    ]


def activity_amp(engine: LivingEngine, frame: FrameState) -> tuple[np.ndarray, np.ndarray]:
    """Return (|activity|, trailed amplitude in [0,1]). Caches on frame for multi-renderer ticks."""
    cached = getattr(frame, "_amp_cache", None)
    if cached is not None:
        return cached
    act = np.abs(frame.activity).astype(np.float32)
    if act.size == 0:
        out = (act, act)
        setattr(frame, "_amp_cache", out)
        return out
    prev = np.abs(engine.prev_activity)
    if prev.shape != act.shape:
        prev = np.zeros_like(act)
    trail = 0.72 * prev + 0.28 * act
    amp = trail / (np.percentile(trail, 95) + 1e-6)
    amp = np.clip(amp, 0, 1).astype(np.float32)
    out = (act, amp)
    setattr(frame, "_amp_cache", out)
    return out


def commit_trail(engine: LivingEngine, frame: FrameState) -> None:
    act, _ = activity_amp(engine, frame)
    engine.prev_activity = act.astype(np.float32, copy=True)
