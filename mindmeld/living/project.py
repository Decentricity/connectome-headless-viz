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
    return norm_uv(np.stack([x1, y2], axis=1)), z2.astype(np.float32)


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
