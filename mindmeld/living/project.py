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
    pad_frac: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orbit project with optional shared normalization frame.

    Returns (uv, depth, uv_lo, uv_hi) where lo/hi are the *unpadded* bounds used
    for the frame (so callers can reuse them exactly for wires + points).
    """
    rot = rotate_yaw_pitch(xyz, yaw, pitch, center=center)
    uv_raw = rot[:, :2]
    if uv_lo is None or uv_hi is None:
        uv_lo = uv_raw.min(axis=0)
        uv_hi = uv_raw.max(axis=0)
    span = np.maximum(uv_hi - uv_lo, 1e-6)
    pad = float(pad_frac) * span
    lo = uv_lo - pad
    hi = uv_hi + pad
    span2 = np.maximum(hi - lo, 1e-6)
    uv = ((uv_raw - lo) / span2).astype(np.float32)
    return uv, rot[:, 2].astype(np.float32), np.asarray(uv_lo, dtype=np.float32), np.asarray(uv_hi, dtype=np.float32)


def aabb_wire_segments(
    lo: np.ndarray,
    hi: np.ndarray,
    div: int = 4,
    *,
    face_lattice: bool = True,
) -> np.ndarray:
    """3D grid on the axis-aligned box [lo, hi]. Shape [S,2,3]."""
    lo = np.asarray(lo, dtype=np.float32).reshape(3)
    hi = np.asarray(hi, dtype=np.float32).reshape(3)
    div = max(1, int(div))
    segs: list[tuple[np.ndarray, np.ndarray]] = []

    def add(a, b):
        segs.append((np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))

    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    for i, j in (
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
    ):
        add(corners[i], corners[j])

    if face_lattice and div >= 2:
        for i in range(1, div):
            tx = x0 + (x1 - x0) * (i / div)
            ty = y0 + (y1 - y0) * (i / div)
            tz = z0 + (z1 - z0) * (i / div)
            # bottom / top
            add((tx, y0, z0), (tx, y1, z0))
            add((x0, ty, z0), (x1, ty, z0))
            add((tx, y0, z1), (tx, y1, z1))
            add((x0, ty, z1), (x1, ty, z1))
            # verticals on sides
            add((tx, y0, z0), (tx, y0, z1))
            add((tx, y1, z0), (tx, y1, z1))
            add((x0, ty, z0), (x0, ty, z1))
            add((x1, ty, z0), (x1, ty, z1))
            # constant-z rings on front/back
            add((x0, y0, tz), (x1, y0, tz))
            add((x0, y1, tz), (x1, y1, tz))
            add((x0, y0, tz), (x0, y1, tz))
            add((x1, y0, tz), (x1, y1, tz))

    return np.stack(segs, axis=0).astype(np.float32)


def unit_cube_wire_segments(div: int = 6) -> np.ndarray:
    """Deprecated alias — unit cube; prefer aabb_wire_segments on neuron bounds."""
    return aabb_wire_segments(np.zeros(3, np.float32), np.ones(3, np.float32), div=div)


def orbit_frame_for_cloud(
    xyz: np.ndarray, yaw: float, pitch: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared orbit frame from neuron AABB box + points (aligned wires).

    Returns (center, uv_lo, uv_hi, segments) — outer bounding-box edges only.
    """
    center = xyz.mean(axis=0).astype(np.float32)
    lo = xyz.min(axis=0).astype(np.float32)
    hi = xyz.max(axis=0).astype(np.float32)
    # Expand 2% so the cage sits just outside the cloud
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - 0.02 * span
    hi = hi + 0.02 * span
    segs = aabb_wire_segments(lo, hi, div=1, face_lattice=False)
    # Frame from neurons + all wire endpoints so box edges aren't warped
    ends = segs.reshape(-1, 3)
    pts = np.concatenate([xyz, ends], axis=0)
    rot = rotate_yaw_pitch(pts, yaw, pitch, center=center)
    uv_lo = rot[:, :2].min(axis=0).astype(np.float32)
    uv_hi = rot[:, :2].max(axis=0).astype(np.float32)
    return center, uv_lo, uv_hi, segs


def project_wire_segments(
    segments: np.ndarray,
    yaw: float,
    pitch: float,
    *,
    center: np.ndarray,
    uv_lo: np.ndarray,
    uv_hi: np.ndarray,
    pad_frac: float = 0.04,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Project 3D wire segments into shared orbit UV frame. Returns list of (uv_a, uv_b)."""
    out = []
    for a, b in segments:
        pts = np.stack([a, b], axis=0)
        uv, _, _, _ = project_orbit_framed(
            pts, yaw, pitch, center=center, uv_lo=uv_lo, uv_hi=uv_hi, pad_frac=pad_frac
        )
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
