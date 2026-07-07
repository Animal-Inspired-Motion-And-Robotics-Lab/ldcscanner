# ---------------------------------------------------------------------------
# scanner/surface.py — Surface geometry (pure functions)
#
# Pure helpers that build the 3D ribbon "curtain" mesh and decide whether R_p is
# usable.  No Qt — the results are consumed both by the live GL view and by the
# snapshot exporter.
# ---------------------------------------------------------------------------

import numpy as np

from scanner.config import RP_ZERO_EPSILON, RP_ZERO_FALLBACK_WINDOW


def has_usable_rp(rp_vals, eps=RP_ZERO_EPSILON):
    """True when recent R_p values vary by more than ``eps`` (i.e. not flat)."""
    vals = np.asarray(rp_vals, dtype=float)
    if vals.size < 2:
        return False
    window = min(int(RP_ZERO_FALLBACK_WINDOW), vals.size)
    recent = vals[-window:]
    recent = recent[np.isfinite(recent)]
    if recent.size < 2:
        return False
    return float(np.ptp(recent)) > float(eps)


def build_surface_data(x_vals, rp_vals, l_vals):
    """Build the 3D ribbon "curtain" mesh between the live trace and a floor.

    Returns ``(vertices, faces, face_colors, line_pos)`` or ``None`` when there
    are too few points.  Each axis is robustly normalized so it stays active
    even with outliers present.
    """
    if len(x_vals) < 2:
        return None

    x_recent = np.asarray(x_vals, dtype=float)
    y_recent = np.asarray(rp_vals, dtype=float)
    z_recent = np.asarray(l_vals, dtype=float)
    if len(x_recent) < 2:
        return None

    def normalize_centered(vals):
        # Robust scaling keeps each axis active even when outliers are present.
        p_low = float(np.percentile(vals, 5.0))
        p_high = float(np.percentile(vals, 95.0))
        center = 0.5 * (p_low + p_high)
        robust_span = p_high - p_low
        std_span = float(np.std(vals)) * 6.0
        span = max(robust_span, std_span, 1e-9)
        norm = (vals - center) / span
        return np.clip(norm, -0.5, 0.5), center, span

    x_norm, _, _ = normalize_centered(x_recent)
    y_norm, _, _ = normalize_centered(y_recent)
    z_norm, _, _ = normalize_centered(z_recent)

    z_floor = -0.5
    n = len(x_recent)

    # Build a ribbon "curtain" mesh between the live trace and a floor plane.
    # Everything below is fully vectorized: these arrays are rebuilt on every
    # redraw frame, so per-point Python loops here are what make the GUI lag as
    # the buffers fill — numpy keeps the cost flat regardless of point count.
    vertices = np.empty((2 * n, 3), dtype=np.float32)
    vertices[0::2, 0] = x_norm
    vertices[0::2, 1] = y_norm
    vertices[0::2, 2] = z_floor
    vertices[1::2, 0] = x_norm
    vertices[1::2, 1] = y_norm
    vertices[1::2, 2] = z_norm

    # Two triangles per quad: [b,b+1,b+2] and [b+1,b+3,b+2] with b = 2*i.
    b = (2 * np.arange(n - 1, dtype=np.uint32))
    faces = np.empty((2 * (n - 1), 3), dtype=np.uint32)
    faces[0::2, 0] = b
    faces[0::2, 1] = b + 1
    faces[0::2, 2] = b + 2
    faces[1::2, 0] = b + 1
    faces[1::2, 1] = b + 3
    faces[1::2, 2] = b + 2

    z_color = z_norm + 0.5

    # Per-quad colour interpolates the two edge z-colours, then both triangles
    # of the quad share it.
    c = 0.5 * (z_color[:-1] + z_color[1:])
    r = 0.1 + 0.9 * c
    g = 0.5 * (1.0 - c)
    bcol = 1.0 - 0.8 * c
    face_colors = np.empty((faces.shape[0], 4), dtype=np.float32)
    face_colors[0::2, 0] = r
    face_colors[1::2, 0] = r
    face_colors[0::2, 1] = g
    face_colors[1::2, 1] = g
    face_colors[0::2, 2] = bcol
    face_colors[1::2, 2] = bcol
    face_colors[:, 3] = 0.32

    line_pos = np.column_stack((x_norm, y_norm, z_norm)).astype(np.float32)
    return vertices, faces, face_colors, line_pos
