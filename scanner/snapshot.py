# ---------------------------------------------------------------------------
# scanner/snapshot.py — Journal-quality PDF export of the four live plots
#
# Re-renders what is currently on screen with matplotlib (true vector, clean
# white publication style) rather than grabbing the dark-themed GL widgets.  The
# entry file wires the "Snapshot" button to ``make_snapshot`` and supplies the
# output root + a status callback, so this module never touches Qt widgets.
# ---------------------------------------------------------------------------

import os
from datetime import datetime

import numpy as np

from scanner.surface import build_surface_data

# Axis labels shared by the live view and the exported figures.
_SNAP_L_LABEL = "L (µH)"
_SNAP_RP_LABEL = "R_p (Ω)"
_SNAP_T_LABEL = "timestamp (ms)"


def make_snapshot(state, out_root, on_status=print):
    """Render the four live plots as journal-quality vector PDFs.

    Reproduces exactly what is currently on screen — Phase Space (L vs R_p),
    Time Trace (L vs t), the 3D surface trace, and the crack-event plot — using
    matplotlib so the output is true vector with a clean white (publication)
    style.  Files land in ``<out_root>/Snapshot <timestamp>/``.

    ``state`` is the :class:`~scanner.state.ScannerState` whose ``snap_*`` caches
    hold the current plot window; ``on_status`` receives human-readable progress
    / error strings (defaults to ``print``).  Returns the list of written paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    except Exception as exc:
        on_status(f"Snapshot failed: matplotlib unavailable ({exc})")
        return []

    time_arr = np.asarray(state.snap_time, dtype=float)
    rp_arr = np.asarray(state.snap_rp, dtype=float)
    l_arr = np.asarray(state.snap_l, dtype=float)
    if time_arr.size == 0:
        on_status("Snapshot skipped: no plotted data yet")
        return []

    crack_times = np.asarray(state.snap_crack_times, dtype=float)
    if state.crack_y_mode == "mag":
        crack_vals = np.asarray(state.snap_crack_mags, dtype=float)
        crack_label = "magnitude (au)"
    else:
        crack_vals = np.asarray(state.snap_crack_sizes, dtype=float)
        crack_label = "crack_size"

    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    out_dir = os.path.join(out_root, f"Snapshot {stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Publication style: white background, black text, serif labels.
    style = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "font.family": "serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "lines.linewidth": 1.2,
    }

    def draw_phase(ax):
        ax.plot(rp_arr, l_arr, color="#c0392b", linewidth=1.0)
        ax.plot(rp_arr[-1], l_arr[-1], "o", color="black", markersize=4)
        ax.set_xlabel(_SNAP_RP_LABEL)
        ax.set_ylabel(_SNAP_L_LABEL)
        ax.set_title("Phase Space")
        ax.grid(True, alpha=0.3)

    def draw_time(ax):
        ax.plot(time_arr, l_arr, color="#c0392b", linewidth=1.0)
        ax.set_xlabel(_SNAP_T_LABEL)
        ax.set_ylabel(_SNAP_L_LABEL)
        ax.set_title("Time Trace")
        ax.grid(True, alpha=0.3)

    def draw_crack(ax):
        if crack_times.size:
            ax.vlines(crack_times, 0.0, crack_vals, color="#e67e22", linewidth=1.0)
            ax.set_ylim(0.0, max(float(np.max(crack_vals)) * 1.1, 1e-6))
        ax.set_xlabel(_SNAP_T_LABEL)
        ax.set_ylabel(crack_label)
        ax.set_title("Crack Events")
        ax.grid(True, alpha=0.3)

    def draw_surface(ax):
        surf = build_surface_data(time_arr, rp_arr, l_arr)
        if surf is None:
            ax.text2D(0.5, 0.5, "Not enough data", ha="center", va="center",
                      transform=ax.transAxes)
            return
        _, faces, face_colors, _ = surf
        n = time_arr.size
        l_floor = float(np.min(l_arr))
        # Physical-coordinate "curtain" with the same per-face coloring as the
        # live GL view (face_colors order matches this vertex/face layout).
        verts = np.empty((2 * n, 3), dtype=float)
        verts[0::2, 0] = time_arr
        verts[0::2, 1] = rp_arr
        verts[0::2, 2] = l_floor
        verts[1::2, 0] = time_arr
        verts[1::2, 1] = rp_arr
        verts[1::2, 2] = l_arr
        tris = [verts[f] for f in faces]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=face_colors,
                                             edgecolors="none"))
        ax.plot(time_arr, rp_arr, l_arr, color="#e69b2a", linewidth=1.3)
        ax.set_xlim(float(time_arr.min()), float(time_arr.max()))
        ax.set_ylim(float(rp_arr.min()), float(rp_arr.max()))
        ax.set_zlim(l_floor, max(float(np.max(l_arr)), l_floor + 1e-9))
        ax.set_xlabel("Time")
        ax.set_ylabel(_SNAP_RP_LABEL)
        ax.set_zlabel(_SNAP_L_LABEL)
        ax.set_title("3D Surface Trace")
        ax.view_init(elev=22, azim=-35)

    saved = []
    try:
        with plt.rc_context(style):
            panels = [
                ("phase_space.pdf", draw_phase, (5.0, 4.0), False),
                ("time_trace.pdf", draw_time, (5.0, 4.0), False),
                ("crack_events.pdf", draw_crack, (5.0, 2.0), False),
                ("surface_3d.pdf", draw_surface, (5.5, 4.5), True),
            ]
            for name, draw, figsize, is_3d in panels:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(1, 1, 1, projection="3d") if is_3d else fig.add_subplot(1, 1, 1)
                draw(ax)
                path = os.path.join(out_dir, name)
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)

            # Combined 2x2 sheet for a single-figure submission.
            fig = plt.figure(figsize=(11.0, 8.5))
            draw_phase(fig.add_subplot(2, 2, 1))
            draw_time(fig.add_subplot(2, 2, 2))
            draw_surface(fig.add_subplot(2, 2, 3, projection="3d"))
            draw_crack(fig.add_subplot(2, 2, 4))
            fig.tight_layout()
            path = os.path.join(out_dir, "combined.pdf")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    except Exception as exc:
        on_status(f"Snapshot failed: {exc}")
        return saved

    on_status(f"Snapshot saved ({len(saved)} PDFs): {out_dir}")
    return saved
