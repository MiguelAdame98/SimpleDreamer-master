#!/usr/bin/env python3
"""
overlay_cog_env.py

Batch (CLI) usage examples:

# 1) Use the pre-rendered transparent cog PNG (fast path)
python overlay_cog_env.py \
  --snapshot dbg/cogmap/t0042/snapshot_t0042.json \
  --out dbg/cogmap/t0042/overlay_manual.png \
  --alpha 0.65 --deg 90 --mirror x --scale 1.0 --tx 0 --ty 0

# 2) Re-draw vectors from JSON (ignores cog.png), with new node styling
python overlay_cog_env.py \
  --snapshot dbg/cogmap/t0042/snapshot_t0042.json \
  --out dbg/cogmap/t0042/overlay_vectors.png \
  --alpha 0.75 --deg -30 --scale 1.2 --tx 3 --ty -2 --vector

Interactive usage (recommended for manual fitting):

python overlay_cog_env.py \
  --snapshot dbg/cogmap/t0042/snapshot_t0042.json \
  --interactive --out dbg/cogmap/t0042/overlay_interactive.png

Controls (interactive):
- Move: ← ↑ ↓ → buttons (step = 0.5 grid units)
- Rotate: ⟲ -5°, ⟳ +5°, and ±90° buttons
- Mirror: toggle X, toggle Y
- Sliders: Scale (0.25–3.0), Alpha (0–1)
- Save: writes current view to --out (or auto-named alongside snapshot)
- Reset: returns to initial params (deg=0, tx=ty=0, scale=1, no mirroring)
"""

import json
import argparse
from pathlib import Path
import numpy as np

# --- Pure-logic helpers (no pyplot/backend here) ---

def load_snapshot(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_affine(deg: float, mirror: str, scale: float, tx: float, ty: float):
    """
    Returns (sx, sy, deg, tx, ty) components; actual Affine2D is built inside draw code.
    mirror ∈ {"none","x","y","xy","yx"}.
    """
    sx, sy = 1.0, 1.0
    m = (mirror or "none").lower()
    if m in ("x", "xy", "yx"):
        sx *= -1.0
    if m in ("y", "xy", "yx"):
        sy *= -1.0
    return sx * scale, sy * scale, float(deg), float(tx), float(ty)


# --- Batch drawing paths (non-interactive) ---

def overlay_with_raster(ax, env, cog, deg, mirror, scale, tx, ty, alpha, mpl_modules):
    mpimg, transforms = mpl_modules
    env_img = mpimg.imread(env["img_path"])
    Wc, Hc = env["grid_size"]
    ax.imshow(env_img, origin="lower", extent=(0, Wc, 0, Hc))
    ax.set_xlim(0, Wc); ax.set_ylim(0, Hc); ax.set_aspect("equal", "box")

    if not cog.get("png_path"):
        print("[overlay] No cog PNG available; falling back to --vector draw. Use --vector.")
        return

    bbox = cog.get("bbox", [0, 1, 0, 1])
    x0, x1, y0, y1 = bbox
    cog_img = mpimg.imread(cog["png_path"])

    sx, sy, d, tx, ty = build_affine(deg, mirror, scale, tx, ty)
    T = transforms.Affine2D().scale(sx, sy).rotate_deg(d).translate(tx, ty)

    ax.imshow(
        cog_img, origin="lower", extent=(x0, x1, y0, y1),
        alpha=alpha, transform=T + ax.transData, interpolation="bilinear",
    )

def overlay_with_vectors(ax, env, cog, deg, mirror, scale, tx, ty, alpha, annotate_ids, mpl_modules):
    mpimg, transforms, LineCollection = mpl_modules
    env_img = mpimg.imread(env["img_path"])
    Wc, Hc = env["grid_size"]
    ax.imshow(env_img, origin="lower", extent=(0, Wc, 0, Hc))
    ax.set_xlim(0, Wc); ax.set_ylim(0, Hc); ax.set_aspect("equal", "box")

    # Build transform
    sx, sy, d, tx, ty = build_affine(deg, mirror, scale, tx, ty)
    T = transforms.Affine2D().scale(sx, sy).rotate_deg(d).translate(tx, ty)

    def apply_T(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return T.transform(X)

    # Links (as a LineCollection)
    links = cog.get("links_xy", []) or []
    segs = []
    for i in range(0, len(links) - 1, 2):
        a = np.asarray(links[i], dtype=float)
        b = np.asarray(links[i+1], dtype=float)
        A = apply_T(a)[0]; B = apply_T(b)[0]
        segs.append([A, B])
    if segs:
        lc = LineCollection(segs, colors="k", linewidths=1.6, alpha=alpha, zorder=2)
        ax.add_collection(lc)

    # Nodes (white fill, black edge circles)
    from matplotlib import patheffects as pe
    nodes = cog.get("nodes", []) or []
    if nodes:
        pts = np.array([[n["x"], n["y"]] for n in nodes], dtype=float)
        P = apply_T(pts)
        ax.scatter(P[:,0], P[:,1],
                   s=120, marker="o",
                   facecolors="white", edgecolors="black",
                   linewidths=1.5, alpha=alpha, zorder=3)
        # Always annotate ids (if missing, show index)
        for i, (n, p) in enumerate(zip(nodes, P)):
            nid = n.get("id")
            label = str(nid if (nid is not None) else i)
            txt = ax.text(p[0], p[1], label,
                          fontsize=10, color="black",
                          ha="center", va="center", zorder=4,
                          alpha=min(1.0, alpha+0.2))
            # White halo for legibility
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    # Current node marker
    cur_id = cog.get("current_exp_id", None)
    if (cur_id is not None) and nodes:
        for (n, p) in zip(nodes, P):
            if n.get("id") == cur_id:
                ax.plot([p[0]], [p[1]], marker="x", color="r", mew=2, ms=9, alpha=min(1.0, alpha+0.1), zorder=5)
                break


# --- Interactive app ---

def run_interactive(snapshot_path: str, out_path: str|None, annotate_ids: bool):
    import matplotlib
    # Use default interactive backend; DO NOT force Agg here.
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    import matplotlib.image as mpimg
    from matplotlib.widgets import Button, Slider
    from matplotlib.collections import LineCollection
    import time

    snap = load_snapshot(snapshot_path)
    env, cog = snap["env"], snap["cog"]
    Wc, Hc = env["grid_size"]

    # Prepare base data
    nodes = cog.get("nodes", []) or []

    # Points in COG (emap units)
    pts_cog = np.array([[n["x"], n["y"]] for n in nodes], dtype=float) if nodes else np.zeros((0,2))

    # Points in REAL (env cell coords from real_pose[:2]); fallback to COG if missing
    pts_real = []
    for i, n in enumerate(nodes):
        rp = n.get("real_pose")
        if isinstance(rp, (list, tuple)) and len(rp) >= 2:
            pts_real.append([float(rp[0]), float(rp[1])])
        else:
            # fallback to COG so we never lose a node
            pts_real.append([float(n.get("x", 0.0)), float(n.get("y", 0.0))])
    pts_real = np.array(pts_real, dtype=float) if nodes else np.zeros((0,2))

    # Links provided as coordinate pairs in COG space; build two versions:
    links = cog.get("links_xy", []) or []
    seg_pairs_cog = np.array(links, dtype=float).reshape(-1,2,2) if len(links) >= 2 else np.zeros((0,2,2))

    # Remap COG link endpoints to REAL by snapping to nearest node (in COG)
    def remap_links_to_real(seg_pairs_cog, pts_cog, pts_real):
        if len(seg_pairs_cog) == 0 or len(pts_cog) == 0:
            return np.zeros((0,2,2))
        out = []
        for (a, b) in seg_pairs_cog:
            # nearest node in COG for each endpoint
            ia = int(np.argmin(np.sum((pts_cog - a)**2, axis=1)))
            ib = int(np.argmin(np.sum((pts_cog - b)**2, axis=1)))
            A = pts_real[ia] if ia >= 0 else a
            B = pts_real[ib] if ib >= 0 else b
            out.append([A, B])
        return np.array(out, dtype=float)

    seg_pairs_real = remap_links_to_real(seg_pairs_cog, pts_cog, pts_real)
    # State
    state = {
        "deg": 0.0, "mirror_x": False, "mirror_y": False,
        "scale": 1.0, "tx": 0.0, "ty": 0.0, "alpha": 0.65,
        "step": 0.5,"coord_mode": "cog",
    }

    # Figure + axes layout
    fig = plt.figure(figsize=(9.5, 7.0), dpi=120)
    ax = fig.add_axes([0.06, 0.12, 0.73, 0.84])     # main plot
    # Controls area (bottom row)
    ax_btn_left  = fig.add_axes([0.82, 0.78, 0.07, 0.06])
    ax_btn_up    = fig.add_axes([0.90, 0.86, 0.07, 0.06])
    ax_btn_down  = fig.add_axes([0.90, 0.78, 0.07, 0.06])
    ax_btn_right = fig.add_axes([0.98, 0.78, 0.07, 0.06])

    ax_btn_r_m5  = fig.add_axes([0.82, 0.66, 0.07, 0.06])
    ax_btn_r_p5  = fig.add_axes([0.90, 0.66, 0.07, 0.06])
    ax_btn_r_m90 = fig.add_axes([0.82, 0.58, 0.07, 0.06])
    ax_btn_r_p90 = fig.add_axes([0.90, 0.58, 0.07, 0.06])

    ax_btn_mx    = fig.add_axes([0.82, 0.46, 0.07, 0.06])
    ax_btn_my    = fig.add_axes([0.90, 0.46, 0.07, 0.06])

    ax_sld_scale = fig.add_axes([0.82, 0.36, 0.23, 0.03])
    ax_sld_alpha = fig.add_axes([0.82, 0.30, 0.23, 0.03])

    ax_btn_reset = fig.add_axes([0.82, 0.18, 0.10, 0.06])
    ax_btn_save  = fig.add_axes([0.95, 0.18, 0.10, 0.06])
    ax_btn_coords = fig.add_axes([0.82, 0.24, 0.23, 0.06])

    # Draw env background
    env_img = mpimg.imread(env["img_path"])
    im_env = ax.imshow(env_img, origin="lower", extent=(0, Wc, 0, Hc), zorder=0)
    ax.set_xlim(0, Wc); ax.set_ylim(0, Hc); ax.set_aspect("equal", "box")
    ax.axis("off")

    # Artists (links as LineCollection, nodes as scatter)
    from matplotlib import patheffects as pe

    # Artists (links as LineCollection, nodes as scatter)
    link_coll = LineCollection([], colors="k", linewidths=1.8, zorder=2, alpha=state["alpha"])
    ax.add_collection(link_coll)

    scat = ax.scatter([], [], s=120, marker="o", facecolors="white",
                      edgecolors="black", linewidths=1.5,
                      alpha=state["alpha"], zorder=3)

    # Always-on numeric labels (with white halo)
    text_labels = []
    for i, _ in enumerate(nodes):
        txt = ax.text(0, 0, "", fontsize=10, ha="center", va="center",
                      color="black", alpha=min(1.0, state["alpha"]+0.2), zorder=4)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
        text_labels.append(txt)
    # Current node
    cur_id = cog.get("current_exp_id", None)
    cur_mark, = ax.plot([], [], marker="x", color="r", mew=2, ms=9,
                        alpha=min(1.0, state["alpha"]+0.1), zorder=5)

    def mirror_string():
        if state["mirror_x"] and state["mirror_y"]: return "xy"
        if state["mirror_x"]: return "x"
        if state["mirror_y"]: return "y"
        return "none"

    def apply_transform():
        sx, sy, d, tx, ty = build_affine(
            deg=state["deg"], mirror=mirror_string(),
            scale=state["scale"], tx=state["tx"], ty=state["ty"]
        )
        T = Affine2D().scale(sx, sy).rotate_deg(d).translate(tx, ty)
        return T

    def redraw():
        # transform
        T = apply_transform()

        # pick coord set
        if state["coord_mode"] == "real":
            local_pts = pts_real
            local_segs = seg_pairs_real
        else:
            local_pts = pts_cog
            local_segs = seg_pairs_cog

        # links
        if len(local_segs):
            A = local_segs.reshape(-1, 2)
            B = T.transform(A).reshape(-1, 2, 2)
            link_coll.set_segments(B)
            link_coll.set_alpha(state["alpha"])
        else:
            link_coll.set_segments([])

        # nodes
        if len(local_pts):
            P = T.transform(local_pts)
            scat.set_offsets(P)
            scat.set_alpha(state["alpha"])
            for i, n in enumerate(nodes):
                label = str(n.get("id", i))
                text_labels[i].set_text(label)
                if len(P) > i:
                    text_labels[i].set_position((P[i,0], P[i,1]))
                    text_labels[i].set_alpha(min(1.0, state["alpha"]+0.2))
        else:
            scat.set_offsets(np.zeros((0,2)))
            for txt in text_labels:
                txt.set_alpha(0.0)

        # current node (mark by id if we have it; uses whichever coord set is active)
        cur_id = cog.get("current_exp_id", None)
        if (cur_id is not None) and len(nodes) and len(local_pts):
            # find index by id; fallback to same index if missing id
            idx = None
            for i, n in enumerate(nodes):
                if n.get("id") == cur_id:
                    idx = i; break
            if idx is not None and idx < len(local_pts):
                Pc = T.transform(local_pts[idx])
                cur_mark.set_data([Pc[0]], [Pc[1]])
                cur_mark.set_alpha(min(1.0, state["alpha"]+0.1))
            else:
                cur_mark.set_data([], [])
        else:
            cur_mark.set_data([], [])

        fig.canvas.draw_idle()

    # --- Widgets ---
    btn_left  = Button(ax_btn_left,  "←")
    btn_up    = Button(ax_btn_up,    "↑")
    btn_down  = Button(ax_btn_down,  "↓")
    btn_right = Button(ax_btn_right, "→")

    btn_r_m5  = Button(ax_btn_r_m5,  "⟲ -5°")
    btn_r_p5  = Button(ax_btn_r_p5,  "⟳ +5°")
    btn_r_m90 = Button(ax_btn_r_m90, "−90°")
    btn_r_p90 = Button(ax_btn_r_p90, "+90°")

    btn_mx = Button(ax_btn_mx, "Mirror X")
    btn_my = Button(ax_btn_my, "Mirror Y")

    sld_scale = Slider(ax_sld_scale, "Scale", 0.25, 3.0, valinit=state["scale"], valstep=0.01)
    sld_alpha = Slider(ax_sld_alpha, "Alpha", 0.0, 1.0, valinit=state["alpha"], valstep=0.01)

    btn_reset = Button(ax_btn_reset, "Reset")
    btn_save  = Button(ax_btn_save,  "Save")
    btn_coords = Button(ax_btn_coords, "Coords: COG")
    def toggle_coords(_):
        state["coord_mode"] = "real" if state["coord_mode"] == "cog" else "cog"
        btn_coords.label.set_text(f"Coords: {state['coord_mode'].upper()}")
        redraw()
    btn_coords.on_clicked(toggle_coords)

    # --- Callbacks ---
    def move(dx, dy):
        state["tx"] += dx; state["ty"] += dy; redraw()

    def rotate(dd):
        state["deg"] = (state["deg"] + dd) % 360.0; redraw()

    def toggle_mx(_):
        state["mirror_x"] = not state["mirror_x"]; redraw()

    def toggle_my(_):
        state["mirror_y"] = not state["mirror_y"]; redraw()

    def on_scale(val):
        state["scale"] = float(val); redraw()

    def on_alpha(val):
        state["alpha"] = float(val); redraw()

    def reset(_=None):
        state.update({"deg":0.0,"mirror_x":False,"mirror_y":False,
                      "scale":1.0,"tx":0.0,"ty":0.0,"alpha":0.65})
        sld_scale.set_val(state["scale"])
        sld_alpha.set_val(state["alpha"])
        redraw()

    def save(_=None):
        # Determine output path
        out = out_path
        if not out:
            base = Path(snapshot_path).parent
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = base / f"overlay_{stamp}.png"
        else:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
        # Save current view
        ax.axis("off")
        fig.savefig(out, bbox_inches="tight", pad_inches=0)
        print(f"[overlay interactive] wrote {out}")

    # Wire buttons
    btn_left.on_clicked(lambda _ : move(-state["step"], 0))
    btn_right.on_clicked(lambda _ : move( state["step"], 0))
    btn_up.on_clicked(lambda _ : move(0,  state["step"]))
    btn_down.on_clicked(lambda _ : move(0, -state["step"]))

    btn_r_m5.on_clicked(lambda _ : rotate(-5))
    btn_r_p5.on_clicked(lambda _ : rotate(+5))
    btn_r_m90.on_clicked(lambda _ : rotate(-90))
    btn_r_p90.on_clicked(lambda _ : rotate(+90))

    btn_mx.on_clicked(toggle_mx)
    btn_my.on_clicked(toggle_my)

    sld_scale.on_changed(on_scale)
    sld_alpha.on_changed(on_alpha)

    btn_reset.on_clicked(reset)
    btn_save.on_clicked(save)

    # Keyboard shortcuts (optional quality-of-life)
    def on_key(event):
        if event.key == "left":   move(-state["step"], 0)
        elif event.key == "right":move( state["step"], 0)
        elif event.key == "up":   move(0,  state["step"])
        elif event.key == "down": move(0, -state["step"])
        elif event.key == "r":    rotate(+5)
        elif event.key == "R":    rotate(-5)
        elif event.key == "x":    toggle_mx(None)
        elif event.key == "y":    toggle_my(None)
        elif event.key == "s":    save(None)
        elif event.key == "0":    reset(None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial draw
    redraw()
    fig.suptitle("Overlay Interactive — arrows=move, r/R=rotate ±5°, x/y=mirror, s=save, 0=reset", fontsize=10)
    plt.show()


# --- CLI ---

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=str)
    ap.add_argument("--out", type=str, default=None, help="Output PNG path (used by batch and interactive Save)")
    ap.add_argument("--deg", type=float, default=0.0, help="Rotation in degrees (CCW)")
    ap.add_argument("--mirror", type=str, default="none", choices=["none", "x", "y", "xy", "yx"])
    ap.add_argument("--scale", type=float, default=1.0, help="Uniform scale on cog units")
    ap.add_argument("--tx", type=float, default=0.0, help="Translation in env grid units (x)")
    ap.add_argument("--ty", type=float, default=0.0, help="Translation in env grid units (y)")
    ap.add_argument("--alpha", type=float, default=0.65, help="Cog translucency")
    ap.add_argument("--vector", action="store_true", help="Re-draw graph from nodes/links (ignores cog.png)")
    ap.add_argument("--annotate-ids", action="store_true", help="Annotate node ids when using vector draw")
    ap.add_argument("--interactive", action="store_true", help="Open interactive window for live fitting & saving")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.interactive:
        # Interactive mode (GUI backend)
        run_interactive(args.snapshot, args.out, annotate_ids=args.annotate_ids)
        return

    # Batch mode: we can force Agg safely here
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    import matplotlib.image as mpimg
    from matplotlib.collections import LineCollection

    snap = load_snapshot(args.snapshot)
    env, cog = snap["env"], snap["cog"]

    fig, ax = plt.subplots(1, 1, dpi=160)
    if args.vector or not cog.get("png_path"):
        overlay_with_vectors(
            ax, env, cog, args.deg, args.mirror, args.scale, args.tx, args.ty, args.alpha,
            annotate_ids=args.annotate_ids,
            mpl_modules=(mpimg, Affine2D, LineCollection)
        )
    else:
        overlay_with_raster(
            ax, env, cog, args.deg, args.mirror, args.scale, args.tx, args.ty, args.alpha,
            mpl_modules=(mpimg, Affine2D)
        )

    ax.axis("off")
    out_path = args.out or (Path(args.snapshot).with_name("overlay.png"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[overlay] wrote {out_path}")

if __name__ == "__main__":
    main()
