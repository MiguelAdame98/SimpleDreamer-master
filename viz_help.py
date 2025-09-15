#!/usr/bin/env python3
"""
unified_test_runner.py

Standalone utilities for step-by-step video recording (env + GT obs + Dreamer decode +
HMM panel + experience map + pose panel), designed to be reused across scripts.

Typical usage:

    from unified_test_runner import (
        VideoGridmap, append_vis_frame_unified, record_video_frames,
        safe_get_memory_map_data_from, compute_mse
    )

    video = VideoGridmap("dbg/unified_runner.mp4", fps=10)
    env_definition = {"n_row": 5, "n_col": 5, "max_steps": 600}
    visited_rooms = []

    # ... inside your control loop:
    data = {
        "env_image": env_img,                 # HxWx3 uint8 or compatible tensor
        "ground_truth_ob": obs["image"],      # HxWx3
        "decoded": decoded_img,               # optional
        "image_predicted": predicted_img,     # optional
        "mse": compute_mse(obs["image"], decoded_img) if decoded_img is not None else 0.0,
        "GP": GP,                             # optional: global pose
        "pose": obs.get("pose"),              # optional: local pose
        # Optional HMM block:
        # "recommended_mode": hmm_mode,
        # "recommended_submode": hmm_submode,
        # "mode_confidence": 0.87,
        # "submode_confidence": 0.66,
        # "uncertainty": 0.42,
        # "changepoint_mass": 0.07,
    }

    append_vis_frame_unified(
        planner_like=planner,                 # must expose get_memory_map_data() OR planner.cog/.emap
        video_gridmap=video,
        data=data,
        env_definition=env_definition,
        visited_rooms=visited_rooms,
        step_count=t
    )

    # on exit (even if Ctrl+C):
    video.close()

Notes:
- This file is intentionally headless-safe (sets matplotlib backend to 'Agg' when needed).
- `safe_get_memory_map_data_from()` reconstructs memory_map_data if planner doesn’t expose it directly.
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Tuple, Optional
import os, shutil, subprocess


# Headless-friendly Matplotlib
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")  # headless-safe
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.pyplot as plt
import numpy as np

# OpenCV is optional but strongly recommended for MP4 writing
try:
    import cv2
except Exception:
    cv2 = None  # we’ll raise a helpful error on writer init if it’s missing


# ------------------------------ Public API ---------------------------------

__all__ = [
    "VideoGridmap",
    "dbg_stats",
    "prepare_for_imshow",
    "get_image_plot",
    "plot_MSE_bar",
    "plot_visited_rooms",
    "plot_memory_map",
    "print_positions",
    "record_video_frames",
    "safe_get_memory_map_data_from",
    "append_vis_frame_unified",
    "compute_mse",
]


# --- Minimal writer with old API (append_data / close) ---------------------
class VideoGridmap:
    """
    Browser-safe video writer.
    Prefers FFmpeg (H.264 baseline, yuv420p, faststart), falls back to OpenCV avc1/mp4v or MJPG/AVI.
    API: append_data(frame: HxWx{3,4} uint8 RGB), close()
    """
    def __init__(self, out_path: str, fps: int = 10):
        self.out_path = str(out_path)
        self.fps = int(fps)
        self._method = None          # 'ffmpeg' | 'opencv'
        self._proc = None            # FFmpeg subprocess
        self._writer = None          # cv2.VideoWriter
        self._shape = None           # (H, W)
        self._count = 0
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

        # detect ffmpeg
        self._ffmpeg = shutil.which("ffmpeg")

    # ---------- FFmpeg path ----------
    def _start_ffmpeg(self, W: int, H: int):
        base, ext = os.path.splitext(self.out_path)
        # force .mp4 for browser playback
        if ext.lower() != ".mp4":
            self.out_path = base + ".mp4"

        cmd = [
            self._ffmpeg, "-loglevel", "error", "-y",
            # raw RGB input via stdin
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}",
            "-r", str(self.fps),
            "-i", "-",
            # no audio
            "-an",
            # browser-safe H.264
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            # CFR at the same fps on the output side too
            "-r", str(self.fps),
            self.out_path,
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self._method = "ffmpeg"
            print(f"[VideoGridmap] Using FFmpeg → {self.out_path}  {W}x{H}@{self.fps}fps")
        except Exception as e:
            self._proc = None
            print("[VideoGridmap] FFmpeg failed to start:", e)

    def _ffmpeg_write(self, frame_rgb: np.ndarray):
        try:
            self._proc.stdin.write(frame_rgb.tobytes())
        except BrokenPipeError:
            raise RuntimeError("[VideoGridmap] FFmpeg pipe broken (check codec support).")

    def _ffmpeg_close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.wait()
            self._proc = None

    # ---------- OpenCV fallback ----------
    def _opencv_init(self, W: int, H: int):
        import cv2
        tried = []
        base, _ = os.path.splitext(self.out_path)

        # best → avc1 (if your OpenCV build can encode H.264)
        for fourcc_str, ext in [("avc1", "mp4"), ("mp4v", "mp4"), ("MJPG", "avi")]:
            out = f"{base}.{ext}"
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(out, fourcc, self.fps, (W, H))
            ok = bool(writer is not None and writer.isOpened())
            tried.append((fourcc_str, out, ok))
            if ok:
                self.out_path = out
                self._writer = writer
                self._shape = (H, W)
                self._method = "opencv"
                print(f"[VideoGridmap] OpenCV writer fourcc={fourcc_str} → {out} {W}x{H}@{self.fps}fps")
                if ext != "mp4":
                    print("[VideoGridmap] NOTE: AVI/MJPG won’t play in browsers. Use FFmpeg path for browser-safe MP4.")
                return

        raise RuntimeError(f"[VideoGridmap] Could not open any OpenCV VideoWriter. Tried={tried}")

    def append_data(self, frame: np.ndarray) -> None:
        if frame is None:
            print("[VideoGridmap] WARNING: None frame; skipping")
            return

        # ensure HxWx3 uint8 RGB
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        H, W, _ = frame.shape

        # lazy-init on first frame, prefer FFmpeg
        if self._method is None:
            if self._ffmpeg is not None:
                self._start_ffmpeg(W, H)
            if self._method is None:
                # FFmpeg not available or failed; try OpenCV
                self._opencv_init(W, H)

            # dump first frame for sanity
            try:
                import cv2
                dbg_png = os.path.join(os.path.dirname(self.out_path), "_first_frame_rgb.png")
                cv2.imwrite(dbg_png, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"[VideoGridmap] First-frame debug → {dbg_png} (shape={frame.shape})")
            except Exception as e:
                print("[VideoGridmap] Could not write first-frame PNG:", e)

            self._shape = (H, W)

        # size consistency (resize to the first frame’s shape)
        if (H, W) != self._shape:
            import cv2
            frame = cv2.resize(frame, (self._shape[1], self._shape[0]), interpolation=cv2.INTER_AREA)

        # write
        if self._method == "ffmpeg":
            self._ffmpeg_write(frame)  # rgb24 piped in
        elif self._method == "opencv":
            import cv2
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            raise RuntimeError("[VideoGridmap] No writer initialized")

        self._count += 1

    def close(self) -> None:
        # close writers and report
        try:
            if self._method == "ffmpeg":
                self._ffmpeg_close()
            elif self._method == "opencv":
                if self._writer is not None:
                    self._writer.release()
        finally:
            self._writer = None
            self._proc = None

        try:
            sz = os.path.getsize(self.out_path)
            print(f"[VideoGridmap] Closed → {self.out_path}  frames={self._count}  size={sz/1024:.1f} KiB  fps={self.fps}  method={self._method}")
        except Exception as e:
            print("[VideoGridmap] Could not stat output:", e)

# --- Small debug printer used by the figure builder ------------------------
def dbg_stats(name: str, obj: Any) -> None:
    try:
        import torch
    except Exception:
        torch = None
    shp = None
    if isinstance(obj, np.ndarray):
        shp = obj.shape, obj.dtype
    elif torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(obj):
        shp = tuple(obj.shape), str(obj.dtype)
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], np.ndarray):
        shp = [x.shape for x in obj[:3]] + (["..."] if len(obj) > 3 else [])
    print(f"[DBG] {name}: {shp}")


def prepare_for_imshow(x: Any) -> np.ndarray:
    """
    Accepts: numpy HxWx3 uint8, torch (C,H,W)/(H,W,3) in [-0.5,0.5] or [0,1], or list of such.
    Returns: numpy HxWx3 uint8.
    """
    try:
        import torch
    except Exception:
        torch = None

    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        a, b = float(np.nanmin(arr)), float(np.nanmax(arr))
        if b <= 1.0 and a >= 0.0:
            # already [0,1] → just scale
            arr = np.clip(arr * 255.0, 0, 255)
        elif b <= 0.5 and a >= -0.5:
            # Dreamer-style [-0.5,0.5] → shift then scale
            arr = np.clip((arr + 0.5) * 255.0, 0, 255)
        else:
            # assume already [0,255]ish
            arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)

    if x is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    if isinstance(x, (list, tuple)) and len(x) > 0:
        return prepare_for_imshow(x[0])

    if isinstance(x, np.ndarray):
        arr = x
        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)
        return arr

    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
        t = x.detach().float()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0).contiguous()
        if t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        arr = t.cpu().numpy()
        return _to_uint8(arr)

    return np.zeros((64, 64, 3), dtype=np.uint8)


# --- Tiny helpers used by record_video_frames --------------------------------
def get_image_plot(ax, data, title):
    # Guard: if data is None, draw a placeholder so figure still rasterizes
    if data is None:
        ax.text(0.5, 0.5, f"{title}: None", ha="center", va="center", fontsize=10)
        ax.set_title(title)
        ax.axis("off")
        return ax
    ax.imshow(data)
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_MSE_bar(ax, mse_err: float, agent_lost: bool):
    try:
        color = "blue" if (mse_err is not None and mse_err < 0.5) else "red"
        ax.bar("MSE error", mse_err if mse_err is not None else 0.0, color=color, width=0.1)
    except Exception as e:
        ax.text(-0.3, 0.9, f"{e}", fontsize=10, color="black")
    ax.set_ylim(0, 1)
    ax.set_facecolor("#ff8970" if agent_lost else "0.8")
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("mse ob/expectation")


def plot_visited_rooms(ax, visited_rooms: List[Tuple[int, int]], env_definition: Dict[str, Any]):
    """
    Robust version: renders a simple grid heatmap if n_row/n_col exist,
    otherwise shows a small 'no grid' note.
    """
    try:
        n_row = int(env_definition.get("n_row", 0))
        n_col = int(env_definition.get("n_col", 0))
    except Exception:
        n_row = n_col = 0

    if n_row > 0 and n_col > 0:
        grid = np.ones((n_row, n_col), dtype=float)
        for i in range(n_col):
            for j in range(n_row):
                if (i, j) in visited_rooms:
                    pos = visited_rooms.index((i, j))
                    grid[j, i] = pos / max(1, len(visited_rooms))
        im = ax.imshow(grid, cmap="viridis", origin="lower")
        ax.set_title("Rooms ordered by discovery")
        ax.set_xticks(range(n_col)), ax.set_yticks(range(n_row))
        ax.set_xticklabels(range(n_col)), ax.set_yticklabels(range(n_row))
        # shrunk colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation="vertical", label="Visited Rooms")
        cbar.set_ticks([0, 0.9, 1])
        cbar.set_ticklabels(["oldest", "newest", "unknown"])
    else:
        ax.text(0.5, 0.5, "No room grid available", ha="center", va="center")
        ax.set_title("Rooms")
    ax.set_aspect("equal")
    ax.grid(False)
    return ax


def plot_memory_map(ax, memory_map_data: Dict[str, Any], *, dbg: bool = False):
    """
    Experience-map visual with guards and dedup. (Enhanced version.)
    """
    ax.cla()
    ax.set_title("Experience Map")
    ax.grid(True)
    ax.set_aspect("equal", "datalim")

    curr_id = memory_map_data.get("current_exp_id", -1)
    if curr_id < 0:
        if dbg:
            print("[VIS] No current exp → skipping map.")
        return ax

    exps_gp    = memory_map_data.get("exps_GP", [])
    exps_decay = memory_map_data.get("exps_decay", [])
    if exps_gp:
        pts = np.array(exps_gp)
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=exps_decay if len(exps_decay) == len(pts) else None,
            cmap="viridis",
            s=60, edgecolor="k", linewidth=0.5, label="Places"
        )
        # annotate w/ index when we can
        for idx, (x, y) in enumerate(pts):
            ax.text(x, y, str(idx), fontsize=7, color="white",
                    ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.35, pad=0.5))

    # Ghost links
    gl = memory_map_data.get("ghost_exps_link", [])
    for i in range(0, len(gl) // 2 * 2, 2):
        p0 = tuple(gl[i]); p1 = tuple(gl[i+1])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                ls="--", color="magenta", alpha=0.3, linewidth=2)

    # Real links (dedup)
    rl, seen = [], set()
    raw = memory_map_data.get("exps_links", [])
    for i in range(0, len(raw) // 2 * 2, 2):
        p0 = tuple(raw[i]); p1 = tuple(raw[i+1])
        if (p0, p1) not in seen:
            seen.add((p0, p1))
            rl.append((p0, p1))
    for p0, p1 in rl:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="gray", linewidth=2)

    # current exp
    cx, cy = memory_map_data.get("current_exp_GP", [0, 0])[:2]
    ax.plot(cx, cy, marker="X", color="red", markersize=10, label="Current")
    ax.text(cx, cy, str(curr_id), color="white", fontsize=9, weight="bold",
            ha="center", va="center", bbox=dict(facecolor="red", alpha=0.5, pad=1))

    if dbg:
        ax.legend(loc="upper right", fontsize=8)
    return ax


def print_positions(ax, GP, pose, step_count: int, mode_str: Optional[str] = None):
    current_gp = [float(x) for x in np.around(np.array(GP), 2)] if GP is not None else ["?", "?", "?"]
    lines = [
        f"Step: {step_count}",
        "Global Position [x,y,th]:",
        str(current_gp),
        "Local Position  [x y th]:",
        str(pose if pose is not None else ["?", "?", "?"]),
    ]
    if mode_str:
        lines += ["", mode_str]
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=10, va="top", ha="left", family="monospace")
    ax.set_axis_off()
    return ax


# --- The main frame builder you already use (ported + safer reshape) --------
def record_video_frames(data: Dict[str, Any],
                        env_definition: Dict[str, Any],
                        agent_lost: bool,
                        visited_rooms: List[Tuple[int, int]],
                        memory_map_data: Dict[str, Any],
                        step_count: int) -> np.ndarray:
    """
    Creates a full HxWx3 uint8 image for the video.
    """
    # IMPORTANT: don't reuse a fixed figure number; stale canvases can corrupt buffers.
    fig = plt.figure(figsize=(12, 6), dpi=160)
    s = fig.add_gridspec(
        4, 6,
        width_ratios=[3, 3, 3, 3, 3, 3],
        height_ratios=[3, 3, 3, 3],
        wspace=0.8, hspace=0.4
    )

    # WORLD
    ax1 = fig.add_subplot(s[:2, :2])
    dbg_stats("env_image", data.get('env_image'))
    ax1 = get_image_plot(ax1, prepare_for_imshow(data.get('env_image')), 'World')

    # GT OBS
    ax2 = fig.add_subplot(s[:1, 2:3])
    dbg_stats("ground_truth_ob", data.get('ground_truth_ob'))
    ax2 = get_image_plot(ax2, prepare_for_imshow(data.get('ground_truth_ob')), 'GT OB')

    # PREDICTED
    if data.get('image_predicted') is not None:
        pred_img = prepare_for_imshow(data['image_predicted'])
        ax3 = fig.add_subplot(s[:2, 3])
        ax3 = get_image_plot(ax3, pred_img, 'Predicted ob')

    # DECODED
    if data.get('decoded') is not None:
        dec_img = prepare_for_imshow(data['decoded'])
        ax_dec = fig.add_subplot(s[1:2, 2:3])
        ax_dec = get_image_plot(ax_dec, dec_img, 'Decoded ob')

    # VISITED ROOMS
    ax4 = fig.add_subplot(s[2:4, :2])
    ax4 = plot_visited_rooms(ax4, visited_rooms, env_definition)

    # MODE / SUBMODE (HMM)
    mode_line = ""
    if 'recommended_mode' in data:
        mode_line = f"mode={data['recommended_mode']}   submode={data.get('recommended_submode')}"
        ax_state = fig.add_subplot(s[2:4, 2:4])
        txt = []
        txt.append(f"► MODE :  {data['recommended_mode']} ({data.get('mode_confidence', 0):.2f})")
        txt.append(f"► SUB  :  {data.get('recommended_submode')} ({data.get('submode_confidence', 0) or 0:.2f})")
        if 'uncertainty' in data:
            txt.append(f"entropy={data['uncertainty']:.2f}   changepoint P={data.get('changepoint_mass', 0):.2f}")
        ax_state.text(0.0, 1.0, "\n".join(txt), fontsize=10, va="top", ha="left", family="monospace")
        ax_state.set_axis_off()

    # MSE
    ax6 = fig.add_subplot(s[:2, 4])
    ax6 = plot_MSE_bar(ax6, data.get('mse', 0.0), agent_lost)

    # COGNITIVE MAP
    ax7 = fig.add_subplot(s[2:4, 4:])
    ax7 = plot_memory_map(ax7, memory_map_data)

    # POSES
    ax8 = fig.add_subplot(s[:2, 5])
    ax8 = print_positions(ax8, data.get('GP'), data.get('pose'), step_count, mode_str=mode_line)

    # ---- Ensure Agg canvas and explicit draw before grabbing buffer ----
    fig.set_canvas(FigureCanvas(fig))
    fig.canvas.draw()

    s_bytes, (width, height) = fig.canvas.print_to_buffer()
    if not s_bytes:
        print("[RECORDER] WARNING: print_to_buffer() returned empty bytes")
    buf = np.frombuffer(s_bytes, np.uint8).reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return buf



# --- Adapter that mirrors your old record_data() but for unified runner ----
def safe_get_memory_map_data_from(planner_like: Any) -> Dict[str, Any]:
    """
    Tries several ways to fetch memory_map_data with fields expected by plot_memory_map.
    Falls back to {'current_exp_id': -1} if we can’t access the map yet.
    """
    # 1) Dedicated method present?
    if hasattr(planner_like, "get_memory_map_data"):
        try:
            return planner_like.get_memory_map_data(dbg=False)  # type: ignore[arg-type]
        except Exception:
            pass

    # 2) Try to reconstruct from planner.emap / planner.cog (best-effort)
    try:
        mg = getattr(planner_like, "cog", None).mg if hasattr(planner_like, "cog") else None
        if mg is None:
            mg = getattr(planner_like, "emap", None)  # if you exposed it
        if mg is None:
            return {'exps_GP': [], 'exps_decay': [], 'ghost_exps_GP': [],
                    'ghost_exps_link': [], 'exps_links': [], 'current_exp_id': -1}
        emap = mg.experience_map
        data = {'exps_GP': [], 'exps_decay': [], 'ghost_exps_GP': [],
                'ghost_exps_link': [], 'exps_links': []}
        data['current_exp_id'] = mg.get_current_exp_id()
        data['current_GP']     = mg.get_global_position()
        if data['current_exp_id'] < 0:
            return data
        data['current_exp_GP'] = mg.get_exp_global_position()
        for vc in mg.view_cells.cells:
            for exp in vc.exps:
                data['exps_GP'].append([exp.x_m, exp.y_m])
                data['exps_decay'].append(vc.decay)
        for ghost in emap.ghost_exps:
            data['ghost_exps_GP'].append([ghost.x_m, ghost.y_m])
            for link in ghost.links:
                data['ghost_exps_link'] += [[ghost.x_m, ghost.y_m], [link.target.x_m, link.target.y_m]]
        for exp in emap.exps:
            for link in exp.links:
                if not getattr(link.target, 'ghost_exp', False):
                    data['exps_links'] += [[link.target.x_m, link.target.y_m], [exp.x_m, exp.y_m]]
        # dedup
        clean, seen = [], set()
        pts = data['exps_links']
        for i in range(0, len(pts) // 2 * 2, 2):
            p0, p1 = tuple(pts[i]), tuple(pts[i+1])
            if (p0, p1) not in seen:
                seen.add((p0, p1))
                clean += [list(p0), list(p1)]
        data['exps_links'] = clean
        return data
    except Exception:
        return {'exps_GP': [], 'exps_decay': [], 'ghost_exps_GP': [],
                'ghost_exps_link': [], 'exps_links': [], 'current_exp_id': -1}


def append_vis_frame_unified(*,
                             planner_like: Any,
                             video_gridmap: VideoGridmap,
                             data: Dict[str, Any],
                             env_definition: Dict[str, Any],
                             visited_rooms: List[Tuple[int, int]],
                             step_count: int,
                             seconds_per_step: float | None = None) -> None:
    """
    Builds memory_map_data and pushes a frame to the writer.
    If seconds_per_step is provided, duplicates the frame to hold on screen.
    """
    memory_map_data = safe_get_memory_map_data_from(planner_like)
    frame = record_video_frames(
        data, env_definition, agent_lost=False,
        visited_rooms=visited_rooms, memory_map_data=memory_map_data,
        step_count=step_count
    )

    # hold each step for N frames (CFR ensures duration = N / fps seconds)
    if seconds_per_step and video_gridmap.fps > 0:
        repeats = max(1, int(round(video_gridmap.fps * float(seconds_per_step))))
    else:
        repeats = 1

    for _ in range(repeats):
        video_gridmap.append_data(frame)


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    try:
        a = prepare_for_imshow(a).astype(np.float32) / 255.0
        b = prepare_for_imshow(b).astype(np.float32) / 255.0
        return float(np.mean((a - b) ** 2))
    except Exception:
        return 0.0


# (Optional) tiny smoke test when run directly
if __name__ == "__main__":
    # Creates a 2-frame demo MP4 in ./dbg/ just to verify dependencies
    os.makedirs("dbg", exist_ok=True)
    writer = VideoGridmap("dbg/_smoketest.mp4", fps=2)
    dummy_env_def = {"n_row": 2, "n_col": 3, "max_steps": 2}
    mem = {'current_exp_id': -1}
    for t in range(2):
        img = (np.random.rand(240, 480, 3) * 255).astype(np.uint8)
        data = {
            "env_image": img,
            "ground_truth_ob": img,
            "decoded": img[::-1],  # just to exercise the slots
            "mse": 0.2,
            "GP": [t, t, 0.0],
            "pose": [t, t, 0.0],
            "recommended_mode": "EXPLORE",
            "recommended_submode": "ego",
            "mode_confidence": 0.9,
            "submode_confidence": 0.6,
            "uncertainty": 0.3,
            "changepoint_mass": 0.1,
        }
        frame = record_video_frames(data, dummy_env_def, agent_lost=False,
                                    visited_rooms=[(0,0), (1,0)], memory_map_data=mem, step_count=t)
        writer.append_data(frame)
    writer.close()
    print("✓ smoketest video → dbg/_smoketest.mp4")
