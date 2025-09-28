
#!/usr/bin/env python3
"""
Plan visualization utility.

Usage:
    python plan_viz.py --run <RUN_DIR>

Where RUN_DIR is something like:
    runs/plan_exports/NAVIGATE/20250925-183012/

It expects a 'plan.json' inside that directory and optionally an 'images/' folder
with files named 'node_XXXXX.png' created when nodes were reached.
"""

import os, json, glob, math, argparse, datetime
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import re 

# --- add near other helpers in viz_help.py -----------------------------------
import os
import matplotlib.pyplot as plt

def _canon_pair(u, v):
    u, v = int(u), int(v)
    return (u, v) if u <= v else (v, u)

def _node_xy_minimal(n: dict) -> tuple[float, float]:
    # mirror plan_viz semantics
    for kx, ky in (("x", "y"), ("x_m", "y_m")):
        if kx in n and ky in n:
            return float(n[kx]), float(n[ky])
    pose = n.get("pose", None)
    if pose and isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    return 0.0, 0.0

def draw_replan_compare(run_dir: str, plan: dict, prev_tokens, new_tokens,
                        *, title_left: str = "", title_right: str = "",
                        save_path: str) -> str:
    """
    Draw two side-by-side panels with ONLY the plan nodes & links from prev_tokens vs new_tokens.
    Respects inferred-vs-real edge style using plan['inferred_pairs'] (dashed when inferred).
    Colors: previous=#1f77b4, new=#ff7f0e
    """
    # Prepare data
    nodes = plan.get("nodes", {}) or {}
    id2node = {}
    if isinstance(nodes, dict):
        for k, v in nodes.items():
            id2node[int(k)] = v
    elif isinstance(nodes, list):
        for nd in nodes:
            id2node[int(nd.get("id"))] = nd

    inferred_pairs = set(_canon_pair(int(u), int(v))
                         for (u, v) in (plan.get("inferred_pairs") or []))

    def _unique_nodes(tt):
        s = set()
        for u, v in tt:
            s.add(int(u)); s.add(int(v))
        return sorted(s)

    def _draw_one(ax, tokens, color, title):
        # gather just the involved nodes
        nids = _unique_nodes(tokens)
        # edges
        for (u, v) in tokens:
            nd_u, nd_v = id2node.get(int(u), {}), id2node.get(int(v), {})
            ux, uy = _node_xy_minimal(nd_u); vx, vy = _node_xy_minimal(nd_v)
            style = '--' if _canon_pair(u, v) in inferred_pairs else '-'
            ax.plot([ux, vx], [uy, vy], linestyle=style, linewidth=3.0, alpha=0.95, color=color)
        # nodes
        xs, ys = [], []
        for nid in nids:
            x, y = _node_xy_minimal(id2node.get(int(nid), {}))
            xs.append(x); ys.append(y)
        if xs:
            ax.scatter(xs, ys, s=30, alpha=0.95, color=color, edgecolors='black', linewidths=0.3)
            for nid, x, y in zip(nids, xs, ys):
                ax.text(x, y, str(nid), fontsize=8, ha='center', va='bottom')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # Figure
    fig = plt.figure(figsize=(12, 5), dpi=200)
    axL = fig.add_subplot(1, 2, 1)
    axR = fig.add_subplot(1, 2, 2)
    _draw_one(axL, prev_tokens, "#1f77b4", title_left or "Plan (previous)")
    _draw_one(axR, new_tokens,  "#ff7f0e", title_right or "Plan (replanned)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path

def _latest_run(root: str) -> str | None:
    cands = []
    for mode in ("NAVIGATE", "TASK_SOLVING"):
        d = os.path.join(root, mode)
        if not os.path.isdir(d):
            continue
        for sub in glob.glob(os.path.join(d, "*")):
            if os.path.isdir(sub):
                cands.append(sub)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

# --- add somewhere above _ensure_visuals_for_run ---
def _list_version_pairs(plan: dict) -> list[tuple[dict, dict]]:
    """
    Return consecutive (prev, curr) version entries.
    Prefer plan['versions']; if absent or length<=1, synthesize from events 'replan'.
    """
    vers = plan.get("versions") or []
    pairs = []
    if isinstance(vers, list) and len(vers) >= 2:
        # ensure sorted by id/time just in case
        vers_sorted = sorted(vers, key=lambda x: int(x.get("id", 0)))
        for a, b in zip(vers_sorted[:-1], vers_sorted[1:]):
            pairs.append((a, b))
        return pairs

    # Fallback: synthesize from initial plan.tokens + replan events
    base = {"id": 0, "tokens": [list(map(int, t)) for t in (plan.get("tokens") or [])]}
    curr = base
    for evt in (plan.get("events") or []):
        if evt.get("type") == "replan" and evt.get("new_tokens"):
            nxt = {"id": curr.get("id", 0) + 1, "tokens": [list(map(int, t)) for t in evt["new_tokens"]]}
            pairs.append((curr, nxt))
            curr = nxt
    return pairs


def _parse_args():
    ap = argparse.ArgumentParser(description="Visualize exported hierarchical plan (graph + timeline)")
    ap.add_argument("--run", type=str, default=None, help="Path to a single export run directory (contains plan.json)")
    ap.add_argument("--root", type=str, default="runs/plan_exports", help="Root folder where runs are stored")
    ap.add_argument("--save-prefix", type=str, default=None, help="Optional prefix for output files")
    ap.add_argument("--focus-nodes", type=str, default=None,
                help="Comma-separated node ids to visualize deeply (1 or 2 recommended)")
    ap.add_argument("--focus-k", type=int, default=2,
                    help="If --focus-nodes not set, take the last K path nodes (default 2)")
    ap.add_argument("--steps-per-edge", type=int, default=12,
                    help="Max step thumbnails per edge strip (sampled evenly)")
    ap.add_argument("--force-hierarchy", action="store_true",
                    help="Rebuild ONLY hierarchy.png (others keep their default unless --force)")
    ap.add_argument("--nodes",
    type=str,
    default=None,
    help="Comma-separated node ids to render individual node panels (e.g. '60,45')") 
    ap.add_argument("--force", action="store_true",
                help="Rebuild visuals even if target files already exist")
    ap.add_argument("--recurse", action="store_true",
                    help="If --run points to a parent (e.g., .../NAVIGATE), walk its immediate subfolders")               
    return ap.parse_args()

def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (int(u), int(v))

def _canon_pair(u: int, v: int) -> tuple[int, int]:
    u, v = int(u), int(v)
    return (u, v) if u <= v else (v, u)
def _is_run_dir(p: str) -> bool:
    """A run dir contains a plan.json (images optional)."""
    return os.path.isdir(p) and os.path.exists(os.path.join(p, "plan.json"))

def _child_run_dirs(parent: str) -> list[str]:
    """Immediate subfolders of `parent` that look like runs (contain plan.json)."""
    if not os.path.isdir(parent):
        return []
    kids = []
    for sub in sorted(glob.glob(os.path.join(parent, "*"))):
        if _is_run_dir(sub):
            kids.append(sub)
    return kids

def _choose_focus_nodes(plan: dict, k: int = 2) -> list[int]:
    tokens = [tuple(map(int, t)) for t in plan.get("tokens", [])]
    path_nodes = [v for (u, v) in tokens]
    if path_nodes:
        return path_nodes[-max(1, min(k, 2)):]
    start = plan.get("start_exp_id")
    return [int(start)] if start is not None else []

def _ensure_visuals_for_run(run_dir: str, *, force: bool = False, force_hierarchy: bool = False,
                             focus_k: int = 2, steps_per_edge: int = 12,
                             save_prefix: str | None = None,
                             explicit_nodes: list[int] | None = None) -> list[str]:
    """
    Generate any missing visuals for a single run folder.
    Returns list of files created (skips those already present unless --force).
    """
    created: list[str] = []
    plan_path = os.path.join(run_dir, "plan.json")
    if not os.path.exists(plan_path):
        print(f"[skip] {run_dir} (no plan.json)")
        return created

    with open(plan_path, "r") as f:
        plan = json.load(f)

    # 1) Graph --------------------------------------------------------------
    graph_out = os.path.join(run_dir, (save_prefix or "") + "plan_graph.png")
    if force or not os.path.exists(graph_out):
        try:
            graph_png = draw_graph(run_dir, plan, save_prefix=save_prefix or "")
            created.append(graph_png)
        except Exception as e:
            print(f"[warn] graph failed in {run_dir}: {e}")
    else:
        print(f"[keep] {graph_out}")

    # 2) Timeline -----------------------------------------------------------
    imgs = _collect_images(run_dir)
    timeline_out = os.path.join(run_dir, (save_prefix or "") + "timeline.png")
    if force or not os.path.exists(timeline_out):
        try:
            tl = draw_timeline(run_dir, plan, imgs, save_prefix=save_prefix or "")
            if tl:
                created.append(tl)
            else:
                print(f"[info] no timeline data in {run_dir}")
        except Exception as e:
            print(f"[warn] timeline failed in {run_dir}: {e}")
    else:
        print(f"[keep] {timeline_out}")

    # 3) Node panels (2 max by your existing flow) -------------------------
    if explicit_nodes and len(explicit_nodes) > 0:
        focus_nodes = [int(n) for n in explicit_nodes][:2]
    else:
        focus_nodes = _choose_focus_nodes(plan, k=focus_k)

    for nid in focus_nodes[:2]:
        panel_out = os.path.join(run_dir, (save_prefix or "") + f"node_{int(nid):05d}_panel.png")
        if force or not os.path.exists(panel_out):
            try:
                pth = draw_node_panel(run_dir, plan, nid, save_prefix=save_prefix or "")
                if pth:
                    created.append(pth)
            except Exception as e:
                print(f"[warn] node panel {nid} failed in {run_dir}: {e}")
        else:
            print(f"[keep] {panel_out}")

    # 4) 4-level hierarchy panel -------------------------------------------
    hier_out = os.path.join(run_dir, (save_prefix or "") + "hierarchy.png")

    if (force or force_hierarchy) or not os.path.exists(hier_out):
        try:
            hp = render_hierarchy(
                run_dir, plan,
                focus_nodes=focus_nodes if focus_nodes else _choose_focus_nodes(plan, k=1),
                steps_per_edge=steps_per_edge,
                save_prefix=save_prefix or ""
            )
            if hp:
                created.append(hp)
        except Exception as e:
            print(f"[warn] hierarchy failed in {run_dir}: {e}")
    else:
        print(f"[keep] {hier_out}")
    # 5) Replan side-by-side panels (for each version jump) ------------------
    try:
        created += _ensure_replan_visuals_for_run(
            run_dir, plan, force=force, save_prefix=save_prefix or ""
        )
    except Exception as e:
        print(f"[warn] replan visuals failed in {run_dir}: {e}")
    

    return created

# --- add just below _ensure_visuals_for_run or alongside it ---
def _ensure_replan_visuals_for_run(run_dir: str, plan: dict, *, force: bool, save_prefix: str | None) -> list[str]:
    created = []
    pairs = _list_version_pairs(plan)
    if not pairs:
        return created

    # For titles / counts we optionally use the diff if available
    for prev, curr in pairs:
        pid = int(prev.get("id", 0)); cid = int(curr.get("id", pid+1))
        prev_tok = [tuple(map(int, t)) for t in (prev.get("tokens") or [])]
        curr_tok = [tuple(map(int, t)) for t in (curr.get("tokens") or [])]

        # filename pattern (keeps your prefix)
        out = os.path.join(run_dir, (save_prefix or "") + f"replan_v{pid:03d}_to_v{cid:03d}.png")
        if (not force) and os.path.exists(out):
            print(f"[keep] {out}")
            continue

        # Build informative titles (use diff if present, else compute)
        def _diff(a, b):
            sa, sb = set(map(tuple, a)), set(map(tuple, b))
            return sorted(list(sb - sa)), sorted(list(sa - sb))  # added, removed
        add_list = (curr.get("diff") or {}).get("added")
        rem_list = (curr.get("diff") or {}).get("removed")
        if add_list is None or rem_list is None:
            add_list, rem_list = _diff(prev_tok, curr_tok)
        tL = f"v{pid}  (edges={len(prev_tok)})"
        tR = f"v{cid}  (edges={len(curr_tok)})  +{len(add_list)}  -{len(rem_list)}"

        try:
            p = draw_replan_compare(run_dir, plan, prev_tok, curr_tok,
                                    title_left=tL, title_right=tR, save_path=out)
            if p:
                created.append(p)
        except Exception as e:
            print(f"[warn] replan compare v{pid}->{cid} failed in {run_dir}: {e}")
    return created


def _load_plan(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "plan.json"), "r") as f:
        return json.load(f)
def _collect_nodes_media(run_dir: str, plan: dict):
    """
    Return dict[int, dict]:
      node_id -> {'snapshot': absolute_path_or_None, 'enhanced': {label->absolute_path}}
    Priority & selection:
      - Prefer *_abs if it EXISTS on disk; otherwise fallback to relative (joined with run_dir) if that EXISTS.
      - If neither exists, keep the preferred one (abs if present, else rel) and log that it is missing.
    Sources (in order):
      1) events[].media.snapshot_abs / enhanced_abs  (new, authoritative)
      2) nodes_media[*].snapshot_abs / enhanced_abs  (older mapping)
      3) nodes_media[*].snapshot / enhanced          (relative)
      4) nodes[*].snapshot_abs / enhanced_preds_abs  (older absolute list fallback)
    """
    def _prefer_existing_path(rel_path: str | None, abs_path: str | None) -> tuple[str | None, str]:
        # returns (chosen_path, tag) where tag is one of: 'abs-ok','rel-ok','abs-missing','rel-missing','none'
        if abs_path and os.path.isabs(abs_path) and os.path.exists(abs_path):
            return abs_path, "abs-ok"
        rel_full = os.path.join(run_dir, rel_path) if rel_path else None
        if rel_full and os.path.exists(rel_full):
            return rel_full, "rel-ok"
        if abs_path and os.path.isabs(abs_path):
            return abs_path, "abs-missing"
        if rel_full:
            return rel_full, "rel-missing"
        return None, "none"

    out: dict[int, dict] = {}

    # ---- (B) New events: node_media_attached (authoritative) ----
    ev_count = 0
    for evt in (plan.get("events") or []):
        if evt.get("type") != "node_media_attached":
            continue
        ev_count += 1
        try:
            nid = int(evt.get("node_id"))
        except Exception:
            continue
        media = evt.get("media") or {}
        snap, tag = _prefer_existing_path(media.get("snapshot"), media.get("snapshot_abs"))
        enh_abs  = media.get("enhanced_abs") or {}
        enh_rel  = media.get("enhanced") or {}
        enh_map: dict[str, str] = {}
        for lab in ("L", "R", "LL", "RR"):
            chosen, ctag = _prefer_existing_path(enh_rel.get(lab), enh_abs.get(lab))
            if chosen:
                enh_map[lab] = chosen
            print(f"run={run_dir} nid={nid} {lab}: {chosen} ({ctag}) exists={os.path.exists(chosen) if chosen else None}")
        prev = out.get(nid, {"snapshot": None, "enhanced": {}})
        if snap:
            prev["snapshot"] = snap
        prev["enhanced"].update(enh_map)
        out[nid] = prev
        print(f"run={run_dir} nid={nid} snapshot: {snap} ({tag}) exists={os.path.exists(snap) if snap else None}")

    print(f"run={run_dir} node_media_attached events found: {ev_count}")

    # ---- (A) Older plan['nodes_media'] mapping (used if no event provided) ----
    nm = plan.get("nodes_media", {}) or {}
    for nid_str, bundle in nm.items():
        try:
            nid = int(nid_str)
        except Exception:
            continue
        snap, tag = _prefer_existing_path(bundle.get("snapshot"), bundle.get("snapshot_abs"))
        enh_abs  = bundle.get("enhanced_abs") or {}
        enh_rel  = bundle.get("enhanced") or {}
        enh_map: dict[str, str] = {}
        for lab in ("L", "R", "LL", "RR"):
            chosen, ctag = _prefer_existing_path(enh_rel.get(lab), enh_abs.get(lab))
            if chosen:
                enh_map[lab] = chosen
            print(f"[fallback nodes_media] run={run_dir} nid={nid} {lab}: {chosen} ({ctag}) exists={os.path.exists(chosen) if chosen else None}")
        # do not overwrite event-sourced entries; only fill missing
        entry = out.setdefault(nid, {"snapshot": None, "enhanced": {}})
        if (entry["snapshot"] is None) and snap:
            entry["snapshot"] = snap
            print(f"[fallback nodes_media] run={run_dir} nid={nid} snapshot: {snap} ({tag}) exists={os.path.exists(snap) if snap else None}")
        for k, v in enh_map.items():
            if k not in entry["enhanced"]:
                entry["enhanced"][k] = v

    # ---- (C) nodes[*] absolute list fallback ----
    nodes = plan.get("nodes", {})
    items = nodes.items() if isinstance(nodes, dict) else [(str(n.get("id")), n) for n in (nodes or [])]
    for k, nd in items:
        try:
            nid = int(k)
        except Exception:
            continue
        snap_abs = nd.get("snapshot_abs")
        enh_abs_list = nd.get("enhanced_preds_abs") or []
        if not snap_abs and not enh_abs_list:
            continue
        entry = out.setdefault(nid, {"snapshot": None, "enhanced": {}})
        # only fill if still missing
        if (entry["snapshot"] is None) and snap_abs:
            tag = "abs-ok" if os.path.exists(snap_abs) else "abs-missing"
            entry["snapshot"] = snap_abs
            print(f"[fallback nodes[]] run={run_dir} nid={nid} snapshot_abs: {snap_abs} ({tag}) exists={os.path.exists(snap_abs)}")
        labs = ["L", "R", "LL", "RR"]
        for i, p in enumerate(enh_abs_list):
            if i >= len(labs) or not p:
                continue
            lab = labs[i]
            if lab not in entry["enhanced"]:
                tag = "abs-ok" if os.path.exists(p) else "abs-missing"
                entry["enhanced"][lab] = p
                print(f"[fallback nodes[]] run={run_dir} nid={nid} {lab}: {p} ({tag}) exists={os.path.exists(p)}")

    print(f"run={run_dir} _collect_nodes_media -> {len(out)} nodes with media")
    return out



def _collect_step_obs(run_dir: str, plan: dict):
    """
    Group per-step images by edge key 'edge_u_v' (or 'phantom'), preserving time order.
    Returns dict[str, list[str]] of absolute image paths.
    """
    out = {}
    for evt in plan.get("events", []):
        if evt.get("type") != "step_obs":
            continue
        img_rel = evt.get("image")
        if not img_rel:
            continue
        edge = evt.get("edge")
        key = "phantom" if not edge else f"edge_{int(edge[0])}_{int(edge[1])}"
        out.setdefault(key, []).append(os.path.join(run_dir, img_rel))
    return out

def _collect_images(run_dir: str) -> dict[int, str]:
    imgs_dir = os.path.join(run_dir, "images")
    if not os.path.isdir(imgs_dir):
        return {}
    mapping = {}
    for p in glob.glob(os.path.join(imgs_dir, "node_*.png")):
        # Expect "node_00012.png"
        stem = os.path.basename(p)
        try:
            nid = int(stem.split("_")[1].split(".")[0])
        except Exception:
            # Accept "node_12.png" too
            try:
                nid = int(stem.split("_")[1].split(".")[0])
            except Exception:
                continue
        mapping[nid] = p
    return mapping
def render_hierarchy(run_dir: str, plan: dict, focus_nodes: List[int],
                     steps_per_edge: int = 12, save_prefix: str | None = None) -> str:
    """
    Compose a 4-level panel:
      1) Graph (top)
      2) Node creation snapshot(s) (middle-1)
      3) Enhanced perception 4-up per node (middle-2)
      4) Per-step strips for edges touching the node(s) (bottom)
    """
    # 1) Reuse (and generate if needed) the graph figure
    graph_png = draw_graph(run_dir, plan, save_prefix=save_prefix or "")
    graph_img = Image.open(graph_png).convert("RGB")

    # 2) Load node media + step groups
    nm = _collect_nodes_media(run_dir, plan)             # node -> snapshot/enhanced
    print(f"hierarchy: focus_nodes={focus_nodes}  media_keys={sorted(nm.keys())[:10]} (total={len(nm)})")

    # Reconcile focus nodes with actual media: if none of the requested nodes have media,
    # fallback to the most recent nodes that do.
    k_media = min(2, max(1, len(focus_nodes) if focus_nodes else 2))
    focus_in_media = [int(n) for n in (focus_nodes or []) if int(n) in nm]
    if not focus_in_media:
        # Prefer chronological order from events (node_media_attached), else the keys order.
        ev_nodes = [int(e.get("node_id")) for e in (plan.get("events") or [])
                    if e.get("type") == "node_media_attached" and isinstance(e.get("node_id"), (int, str))]
        # keep only those that actually have media gathered
        ev_nodes = [n for n in ev_nodes if n in nm]
        fallback = (ev_nodes[-k_media:] if ev_nodes else sorted(nm.keys())[-k_media:])
       
        print(f"[media] hierarchy: focus_nodes={focus_nodes} had no media; using fallback={fallback}")
        focus_nodes = fallback
    else:
        # Drop any requested nodes that don't have media so we don't render empty rows
        dropped = [n for n in (focus_nodes or []) if int(n) not in nm]
        if dropped :
            print(f"[media] hierarchy: dropping nodes without media: {dropped}")
        focus_nodes = focus_in_media

    # Ensure we have at least one focus node if media exists
    if not focus_nodes and nm:
        focus_nodes = [sorted(nm.keys())[-1]]
    print(f"hierarchy: (post-reconcile) focus_nodes={focus_nodes}  media_keys={sorted(nm.keys())[:10]} (total={len(nm)})")

    step_groups = _collect_step_obs(run_dir, plan)       # "edge_u_v" -> [imgs...]

    # utility: build a strip from a list of images
    def strip(img_paths, height=96, max_n=steps_per_edge, label=None):
        if not img_paths:
            # placeholder
            im = Image.new("RGB", (height, height), (240, 240, 240))
            d = ImageDraw.Draw(im); d.text((8,8), "no steps", fill=(0,0,0))
            return im
        # sample evenly
        idxs = np.linspace(0, max(0, len(img_paths)-1), num=min(max_n, len(img_paths)))
        idxs = [int(round(i)) for i in idxs]
        thumbs = []
        for i in idxs:
            try:
                im = Image.open(img_paths[i]).convert("RGB")
            except Exception:
                im = Image.new("RGB", (height, height), (230, 230, 230))
            w, h = im.size
            im = im.resize((int(height * w/h), height))
            thumbs.append(im)
        W = sum(t.width for t in thumbs) + 8*(len(thumbs)+1)
        H = height + 24
        canvas = Image.new("RGB", (W, H), (255,255,255))
        x = 8
        for t in thumbs:
            canvas.paste(t, (x, 8))
            x += t.width + 8
        if label:
            d = ImageDraw.Draw(canvas); d.text((8, height+4), label, fill=(0,0,0))
        return canvas

    # 3) Node rows: snapshot + 4-up enhanced
    def node_row(nid: int):
        # snapshot
        snap = nm.get(nid, {}).get("snapshot")
        print(f"node_row nid={nid} snapshot={snap} exists={os.path.exists(snap) if snap else None}")
        if snap and os.path.exists(snap):
            snap_im = Image.open(snap).convert("RGB")
        else:
            snap_im = Image.new("RGB", (160, 120), (235, 235, 235))
            ImageDraw.Draw(snap_im).text((8,8), f"node {nid}\nno snapshot", fill=(0,0,0))
        # standardize snapshot size
        snap_im = snap_im.resize((200, 150))

        # enhanced 4-up
        enh = nm.get(nid, {}).get("enhanced", {})
        labels = ["L","R","LL","RR"]
        tiles = []
        for lab in labels:
            p = enh.get(lab)
            print(f"node_row nid={nid} {lab} path={p} exists={os.path.exists(p) if p else None}")
            if p and os.path.exists(p):
                im = Image.open(p).convert("RGB")
            else:
                im = Image.new("RGB", (160, 120), (240, 240, 240))
                ImageDraw.Draw(im).text((8,8), lab if p else f"{lab}\n(n/a)", fill=(0,0,0))
            im = im.resize((160, 120))
            tiles.append((lab, im))
        # pack enhanced 4-up horizontally
        enh_W = sum(im.width for _, im in tiles) + 10*(len(tiles)+1)
        enh_H = max(im.height for _, im in tiles) + 24
        enh_canvas = Image.new("RGB", (enh_W, enh_H), (255,255,255))
        x = 10
        d = ImageDraw.Draw(enh_canvas)
        for lab, im in tiles:
            enh_canvas.paste(im, (x, 10))
            d.text((x, im.height + 12), lab, fill=(0,0,0))
            x += im.width + 10

        # side-by-side: snapshot | 4-up
        row_W = snap_im.width + 20 + enh_canvas.width
        row_H = max(snap_im.height, enh_canvas.height)
        row = Image.new("RGB", (row_W, row_H), (255,255,255))
        row.paste(snap_im, (0, (row_H - snap_im.height)//2))
        row.paste(enh_canvas, (snap_im.width + 20, (row_H - enh_canvas.height)//2))
        # title
        title_h = 28
        titled = Image.new("RGB", (row_W, row_H + title_h), (255,255,255))
        titled.paste(row, (0, title_h))
        ImageDraw.Draw(titled).text((4, 6), f"Node {nid} — creation & enhanced perception", fill=(0,0,0))
        return titled

    # 4) Bottom: step strips for edges incident to focus nodes
    # Build a tiny helper to find path edges that touch nid
    tokens = [tuple(map(int, t)) for t in plan.get("tokens", [])]
    incident = []
    for nid in focus_nodes:
        for (u, v) in tokens:
            if nid in (u, v):
                key = f"edge_{u}_{v}"
                ims = step_groups.get(key, [])
                if ims:
                    incident.append((f"{u}->{v}", ims))
    # make one strip per incident edge
    strips = [strip(ims, height=96, max_n=steps_per_edge, label=lbl) for (lbl, ims) in incident]
    if not strips:
        strips = [strip([], height=96, label="no incident edge steps")]

    # 5) Compose the 4 levels
    pad = 24
    col_W = max(graph_img.width, *(r.width for r in (node_row(focus_nodes[0]),)))
    # Node rows for 1–2 nodes
    node_rows = [node_row(nid) for nid in focus_nodes]
    nodes_W = max(r.width for r in node_rows)
    nodes_H = sum(r.height for r in node_rows) + pad*(len(node_rows)-1)

    strips_W = max(s.width for s in strips)
    strips_H = sum(s.height for s in strips) + 8*(len(strips)-1)

    W = max(graph_img.width, nodes_W, strips_W)
    H = graph_img.height + pad + nodes_H + pad + strips_H

    canvas = Image.new("RGB", (W, H), (255,255,255))
    y = 0
    # Level 1: graph
    canvas.paste(graph_img, ((W - graph_img.width)//2, y)); y += graph_img.height + pad
    # Level 2–3: node rows (snapshot + enhanced)
    for nr in node_rows:
        canvas.paste(nr, ((W - nr.width)//2, y))
        y += nr.height + pad
    # Level 4: step strips
    for s in strips:
        canvas.paste(s, ((W - s.width)//2, y))
        y += s.height + 8

    # Optional: arrows between levels (kept subtle; PIL arrows via triangles)
    draw = ImageDraw.Draw(canvas)
    # (draw simple down-arrows between levels)
    def arrow_down(cx, y0, y1):
        draw.line((cx, y0, cx, y1-8), fill=(0,0,0), width=2)
        draw.polygon([(cx-6, y1-8), (cx+6, y1-8), (cx, y1)], fill=(0,0,0))
    cx = W//2
    arrow_down(cx, graph_img.height+4, graph_img.height+pad-4)
    # between node rows and strips
    arrow_down(cx, H - strips_H - pad + 4, H - strips_H - 4)

    out = os.path.join(run_dir, (save_prefix or "") + "hierarchy.png")
    canvas.save(out)
    return out

def _extract_graph(plan: dict):
    nodes = plan.get("nodes", {})
    # nodes might be dict[str->dict] or list
    id2node = {}
    if isinstance(nodes, dict):
        for k, v in nodes.items():
            id2node[int(k)] = v
    elif isinstance(nodes, list):
        for nd in nodes:
            id2node[int(nd["id"])] = nd
    else:
        id2node = {}

    adj = plan.get("adjacency", {})
    adj_norm = {int(u): [int(v) for v in vs] for u, vs in adj.items()}

    tokens = [tuple(map(int, t)) for t in plan.get("tokens", [])]
    tok_edges = set(tokens)

    inferred_pairs = plan.get("inferred_pairs", [])
    inferred_set = { _canon_pair(int(u), int(v)) for (u, v) in inferred_pairs }

    return id2node, adj_norm, tokens, tok_edges, inferred_set

def _resolve(run_dir: str, rel_or_abs: str | None) -> str | None:
    if not rel_or_abs: return None
    p = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(run_dir, rel_or_abs)
    return p if os.path.exists(p) else None

def draw_node_panel(run_dir: str, plan: dict, node_id: int, save_prefix: str | None = None) -> str | None:
    media = plan.get("nodes_media", {}).get(str(int(node_id)), {})
    snap_rel = media.get("snapshot")
    enh_map  = media.get("enhanced", {}) or {}

    snap = _resolve(run_dir, snap_rel)
    L  = _resolve(run_dir, enh_map.get("L"))
    R  = _resolve(run_dir, enh_map.get("R"))
    LL = _resolve(run_dir, enh_map.get("LL"))
    RR = _resolve(run_dir, enh_map.get("RR"))

    # collect bottom-row traversal thumbnails from step_obs along edges touching this node
    step_paths: list[str] = []
    for evt in plan.get("events", []):
        if evt.get("type") == "step_obs":
            edge = evt.get("edge")
            img  = evt.get("image")
            if img and edge and (int(node_id) in edge):
                p = _resolve(run_dir, img)
                if p: step_paths.append(p)
    step_paths = step_paths[:12]  # avoid cramming

    # Layout: title + 3 rows:
    # Row 1: snapshot (centered)
    # Row 2: enhanced (L, R, LL, RR)
    # Row 3: traversal steps (up to 12)
    W, H = 1200, 900
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), f"Node {node_id} — snapshot, enhanced perception, and traversals", fill=(0,0,0))

    def load_or_blank(p, w, h, label=None):
        from PIL import Image, ImageDraw
        if p and os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((w, h))
        else:
            im = Image.new("RGB", (w, h), (230, 230, 230))
            if label:
                ImageDraw.Draw(im).text((8, 8), label, fill=(0,0,0))
        return im

    # Row 1: snapshot
    row1_y = 40
    snap_w, snap_h = 320, 240
    snap_im = load_or_blank(snap, snap_w, snap_h, "no snapshot")
    canvas.paste(snap_im, ((W - snap_w)//2, row1_y))

    # Row 2: enhanced predictions
    row2_y = row1_y + snap_h + 20
    cell_w, cell_h = 240, 180
    col_x = [60, 60+cell_w+20, 60+2*(cell_w+20), 60+3*(cell_w+20)]
    for x, (lbl, p) in zip(col_x, [("L",L),("R",R),("LL",LL),("RR",RR)]):
        im = load_or_blank(p, cell_w, cell_h, f"{lbl} (none)")
        canvas.paste(im, (x, row2_y))
        draw.text((x, row2_y + cell_h + 4), lbl, fill=(0,0,0))

    # Row 3: traversals (per-step FoV touching this node)
    row3_y = row2_y + cell_h + 40
    t_w, t_h = 96, 96
    x = 16
    for p in step_paths:
        im = load_or_blank(p, t_w, t_h)
        canvas.paste(im, (x, row3_y))
        x += t_w + 10
        if x + t_w > W - 16:
            break

    out = os.path.join(run_dir, (save_prefix or "") + f"node_{int(node_id):05d}_panel.png")
    canvas.save(out)
    return out


def _node_xy(n: dict) -> tuple[float, float]:
    # prefer "x" / "y" else "x_m" / "y_m" else pose
    for kx, ky in (("x", "y"), ("x_m", "y_m")):
        if kx in n and ky in n:
            return float(n[kx]), float(n[ky])
    pose = n.get("pose", None)
    if pose and isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    # Fallback: (0,0)
    return 0.0, 0.0

def draw_graph(run_dir: str, plan: dict, save_prefix: str | None = None) -> str:
    id2node, adj, tokens, tok_edges, inferred_set = _extract_graph(plan)
    if not id2node:
        raise RuntimeError("No nodes in plan.json")

    # Normalize layout
    xs, ys = [], []
    for nid, nd in id2node.items():
        x, y = _node_xy(nd)
        xs.append(x); ys.append(y)
    if not xs:
        xs = [0.0]; ys = [0.0]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0

    # Prepare plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f"Hierarchical plan graph – {plan.get('mode','?')} – start={plan.get('start_exp_id')} goal={plan.get('goal_node_id')}")
    ax.set_aspect('equal', adjustable='box')

    # Draw all edges (light)
    for u, vs in adj.items():
        ux, uy = _node_xy(id2node.get(u, {}))
        for v in vs:
            vx, vy = _node_xy(id2node.get(v, {}))
            # Style: inferred vs explicit (global)
            style = '--' if _canon_pair(u, v) in inferred_set else '-'
            lw = 0.8
            ax.plot([ux, vx], [uy, vy], linestyle=style, linewidth=lw, alpha=0.4)

    # Draw path edges (thicker)
    for (u, v) in tokens:
        ux, uy = _node_xy(id2node.get(u, {}))
        vx, vy = _node_xy(id2node.get(v, {}))
        style = '--' if _canon_pair(u, v) in inferred_set else '-'
        ax.plot([ux, vx], [uy, vy], linestyle=style, linewidth=3.0, alpha=0.9)

    # Draw nodes
    for nid, nd in id2node.items():
        x, y = _node_xy(nd)
        ax.scatter([x], [y], s=30, alpha=0.9)
        ax.text(x, y, f"{nid}", fontsize=8, ha='center', va='bottom')

    # Number the order along tokens (1..K) at midpoints
    for i, (u, v) in enumerate(tokens, start=1):
        ux, uy = _node_xy(id2node.get(u, {}))
        vx, vy = _node_xy(id2node.get(v, {}))
        mx, my = (ux + vx) / 2.0, (uy + vy) / 2.0
        ax.text(mx, my, str(i), fontsize=8, ha='center', va='center')

    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Save
    out = os.path.join(run_dir, (save_prefix or "") + "plan_graph.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

def draw_timeline(run_dir: str, plan: dict, img_map: dict[int, str], save_prefix: str | None = None) -> str | None:
    tokens = [tuple(map(int, t)) for t in plan.get("tokens", [])]
    # Order of nodes visited (excluding start self-edge), we mark arrivals to 'v'
    node_order = [v for (u, v) in tokens]
    if not node_order:
        return None

    thumb_w, thumb_h = 96, 96
    margin = 12
    # create canvas width = thumbs + margins
    N = len(node_order)
    W = margin + N * (thumb_w + margin)
    H = thumb_h + 2*margin + 30  # title band
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = f"Timeline of FoV on node arrivals ({plan.get('mode','?')})"
    try:
        draw.text((margin, 6), title, fill=(0,0,0))
    except Exception:
        pass

    for i, nid in enumerate(node_order):
        x0 = margin + i * (thumb_w + margin)
        y0 = 30 + margin
        path = img_map.get(nid, None)
        if path and os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = img.resize((thumb_w, thumb_h))
        else:
            # placeholder
            img = Image.new("RGB", (thumb_w, thumb_h), (230, 230, 230))
            d2 = ImageDraw.Draw(img)
            d2.text((8, 8), f"id {nid}", fill=(0,0,0))
        canvas.paste(img, (x0, y0))
        draw.text((x0, y0 + thumb_h + 2), f"{i+1}:{nid}", fill=(0,0,0))

    out = os.path.join(run_dir, (save_prefix or "") + "timeline.png")
    canvas.save(out)
    return out

def main():
    args = _parse_args()

    # Prefer explicit --run if given; if it is a run dir use it directly.
    # If it is a parent (e.g., .../NAVIGATE), walk its children when --recurse
    # OR when plan.json is missing in --run but present in children.
    run_dir = args.run
    root = args.root

    # If nothing passed, fall back to latest single run (your old behavior)
    if not run_dir:
        run_dir = _latest_run(root)
        if not run_dir:
            raise SystemExit("No run directory found. Pass --run or ensure runs/plan_exports has content.")

    # Case A: --run points to an actual run
    if _is_run_dir(run_dir) and not args.recurse:
        # preserve original single-run behavior (but skip existing outputs)
        with open(os.path.join(run_dir, "plan.json"), "r") as f:
            plan = json.load(f)

        imgs = _collect_images(run_dir)
        # Graph
        graph_out = os.path.join(run_dir, (args.save_prefix or "") + "plan_graph.png")
        if args.force or not os.path.exists(graph_out):
            graph_png = draw_graph(run_dir, plan, save_prefix=args.save_prefix or "")
            print("Saved:", graph_png)
        else:
            print("[keep]", graph_out)

        # Timeline
        timeline_out = os.path.join(run_dir, (args.save_prefix or "") + "timeline.png")
        if args.force or not os.path.exists(timeline_out):
            timeline_png = draw_timeline(run_dir, plan, imgs, save_prefix=args.save_prefix or "")
            if timeline_png:
                print("Saved:", timeline_png)
        else:
            print("[keep]", timeline_out)

        # Node panels (same logic you had: explicit --nodes or fallback -> --focus-nodes)
        nodes_str = getattr(args, "nodes", None)
        node_ids: list[int] = []
        if nodes_str:
            node_ids = [int(x) for x in re.split(r"[,\s]+", nodes_str.strip()) if x]
        if not node_ids and args.focus_nodes:
            node_ids = [int(x) for x in args.focus_nodes.split(",") if x.strip()]
        if not node_ids:
            node_ids = _choose_focus_nodes(plan, k=args.focus_k)

        for nid in node_ids[:2]:
            panel_out = os.path.join(run_dir, (args.save_prefix or "") + f"node_{int(nid):05d}_panel.png")
            if args.force or not os.path.exists(panel_out):
                panel = draw_node_panel(run_dir, plan, nid, save_prefix=args.save_prefix or "")
                if panel:
                    print("Saved:", panel)
            else:
                print("[keep]", panel_out)

        # Hierarchy
        hier_out = os.path.join(run_dir, (args.save_prefix or "") + "hierarchy.png")
        focus_nodes = node_ids[:2] if node_ids else _choose_focus_nodes(plan, k=args.focus_k)
        if (args.force or args.force_hierarchy) or not os.path.exists(hier_out):
            hier_png = render_hierarchy(
                run_dir, plan,
                focus_nodes=focus_nodes,
                steps_per_edge=args.steps_per_edge,
                save_prefix=args.save_prefix or ""
            )
            print("Saved:", hier_png)
        else:
            print("[keep]", hier_out)
        return

    # Case B: --run points to a parent (e.g., .../NAVIGATE or .../TASK_SOLVING)
    # Walk immediate subfolders that contain plan.json (robust to empties).
    parents = [run_dir]
    all_runs = []
    for parent in parents:
        kids = _child_run_dirs(parent)
        all_runs.extend(kids)

    if not all_runs:
        raise SystemExit(f"No run folders with plan.json under: {run_dir}")

    print(f"[batch] Found {len(all_runs)} runs under {run_dir}")
    for i, rd in enumerate(all_runs, 1):
        print(f"\n[{i}/{len(all_runs)}] {rd}")
        try:
            made = _ensure_visuals_for_run(
                rd,
                force=args.force,
                force_hierarchy=args.force_hierarchy,
                focus_k=args.focus_k,
                steps_per_edge=args.steps_per_edge,
                save_prefix=args.save_prefix or "",
                explicit_nodes=[int(x) for x in args.nodes.split(",")] if args.nodes else None
            )
            if made:
                for m in made:
                    print("  Saved:", m)
            else:
                print("  (nothing to do)")
        except Exception as e:
            print(f"  [warn] run failed: {e}")


if __name__ == "__main__":
    main()
