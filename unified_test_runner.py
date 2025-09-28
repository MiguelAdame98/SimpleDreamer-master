import importlib, sys, pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from importlib import util
# or
from importlib.util import spec_from_file_location, module_from_spec
import os
import logging
logging.basicConfig(level=logging.INFO)
# point "dreamer" to dreamer_mg.dreamer
pkg_path = pathlib.Path(__file__).parent / "dreamer"
spec = importlib.util.spec_from_file_location("dreamer", pkg_path / "__init__.py")
dreamer_pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dreamer_pkg)
sys.modules["dreamer"] = dreamer_pkg
import yaml, torch, pathlib, heapq, random, numpy as np
from collections import namedtuple
from types import SimpleNamespace  
from dreamer.modules.encoder import Encoder
from dreamer.modules.model   import RSSM
from dreamer.modules.decoder import Decoder 
from dreamer.modules.model import RewardModel
from attrdict import AttrDict 
import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym.wrappers import ResizeObservation
from dreamer.envs.wrappers import ChannelFirstEnv
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import textwrap, pprint, itertools
import math
from PIL import Image, ImageDraw, ImageFont   # Pillow
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils
import torchvision.transforms.functional as TF 
from torchvision.utils import save_image
from pathlib import Path
from itertools import islice
from torchvision.utils import save_image
 # --- Cognitive graph harness ---------------------------------------------------
from dataclasses import dataclass
from viz_help import (
    VideoGridmap,
    record_video_frames,
    append_vis_frame_unified,
    safe_get_memory_map_data_from,
    prepare_for_imshow,
    compute_mse,
)
from typing import Optional, List, Tuple, Any, Dict
import numpy as np
import sys as _sys


_REPO_ROOT = str(Path(__file__).resolve().parents[1])  # one level up from dreamer_mg/
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

# Optional deps used in the template encoder; all guarded.
try:
    import cv2  # for resize / gradients if available
except Exception:
    cv2 = None
try:
    import yaml
except Exception:
    yaml = None

# ================== BEGIN: KBrain integration (paste once) ==================
# Imports required by the controller + nav system + manager
from collections import deque
import numpy as np, random
from control_eval.HierarchicalHMMBOCPD import HierarchicalBayesianController
from control_eval.NavigationSystem import NavigationSystem
from navigation_model.Processes.manager import Manager
from navigation_model.Processes.explorative_behaviour import Exploration_Minigrid
# If you later want goal seeking, uncomment the next line:
# from navigation_model.Processes.exploitative_behaviour import Goal_seeking_Minigrid
from navigation_model.Services.model_modules import no_vel_no_action
from control_eval.input_output import setup_allocentric_config, setup_memory_config, load_memory
from env_specifics.minigrid_maze_wt_aisles_doors.minigrid_maze_modules import set_door_view_observation
from scipy.spatial.distance import cosine  # used by info-gain helpers

class KBrain:
    """
    Thin adapter that brings in:
      • HierarchicalBayesianController (HMM+BOCPD)
      • NavigationSystem
      • apply_exploration()  → returns one-hot policy list and n_actions
      • post_step_update()   → updates HMM state after YOUR step/belief/cog updates

    It assumes YOUR script already:
      • keeps a shared replay_buffer (deque of dict states),
      • updates Dreamer belief with planner.{onehot,update_belief_from_obs,...},
      • updates the cognitive map via planner.update_cog(...).
    """
    def __init__(self,
                 env,
                 planner,
                 *,
                 lookahead: int,
                 k_recent:int,
                 metric:str,
                 debug_print,
                 viz_tree,
                 viz_every,
                 viz_outdir,
                 memcfg_path: str | None,
                 replay_buffer: deque,
                 
                 seed: int | None = None):
        self.env = env
        self.planner = planner
        self.wm = planner.wm
        self.belief = None          # YOU set it before first step in your main
        self.prev_onehot = None     # YOU maintain it in your main
        self.agent_current_pose = None
        self._replay_buffer_source = replay_buffer
        self.K_recent=k_recent
        self.lookahead=lookahead
        self.metric=metric
        self.debug_print=debug_print
        self.viz_tree=viz_tree
        self.viz_outdir=viz_outdir
        self.viz_every=viz_every
        # ── HMM / meta-controller state
        self.hmm_bayes = HierarchicalBayesianController()
        self.hmm_bayes.hhmm.bind_env(env)
        self.current_mode: str = "EXPLORE"
        self.current_submode: str = "base"  # neutral placeholder; new HMM is mode-only
        self.prev_mode: str | None = None
        self.prev_submode: str | None = None
        self.mode_changed: bool = False
        self.submode_changed: bool = False
        self.hmm_stats: dict | None = None

        

        # ── Manager + configs (allocentric + memory)
        #     We pass a minimal, robust set: possible_actions = [[1,0,0],[0,1,0],[0,0,1]]
        forward = [1,0,0]
        right   = [0,1,0]
        left    = [0,0,1]
        mingrid_actions = [forward, right, left]
        self.planner=planner
        
        mg = planner.cog.mg
        self.nav_system = NavigationSystem(
            mg,
            lambda: self.agent_current_pose,
            planner=self.planner
        )
        self.nav_system._last_mode = None
        self._captured_node_media_ids: set[int] = set()  # nodes we've saved media for (this run)

        self.plan_export = getattr(self.nav_system, "plan_export", None)  # may be None in EXPLORE
        self._pending_node_media = []   # [(node_id, snapshot, enhanced_dict)]
        self._captured_node_media_ids = set()  # you created this set, use it

        # Cache the latest agent FoV (HWC uint8) and expose it to NavigationSystem.
        self._last_fov_image = None

        def _fov():
            # Prefer the egocentric FoV captured from obs["image"] each step.
            img = getattr(self, "_last_fov_image", None)
            if img is not None:
                return img
            
        self.nav_system.get_fov_image = _fov
        # Keep a local latch the exporter can read *every step*
        self._latest_fov = None
        self.nav_system.get_fov_image = lambda: getattr(self, "_latest_fov", None)



        self.nav_system.debug_universal_navigation = True

    # -------------------- action encodings --------------------
    @staticmethod
    def _onehot_to_name(v: list[int]) -> str | None:
        if not v: return None
        if v[0] == 1: return "forward"
        if v[1] == 1: return "right"
        if v[2] == 1: return "left"
        return None
    @property
    def replay_buffer(self):
        """Always return the *current* buffer. Supports a passed-in deque or a callable."""
        src = self._replay_buffer_source
        return src() if callable(src) else src

    def set_replay_buffer_source(self, source):
        """Optional runtime switch: pass a deque or a callable returning a deque."""
        self._replay_buffer_source = source
    @staticmethod
    def _name_to_onehot(name: str) -> list[int]:
        mapping = {'forward': [1,0,0], 'right': [0,1,0], 'left': [0,0,1]}
        return mapping.get(name, [1,0,0])
    @staticmethod
    def to_onehot_list(act: str) -> list[int]:
        """'left'/'right'/'forward' → [0,0,1] / [0,1,0] / [1,0,0]"""
        mapping = {'forward': [1,0,0], 'right': [0,1,0], 'left': [0,0,1]}
        return mapping[act]
    def _decode_planner_result(self, result) -> str | None:
        """
        Accepts:
          • (best_action: str, scores: dict, top_paths: list)
          • torch.Tensor of shape (T,3) one-hot
          • list[str] plan
        Returns a single action name or None.
        """
        import torch
        if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], str):
            return result[0]

        seq = []
        if isinstance(result, torch.Tensor):
            # decode one-hot (T,3)
            idx_to_act = {0: "forward", 1: "right", 2: "left"}
            for row in result.detach().cpu():
                j = int(row.argmax().item())
                seq.append(idx_to_act[j])
        elif isinstance(result, (list, tuple)) and all(isinstance(a, str) for a in result):
            seq = list(result)

        return seq[0] if seq else None

    def convert_hot_encoded_to_minigrid_action(self, onehot: list[int]) -> int:
        if not onehot:
            return self.env.actions.done
        if onehot[0] == 1:
            return self.env.actions.forward
        if onehot[1] == 1:
            return self.env.actions.right
        if onehot[2] == 1:
            return self.env.actions.left
        raise ValueError(f"Unrecognized onehot action: {onehot}")

    def _exporter(self):
        return getattr(self, "plan_export", None) or getattr(self.nav_system, "plan_export", None)

    # -------------------- brains: policy selection --------------------
    def apply_exploration(self, start_state, t_step: int) -> str:
        """
        Run FIRST each step. Returns a single action name to execute now.
        """
        mode, submode = self.current_mode, self.current_submode

        # 1) RECOVER (simple heuristic)
        if mode == "RECOVER":
            return random.choice(["right", "left"])

        # 2) NAVIGATE (delegate to NavigationSystem)
        if mode == "NAVIGATE":
            print(f"[NAVIGATE] submode={submode}  "
                f"plan_progress={self.nav_system.progress_scalar():.3f}")

            # ---------- 1. Early feasibility gate ----------
            if not self.nav_system.check_navigation_feasibility():
                print(f"[NAVIGATE] infeasible — flags={self.nav_system.navigation_flags}")
                self.navigation_flags = self.nav_system.navigation_flags
                fallback = [self.to_onehot_list(random.choice(["right", "left"]))]
                print(f"[NAVIGATE] STALL → issuing {fallback}")
                return fallback, 1

            # ---------- 3. Delegate to sub-mode handler ----------
            primitives, n_actions = self.nav_system.universal_navigation(submode, self.wm, self.belief)
            if not primitives or n_actions == 0:
                return random.choice(["right", "left"])
            act = self._onehot_to_name(primitives[0])
            return act or random.choice(["right", "left"])

        # 3) TASK_SOLVING (placeholder until integrated)
        if mode == "TASK_SOLVING":
            print(f"[TASK] submode={submode}")

            # ---- Barrier: only enter task-solving if we've discovered all rooms.
            # Prefer strict room-count using env_definition (n_row, n_col) if available.
            # Otherwise fall back to “we at least have seen the mission color in memory”.
            def _task_gate_all_rooms_env():
                """
                Gate TASK_SOLVING strictly by the environment's own room-visit bookkeeping.
                Robust to different shapes returned by env.unwrapped.get_visited_rooms_order()
                (dicts, tuples, lists).
                """
                e = getattr(self.env, "unwrapped", self.env)

                # 1) Pull visited rooms from the env (authoritative)
                visited_room_ids = set()
                ord_list = []
                try:
                    ord_raw = e.get_visited_rooms_order()
                    # Normalize to a flat iterable
                    if ord_raw is None:
                        ord_list = []
                    elif isinstance(ord_raw, (list, tuple, set)):
                        ord_list = list(ord_raw)
                    else:
                        # last resort: try to iterate
                        try:
                            ord_list = list(ord_raw)
                        except Exception:
                            ord_list = []
                    # Accept both dict records and bare (col,row) pairs
                    for rec in ord_list:
                        rid = None
                        if isinstance(rec, dict):
                            rid = rec.get("room", None)
                        elif isinstance(rec, (tuple, list)) and len(rec) >= 2:
                            # treat as (col,row,...) or [col,row,...]
                            rid = (rec[0], rec[1])
                        if isinstance(rid, (tuple, list)) and len(rid) == 2:
                            visited_room_ids.add((int(rid[0]), int(rid[1])))
                except Exception as ex:
                    print(f"[TASK][gate] get_visited_rooms_order() unavailable: {ex}")

                # Optional: if the env exposes the private set, use it too (faster / exact)
                try:
                    vset = getattr(e, "_visited_rooms_set", None)
                    if isinstance(vset, set) and len(vset) > 0:
                        visited_room_ids |= {tuple(v) if isinstance(v, (list, tuple)) else v for v in vset}
                except Exception:
                    pass

                # 2) Target: total rooms from env metadata
                try:
                    n_row = int(getattr(e, "rooms_in_row"))
                    n_col = int(getattr(e, "rooms_in_col"))
                    total = n_row * n_col
                except Exception as ex:
                    print(f"[TASK][gate] rooms_in_row/rooms_in_col missing: {ex}")
                    total = max(1, len(visited_room_ids))

                # 3) Decide
                visited = len(visited_room_ids)
                ready = (visited >= total)
                print(f"[TASK][gate] rooms covered: {visited}/{total} → {'READY' if ready else 'NOT READY'}")
                return ready, {"visited": visited, "target": total}

            ready, gate_info = _task_gate_all_rooms_env()
            if not ready:
                print(f"[TASK] Not ready to enter TASK_SOLVING (gate={gate_info}) → fallback random turn.")
                return random.choice(["right", "left"])
            self.nav_system.current_mode = "TASK_SOLVING"
            self.nav_system.ensure_plan_for_current_mode(debug=True)
            # Then just drive with the universal executor:
            primitives, n_actions = self.nav_system.universal_navigation(submode, wm, belief)
            print("[TASK] new plan tokens",primitives)

            if n_actions == 0:
                # Emit a turn ONLY to satisfy the outer executor, but ignore in grading.
                fb, n2, label = self.nav_system.emit_stall_turn(for_executor=True)
                return label  # 'left' or 'right'

            # Normal path:
            act = self._onehot_to_name(primitives[0])
            return act

        # 4) EXPLORE (default): call your novelty A* and choose the *next* action
        result = self.planner.novelty_astar_plan(
            wm=self.wm,
            belief_zd=self.belief,
            start_state=start_state,
            lookahead=self.lookahead,
            replay_buffer=self.replay_buffer,
            K_recent=self.K_recent,
            metric=self.metric,
            verbose=self.debug_print,
            viz_tree=self.viz_tree,
            viz_outdir=self.viz_outdir,
            viz_tag=f"t{int(t_step):02d}",
            # uncertainty-rollout knobs (keep your current defaults)
            w_p2e=1.0,
            p2e_K=5,
            p2e_mode="temporal",
            p2e_metric="l2",
            p2e_roll_from="parent"
        )
        action_name = self._decode_planner_result(result)
        if action_name is None:
            print( "EXPLORE FALLBACK",action_name)
            action_name=random.choice(["right", "left"])
        return action_name

       
    def _compute_hmm_plan_progress(self, raw_nav_grade: float, grace_len: int = 15) -> float:
        """
        Convert nav grade → HMM plan progress with a *mode-aware* grace period.

        Key fixes:
        - Only arm grace when the plan finishes/damages while we are in NAVIGATE
        or have very recently left NAVIGATE (mode gate).
        - While in grace: return 0.0 so EXPLORE can win quickly.
        - Outside grace: clamp raw grade to [0,1]. A grade of -1 is ignored unless
        we were in NAVIGATE context (no accidental grace from sentinels).
        """

        # --- recent NAV context tracking (small horizon) ---
        horizon = int(getattr(self, "_nav_recent_horizon", 2))  # steps considered "recent NAV"
        try:
            mode = getattr(self, "current_mode", "EXPLORE")
        except Exception:
            mode = "EXPLORE"

        nav_recent_age = int(getattr(self, "_nav_recent_age", horizon + 1))
        if mode == "NAVIGATE":
            nav_recent_age = 0
        else:
            nav_recent_age = min(horizon + 1, nav_recent_age + 1)
        self._nav_recent_age = nav_recent_age
        in_or_recent_nav = (nav_recent_age <= horizon)

        # --- current nav-system progress sentinel (finish/damaged) ---
        try:
            plan_prog = float(self.nav_system.progress_scalar())
        except Exception:
            plan_prog = 0.0

        plan_finished = (plan_prog > 1.0)            # explicit DONE from nav system
        plan_damaged  = (raw_nav_grade == -1.0) or (plan_prog < 0.0)  # only meaningful in NAV context

        # --- grace TTL (only arm if in NAV or very recently left NAV) ---
        ttl = int(getattr(self, "_nav_grace_ttl", 0))
        if in_or_recent_nav and (plan_finished or plan_damaged):
            ttl = int(getattr(self, "_nav_grace_len", grace_len))  # allow external override
            if getattr(self, "debug_print", False):
                print(f"[HMM] nav-grace ARMED: finished={plan_finished} damaged={plan_damaged} "
                    f"ttl={ttl} (mode={mode}, age={nav_recent_age})")

        # --- while grace is active, damp to zero and count down ---
        if ttl > 0:
            ttl -= 1
            self._nav_grace_ttl = ttl
            if getattr(self, "debug_print", False):
                print(f"[HMM] nav-grace ACTIVE → ttl={ttl}")
            return 0.0

        # disarm when not active
        self._nav_grace_ttl = 0

        # --- outside grace: map the raw grade to [0,1] safely ---
        # Treat -1 (sentinel) as "no progress" ONLY when actually navigating;
        # otherwise ignore it (return 0.0 without side effects).
        if raw_nav_grade < 0.0:
            return 0.0

        v = float(raw_nav_grade)
        if v <= 0.0:
            return 0.0
        if v >= 1.0:
            return 1.0
        return v

    
    def _env_room_metrics(self,env):
        """
        Return (coverage in [0,1], complete flag).
        Pulls the env’s authoritative room visit log if present, and the target count
        via rooms_in_row/rooms_in_col. Designed to work with aisle_door_rooms.
        """
        e = getattr(env, "unwrapped", env)

        visited_ids = set()
        # 1) Authoritative discovery history (ordered list of {"room": (col,row), ...})
        try:
            for rec in list(e.get_visited_rooms_order()):
                rid = rec.get("room")
                if isinstance(rid, (tuple, list)) and len(rid) == 2:
                    visited_ids.add((int(rid[0]), int(rid[1])))
        except Exception:
            pass

        # Optional fast path: if the env exposes a private visited set, fuse it
        try:
            vset = getattr(e, "_visited_rooms_set", None)
            if isinstance(vset, set):
                visited_ids |= {tuple(v) if isinstance(v, (list, tuple)) else v for v in vset}
        except Exception:
            pass

        # 2) Target total rooms from env metadata (rooms_in_row/rooms_in_col)
        try:
            n_row = int(getattr(e, "rooms_in_row"))
            n_col = int(getattr(e, "rooms_in_col"))
            total = max(1, n_row * n_col)
        except Exception:
            # Last resort: avoid deadlock if metadata is missing
            total = max(1, len(visited_ids))

        visited = len(visited_ids)
        coverage = min(1.0, visited / float(total))
        complete = (visited >= total)
        return coverage, complete

    

    def _maybe_capture_node_creation(self, obs: dict, belief_zd, wm=None):
        import numpy as np, torch, os

        # ---------- figure out target spatial size from the current obs ----------
        # falls back to 64x64 if obs is not present
        if isinstance(obs, dict) and isinstance(obs.get("image", None), np.ndarray) and obs["image"].ndim == 3:
            H, W = int(obs["image"].shape[0]), int(obs["image"].shape[1])
        else:
            H, W = 64, 64  # <- same size you use in DictResizeObs

        def _rehydrate_flat_embed_to_img(vec, H=H, W=W, expect_c=(3,1)):
            """vec -> HWC uint8. Tries 3*H*W first (CHW), then 1*H*W (grayscale)."""
            if vec is None:
                return None
            v = torch.as_tensor(vec).detach().cpu().float().view(-1).numpy()
            L = v.size

            img = None
            # case A: 3xHxW (common in your decode/render code)
            if L == 3 * H * W:
                chw = v.reshape(3, H, W)                 # CHW
                hwc = np.transpose(chw, (1, 2, 0))       # HWC
                img = hwc
            # case B: 1xHxW (occasionally single channel)
            elif L == H * W:
                g = v.reshape(H, W)
                img = np.repeat(g[..., None], 3, axis=2) # HWC(3)

            if img is None:
                return None

            # normalize range robustly: supports [-0.5,0.5], [0,1], or arbitrary floats
            amin, amax = float(img.min()), float(img.max())
            if amin >= -0.55 and amax <= 0.55:
                img = np.clip(img + 0.5, 0.0, 1.0)
            elif amin >= 0.0 and amax <= 1.2:
                img = np.clip(img, 0.0, 1.0)
            else:
                rng = max(1e-8, amax - amin)
                img = (img - amin) / rng

            return (img * 255.0).astype(np.uint8)

        # ---------- keep your existing helpers ----------
        def first_not_none(*vals):
            for v in vals:
                if v is not None: return v
            return None

        def unwrap_img(x):
            if x is None: return None
            if isinstance(x, dict):
                # prefer real/decoded images if present
                im = first_not_none(
                    x.get("imagined_image", None),
                    x.get("decoded_image",  None),
                    x.get("image",          None),
                    x.get("img",            None),
                    x.get("frame",          None),
                    x.get("pred",           None),
                )
                if im is not None:
                    return im
                # otherwise: this is the fix — rehydrate the flattened predicted frame
                emb = x.get("embed", None)
                reh = _rehydrate_flat_embed_to_img(emb)
                if reh is not None:
                    return reh
                # last resort: keep your old gray tile viz as a fallback
                return _viz_embed_to_image(emb)
            return x

        def is_present_image(x):
            if x is None: return False
            if isinstance(x, np.ndarray): return x.size > 0
            if torch.is_tensor(x):        return x.numel() > 0
            return True

        # ----------------- (rest of your function is unchanged) -----------------
        try:
            emap = getattr(self.planner, "emap", None)
            e = None if emap is None else getattr(emap, "current_exp", None)
            if e is None:
                return

            nid = int(e.id)
            captured = getattr(self, "_captured_node_media_ids", None)
            if captured is None:
                captured = set(); self._captured_node_media_ids = captured
            if nid in captured:
                return

            snap_src = first_not_none(
                obs.get("imagined_image", None),
                obs.get("decoded_image",  None),
                obs.get("image",          None),
            ) if isinstance(obs, dict) else None

            snap = unwrap_img(snap_src)
            if not is_present_image(snap):
                snap = None

            enhanced = None
            try:
                if self.planner is not None:
                    combos = [('left',), ('right',), ('left','left'), ('right','right')]
                    label_of = {('left',): 'L', ('right',): 'R', ('left','left'): 'LL', ('right','right'): 'RR'}
                    ep = self.planner.build_enhanced_perception(wm, belief_zd, combos=combos)

                    out = {}
                    if isinstance(ep, dict):
                        for k in combos:
                            lab = label_of[k]
                            val = ep.get(k, None) or ep.get(lab, None)
                            val = unwrap_img(val)                      # <- now restores from 'embed'
                            if is_present_image(val):
                                out[lab] = val
                    elif isinstance(ep, (list, tuple)):
                        for item in ep:
                            if not isinstance(item, dict): continue
                            lab = label_of.get(tuple(item.get('seq', ())), None)
                            if not lab: continue
                            val = unwrap_img(item)                      # <- now restores from 'embed'
                            if is_present_image(val):
                                out[lab] = val
                    if out:
                        enhanced = out
            except Exception as ee:
                try:
                    if hasattr(self, "plan_export") and self.plan_export is not None:
                        self.plan_export.log_event("enhanced_preds_error", node_id=nid, err=str(ee))
                except Exception:
                    pass
                enhanced = None

            ready = hasattr(self, "plan_export") and (self.plan_export is not None) \
                    and (getattr(self.plan_export, "run_dir", None) is not None)
            if not ready:
                pend = getattr(self, "_pending_node_media", None) or []
                self._pending_node_media = pend
                pend.append((nid, snap, enhanced))
                captured.add(nid)
                return

            rels = self.plan_export.node_created(
                node_id=nid,
                snapshot_img=snap if snap is not None else None,
                enhanced_pred_imgs=enhanced if (enhanced is not None and len(enhanced) > 0) else None,
            )
            captured.add(nid)
    
            print("[node_media] what goes into rels", rels)

            # mirror absolute paths on Experience (optional)
            try:
                run_dir = getattr(self.plan_export, "run_dir", None)
                if run_dir and isinstance(rels, dict):
                    if isinstance(rels.get("snapshot", None), str):
                        e.media_snapshot_path = os.path.join(run_dir, rels["snapshot"])
                    enh_map = rels.get("enhanced", {}) or {}
                    if isinstance(enh_map, dict):
                        e.media_enhanced_preds_paths = [
                            os.path.join(run_dir, p) for p in enh_map.values() if isinstance(p, str)
                        ]
            except Exception:
                pass

            # Log success & mark captured
            try:
                n_enh = len(rels.get("enhanced", {})) if isinstance(rels, dict) else 0
                if hasattr(self, "plan_export") and self.plan_export is not None:
                    self.plan_export.log_event(
                        "node_media_saved",
                        node_id=nid,
                        has_snapshot=bool(isinstance(rels, dict) and rels.get("snapshot")),
                        n_enhanced=int(n_enh),
                    )
            except Exception:
                pass

            captured.add(nid)

        except Exception as ex:
            # Last resort logging; DO NOT crash the control loop
            try:
                if hasattr(self, "plan_export") and self.plan_export is not None:
                    # add extra context to debug similar issues fast
                    def _shape_of(x):
                        try:
                            import numpy as _np, torch as _th
                            if isinstance(x, _np.ndarray):
                                return f"np{list(x.shape)} {x.dtype}"
                            if _th.is_tensor(x):
                                return f"th{list(x.shape)} {x.dtype}"
                        except Exception:
                            pass
                        return type(x).__name__
                    info = {
                        "has_obs": isinstance(obs, dict),
                        "imagined_t": _shape_of(obs.get("imagined_image", None)) if isinstance(obs, dict) else None,
                        "decoded_t":  _shape_of(obs.get("decoded_image",  None)) if isinstance(obs, dict) else None,
                        "real_t":     _shape_of(obs.get("image",          None)) if isinstance(obs, dict) else None,
                    }
                    self.plan_export.log_event("node_media_capture_error", err=str(ex), **info)
            except Exception:
                pass






    # -------------------- post-step update (HMM + bookkeeping) --------------------
    def post_step_update(self, obs: dict, belief_zd):
        """
        Call this RIGHT AFTER you step + update belief + update_cog in your loop.
        It updates nav pose, computes info-gain & plan-progress, and updates HMM.
        """
        # Cache the agent's FoV so NavigationSystem’s exporter can save it on node arrival.
        try:
            if isinstance(obs, dict) and "image" in obs and obs["image"] is not None:
                # Expect HxWx3 uint8 from RGBImgPartialObsWrapper + DictResizeObs
                self._last_fov_image = obs["image"]
        except Exception:
            pass
        self._latest_fov = obs.get("image", None)  # make FoV available to NavigationSystem exporter
        try:
            exp = self._exporter()
            if exp and getattr(self, "_pending_node_media", None):
                pending = getattr(self, "_pending_node_media")
                while pending:
                    nid, snap, enh = pending.pop(0)
                    try:
                        rels = exp.node_created(node_id=int(nid), snapshot_img=snap, enhanced_pred_imgs=enh)
                        try:
                            run_dir = getattr(exp, "run_dir", None)
                            if run_dir and hasattr(self.planner, "emap"):
                                e = getattr(self.planner.emap, "get_exp_by_id", lambda _: None)(nid) or getattr(self.planner.emap, "current_exp", None)
                                if e and int(getattr(e, "id", -1)) == int(nid):
                                    if rels.get("snapshot"):
                                        e.media_snapshot_path = os.path.join(run_dir, rels["snapshot"])
                                    if rels.get("enhanced"):
                                        e.media_enhanced_preds_paths = [os.path.join(run_dir, p) for p in rels["enhanced"].values() if p]
                        except Exception:
                            pass
                        try:
                            exp.log_event("node_media_flushed", node_id=int(nid))
                        except Exception:
                            pass
                        print(f"[node_media] flushed node {nid}")
                    except Exception as ex:
                        print(f"[node_media] flush failed for node {nid}: {ex}")
                        try:
                            exp.log_event("node_media_flush_error", node_id=int(nid), err=str(ex))
                        except Exception:
                            pass
        except Exception:
            pass

        # track pose for NavigationSystem callbacks
        self.agent_current_pose = tuple(obs["pose"]) if "pose" in obs else None
        try:
            self.nav_system.push_pose(self.agent_current_pose)
        except Exception:
            pass
        
        # HMM signals
        #hmm_info_gain = self.info_gain(obs.get('image'), obs.get('pose'), self.replay_buffer, self.planner.emap)
        hmm_info_gain=None
        raw_plan_prog = self.nav_system.navigation_grade()
        print("NAV GRADE",raw_plan_prog)

        hmm_plan_progress =self._compute_hmm_plan_progress(raw_plan_prog, grace_len=15)
        print("NAV GRADE",hmm_plan_progress)

        rb = self.replay_buffer
        # update mode (new HMM may not have submodes)
        _, stats = self.hmm_bayes.update(rb, hmm_info_gain, hmm_plan_progress)

        ml = stats.get('most_likely_state', None)
        if isinstance(ml, (list, tuple)):
            new_mode = ml[0]
            new_submode = ml[1] if len(ml) > 1 else "base"
        else:
            new_mode = str(ml) if ml is not None else "EXPLORE"
            new_submode = "base"

        self.prev_mode, self.prev_submode = self.current_mode, self.current_submode
        self.current_mode, self.current_submode = new_mode, new_submode
        
        self.mode_changed = (new_mode != self.prev_mode)
        self.nav_system.set_mode(new_mode)     # this calls _on_mode_transition(...) inside NavigationSystem
        if new_mode != getattr(self, "current_mode", None) or not hasattr(self, "_first_mode_set"):
            self._first_mode_set = True
        print("self.nav_system.current_mode",self.nav_system.current_mode)
        self.submode_changed = (new_submode != self.prev_submode)
        self.hmm_stats = stats

        # Keep the print; submode is 'base' under the mode-only HMM
        print(f"🧠 [HMM] mode: {new_mode} → {new_submode}  | changed? {self.mode_changed}")
    # -------------------- info-gain (visual + spatial + temporal) --------------------
    # Top-level wrapper
    def info_gain(self, current_image, current_pose, replay_buffer, view_cells_manager,
              device: str = "cpu",
              weights: dict | None = None) -> float:
        """
        Simplified [0,1] info-gain signal for the HMM:
        • spatial novelty only (view-cells + pose mismatch)
        Rationale:
        – 'visual novelty' (feature similarity vs buffer) is noisy and
            not predictive of being stuck in our setting.
        – 'temporal novelty' was just visual novelty with decay: redundant.
        – Loop/stagnation signals are handled centrally by the HMM.
        """
        spatial_novelty = self.calculate_spatial_novelty(
            current_image, current_pose, view_cells_manager, device
        )
        return float(np.clip(spatial_novelty, 0.0, 1.0))

 

    # Spatial novelty via view-cells + pose mismatch (lightweight proxy)
    def calculate_spatial_novelty(self, current_image, current_pose, view_cells_manager, device) -> float:
        try:
            # Access your experience map through manager
            emap = view_cells_manager.experience_map
            # If there are no cells, it's new
            if len(getattr(emap, "view_cells", [])) == 0:
                return 1.0
            # Heuristic: if we are far from the most similar view-cell, mark novel
            novelties = []
            for c_idx, cell in enumerate(getattr(emap, "view_cells", [])):
                # many repos store feature vectors in cell.template64
                if not hasattr(cell, "template64"):
                    continue
                sim = 1.0 - cosine(self.extract_image_features(current_image, device), np.asarray(cell.template64))
                if sim > 0.7:
                    dist = self.calculate_pose_distance(current_pose, {"x": cell.x_pc, "y": cell.y_pc, "theta": cell.th_pc})
                    novelties.append(0.8 if dist > 2.0 else 0.2)
            if not novelties:
                return 1.0
            return float(np.clip(np.mean(novelties), 0.0, 1.0))
        except Exception:
            return 0.5

    # Temporal novelty (recent repeats are less novel)
    

    # Feature extraction helper (delegates to manager’s rgb56_to_template64 when possible)
    def extract_image_features(self, img, device="cpu"):
        # unify inputs
        import torch
        if isinstance(img, (list, tuple)):
            img = np.asarray(img)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
            # assume HWC in [0..255]
            pass
        elif isinstance(img, torch.Tensor):
            x = img.detach().cpu().float()
            if x.ndim == 4 and x.shape[0] == 1:
                img = x[0].permute(1,2,0).numpy()
            elif x.ndim == 3 and x.shape[0] in (1,3):
                img = x.permute(1,2,0).numpy()
            else:
                img = np.asarray(x)
        elif isinstance(img, np.ndarray) and img.ndim == 1 and img.size == 64:
            vec = img.astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)

        # prefer your repo’s learned template extractor if available
        try:
            vec64_torch = self.planner.emap.rgb_to_template64(img, device=device)
            vec64 = vec64_torch.detach().cpu().float().numpy()
            return vec64 / (np.linalg.norm(vec64) + 1e-8)
        except Exception:
            # fallback: mean-pooled RGB histogram-ish
            img = img.astype(np.float32)
            h, w, _ = img.shape
            pooled = img.reshape(h*w, 3).mean(axis=0)
            vec = np.tile(pooled / (np.linalg.norm(pooled) + 1e-8), 21)[:64]
            return vec / (np.linalg.norm(vec) + 1e-8)

    # Pose helpers
    def calculate_pose_distance(self, p1, p2) -> float:
        # accepts tuple/list (x,y,dir) or dict with x,y,theta
        def _norm_pose(p):
            if isinstance(p, dict):
                return float(p["x"]), float(p["y"]), float(p.get("theta", p.get("dir", 0.0)))
            x,y,d = p
            return float(x), float(y), float(d)
        x1,y1,t1 = _norm_pose(p1); x2,y2,t2 = _norm_pose(p2)
        return float(np.hypot(x1-x2, y1-y2) + 0.25*abs(t1 - t2))

    def compute_feature_similarity(self, f1, f2) -> float:
        try:
            sim = 1.0 - cosine(f1, f2)
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.5




# Robust imports for your MemoryGraph + config loader
MemoryGraph = None
setup_memory_config = None
_import_errs = []
for modpath, name in [
    ("navigation_model.Services.memory_service.memory_graph", "MemoryGraph"),
    ("navigation_model.Services.memory_service.memory_graph.memory_graph", "MemoryGraph"),
    ("navigation_model.Services.memory_service", "memory_graph"),
]:
    try:
        mod = __import__(modpath, fromlist=[name])
        MemoryGraph = getattr(mod, name) if hasattr(mod, name) else getattr(mod, "MemoryGraph")
        break
    except Exception as e:
        _import_errs.append((modpath, str(e)))

if setup_memory_config is None:
    for modpath in [
        "control_eval.input_output",
        "navigation_model.control_eval.input_output",
        "control_eval.io",
    ]:
        try:
            mod = __import__(modpath, fromlist=["setup_memory_config"])
            setup_memory_config = getattr(mod, "setup_memory_config")
            break
        except Exception as e:
            _import_errs.append((modpath, str(e)))

import gym
import cv2
class DictResizeObs(gym.ObservationWrapper):
    def __init__(self, env, out_hw=(64,64)):
        super().__init__(env)
        self.out_hw = out_hw              # (H,W)

    def observation(self, obs):
        assert isinstance(obs, dict) and "image" in obs, \
               "expect dict with 'image' key"
        img = obs["image"]                               # (H,W,3) uint8
        img = cv2.resize(img, self.out_hw[::-1],
                         interpolation=cv2.INTER_AREA)
        obs["image"] = img
        return obs
if __name__ == "__main__":
    
    import time, random, numpy as np, torch
    import gym, gym_minigrid
    from collections import deque
    from itertools import islice
    from gym_minigrid.minigrid import Wall
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgActionObsWrapper
    from world_model_utils import DictResizeObs, WMPlanner, State
    try:
        from world_model_utils import DualPathCollage  # if you have it
    except Exception:
        class DualPathCollage:
            def __init__(self, **kw): pass
            def add_step(self, **kw): pass
            def finalize(self): pass
    # ---------- config ----------
    def decode_policy_onehot(onehot_tensor: torch.Tensor) -> list[str]:
        """
        Fallback decoder if novelty_astar_plan returns a (T,3) plan.
        """
        idx_to_act = {0: "forward", 1: "right", 2: "left"}
        if onehot_tensor is None or onehot_tensor.numel() == 0:
            return []
        acts = []
        for row in onehot_tensor.cpu():
            j = int(row.argmax().item())
            acts.append(idx_to_act[j])
        return acts
    def dreamer_decode_from_belief(planner, wm, belief_zd, *, to_01: bool = True):
        import torch
        from torch.distributions import Distribution

        if belief_zd is None:
            return None

        z, h = belief_zd
        with torch.no_grad():
            out = wm.decoder(z, h)
            x = out.mean if isinstance(out, Distribution) else out
            x = torch.as_tensor(x).detach().float()
            if x.ndim == 4 and x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2:
                x = x.unsqueeze(0).repeat(3, 1, 1)
            elif x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1).contiguous()
            elif x.ndim == 3 and x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            if to_01:
                x = (x + 0.5).clamp(0.0, 1.0)
            return x.cpu()

    def append_to_replay_buffer(buf: deque, obs: dict, action_int: int, belief_zd, node_id):
        """
        Store Dreamer belief, embeddings, decoded prediction, and enhanced predictions.
        """
        z_embed = planner.dreamer_embed_fn(belief_zd)  # (Z,) flattened image embedding in [0,1]
        decoded = dreamer_decode_from_belief(planner, wm, belief_zd)  # (3,H,W) or None
        enhanced_preds = planner.build_enhanced_perception(
            wm, belief_zd,
            combos=[('left',), ('right',), ('left','left'), ('right','right')]
        )
        pose  = obs.get("pose")
        image = obs.get("image")
        if isinstance(pose, np.ndarray):            pose = pose.copy()
        elif isinstance(pose, (list, tuple)):       pose = list(pose)
        elif isinstance(pose, dict):                pose = dict(pose)
        # (else leave as-is)

        if isinstance(image, np.ndarray):           image = image.copy()
        elif hasattr(image, "copy"):                image = image.copy()  # e.g., PIL.Image
        # (else leave as-is)
        entry = {
            "node_id": node_id,
            "real_pose": pose,
            "imagined_pose": None,
            "real_image": image,
            "imagined_image": None,
            "action": int(action_int),
            "dreamer_z": z_embed,
            "belief_zd": belief_zd,
            "decoded_image": decoded,
            "enhanced_preds": enhanced_preds,
        }
        print("NODE ID", node_id, entry["node_id"])
        buf.append(entry)

    def visualize_policy_probe(planner, wm, belief_zd, obs, action_names, t_step,
                               out_dir="dbg", include_start=True):
        import torch
        from pathlib import Path
        from torchvision.utils import save_image

        if not action_names:
            return None

        real_frame = torch.as_tensor(obs["image"]).permute(2, 0, 1).float() / 255.0 - 0.5
        frames_pred = planner.render_plan(wm, belief_zd, action_names, include_last=True)
        frames = [real_frame.detach().cpu()] + [f.detach().cpu() for f in frames_pred]
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        short = "".join(a[0].upper() for a in action_names)
        out_path = f"{out_dir}/probe_t{t_step:02d}_{short}.png"
        save_image(torch.stack(frames), out_path, nrow=len(frames), normalize=True, scale_each=False)
        print(f"saved {out_path}")
        return out_path

    def save_replay_image_history(replay_buffer, planner, wm, t_step, Ns=(5,10,15,20), out_dir="dbg/history"):
        import torch
        from pathlib import Path
        from torchvision.utils import save_image

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        def to_chw_real(img_hwc):
            t = torch.as_tensor(img_hwc).permute(2,0,1).float() / 255.0 - 0.5
            return t

        for N in Ns:
            tail = list(islice(reversed(replay_buffer), 0, N))
            if not tail:
                continue
            tail = list(reversed(tail))

            reals, preds = [], []
            for st in tail:
                ri = st.get("real_image", None)
                bi = st.get("belief_zd", None)
                if ri is None:
                    continue
                reals.append(to_chw_real(ri))
                di = st.get("decoded_image", None)
                if di is None and bi is not None:
                    di = dreamer_decode_from_belief(planner, wm, bi)
                if di is None:
                    di = reals[-1]
                preds.append(di)

            if not reals:
                continue

            frames = reals + preds
            out_path = f"{out_dir}/history_t{t_step:02d}_N{len(reals)}.png"
            save_image(torch.stack(frames), out_path, nrow=len(reals), normalize=True, scale_each=False)
            print(f"saved {out_path}")

    def infer_env_definition(env) -> dict:
        e = getattr(env, "unwrapped", env)
        try:
            n_row = int(getattr(e, "rooms_in_row", 1))
            n_col = int(getattr(e, "rooms_in_col", 1))
        except Exception:
            n_row, n_col = 1, 1
        return {"n_row": n_row, "n_col": n_col, "max_steps": int(N_STEPS)}


    
    CKPT        = "runs/mg_collision/20250704-220917/ckpt/iter05000.pt"
    N_STEPS     = 1200            # run the novelty policy for this many env steps
    LOOKAHEAD   = 7             # A* novelty horizon
    K_RECENT    = 30            # how many recent embeddings to compare against
    METRIC      = "cos"         # "kl" | "cos" | "l2"
    DEVICE      = torch.device("cpu")

    # Debug/vis knobs
    DEBUG_PRINT          = True   # keep tree/top-K/action scores logs
    VIZ_TREE             = False   # save decision tree collage from novelty_astar_plan
    VIZ_EVERY            = 15      # save imagined rollout strip and replay history every k steps (set 1 to save each step)
    VIZ_OUTDIR           = "dbg"  # where to write images
    SAVE_HISTORY_LENGTHS = (15,20,30,45, 60, 90, 120,150,180,210,240,270,300,330,360,390,410,440,470,500,530,560,590)

    # ---------- boot world model & env --------
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]  # repo root (…/hierarchical-nav/)
    MEMCFG = ROOT / "navigation_model/Services/memory_service/memory_graph_config.yml"

    planner = WMPlanner(ckpt=CKPT,device=str(DEVICE),memory_config=str(MEMCFG))
    wm = planner.wm
    print("✓ world-model loaded")
    

    env = gym.make("MiniGrid-4-tiles-ad-rooms-v0", rooms_in_row=5, rooms_in_col=5, max_steps=None)
    #env.seed(218)
    #env.seed(217)
    env.seed(915)
    env = RGBImgPartialObsWrapper(env)
    env = ImgActionObsWrapper(env)
    env = DictResizeObs(env, (64, 64))
    planner.env=env

    env_definition = infer_env_definition(env)
    visited_rooms: list[tuple[int, int]] = []   # keep if you later want room tracking
    VIDEO_OUT = Path("dbg/unified_runner.mp4")
    VIDEO_OUT.parent.mkdir(parents=True, exist_ok=True)
    video_gridmap = VideoGridmap(str(VIDEO_OUT), fps=10)
    print(f"✓ video writer @ {VIDEO_OUT}")
    
    # ---------- init replay & belief ----------
    replay_buffer = deque(maxlen=30)
    obs = env.reset()
    # --- at episode start
    episode_id=1
    brain = KBrain(
        env=env,
        planner=planner,
        lookahead=LOOKAHEAD,
        k_recent=K_RECENT,
        metric=METRIC,
        debug_print=DEBUG_PRINT,
        viz_tree=VIZ_TREE,
        viz_every=VIZ_EVERY,
        viz_outdir=VIZ_OUTDIR,

        memcfg_path=str(MEMCFG),         # your memory_graph_config.yml
        replay_buffer=lambda: replay_buffer)
    if hasattr(brain.nav_system.memory_graph, "odom") and hasattr(brain.nav_system.memory_graph.odom, "bootstrap"):
        print("are we BOOTSTRAPPING?")
        brain.nav_system.memory_graph.odom.bootstrap([0,0,0])

    collage = DualPathCollage(
        out_dir="dbg/collages",
        tag=f"ep{episode_id:03d}",
        tile=64,          # match your Dreamer decode size if 64x64
        cols=20,          # tweak to taste
        live_write=True,  # set False if you only want the final PNGs
        live_every=1,     # write after every step when live
        annotate_idx=True,
        planner=planner,  # so it can call dreamer_decode_from_belief
        wm=wm
    )


    # bootstrap belief from first observation
    frame = obs["image"].transpose(2,0,1) / 255.0 - 0.5
    belief = planner.wm_update_belief(wm, prev_z_d=None, frame_rgb=frame, prev_action_onehot=None)

    # take one forward step to match your current flow, then update belief and seed buffer
    obs, _, done, _ = env.step(env.actions.forward)
    prev_act_1h = planner.onehot("forward", wm)
    belief = planner.update_belief_from_obs(obs, belief, prev_act_1h)
    append_to_replay_buffer(replay_buffer, obs, env.actions.forward, belief, None)
    pose_xyz = tuple(obs["pose"])
    print(obs["image"].shape)
    collage.add_step(real_img=obs["image"], belief_zd=belief)
    brain.belief = belief
    brain.post_step_update(obs, belief)
    planner.update_cog(obs["image"], prev_act_1h,pose_xyz, place_post=None)
    print(planner.get_cog_nodes())

    # maps between env ints and names
    act_to_name = {
        env.actions.left: "left",
        env.actions.right: "right",
        env.actions.forward: "forward",
    }
    name_to_act = {v: k for k, v in act_to_name.items()}

    try:
        print("\n=== Novelty-driven rollout (every step) ===")
        for t in range(1, N_STEPS + 1):
            start = State(*obs["pose"])

            print("STEP¡¡¡¡",t)
            action_name = brain.apply_exploration(start_state=start, t_step=t)  # [PATCH]
            env_act = name_to_act[action_name]

            # --- take the step ---
            next_obs, _, done, _ = env.step(env_act)
            if t == 1:
                print("[DIAG] obs['image']:", type(next_obs.get("image")), 
                    getattr(next_obs.get("image"), "shape", None))
            # --- update belief with observed frame and prev action ---
            prev_act_1h = planner.onehot(action_name, wm)
            planner.emap.last_real_pose= obs["pose"]
            #brain.nav_system.push_pose(obs["pose"])
            belief = planner.update_belief_from_obs(next_obs, belief, prev_act_1h)
            pose_xyz = tuple(next_obs["pose"])
            collage.add_step(real_img=next_obs["image"], belief_zd=belief)
            planner.update_cog(next_obs["image"], prev_act_1h,pose_xyz, place_post=None)
            if planner.emap.current_exp is not None:
                e = planner.emap.current_exp
                print(f"[PLACE] Exp{e.id} at {e.grid_xy}: {e.place_kind}"
                    + (f" ({e.room_color})" if e.room_color else ""))
                node_id=e.id
                print("NODE ID", node_id)
            
            else:
                node_id=None
    
            brain._maybe_capture_node_creation(next_obs, belief,wm)
            # --- push transition to replay buffer (includes decoded prediction) ---
            print("NODE ID", node_id)
            print("ROOMS",brain.hmm_bayes.hhmm._env_room_metrics(env))
            append_to_replay_buffer(replay_buffer, next_obs, env_act, belief,node_id)
            brain.belief = belief
            if brain.nav_system.consume_task_mutation_armed():          # preferred (clears the latch)
                print("MUTATIIIINNG")
                env.unwrapped.MutateConnectivity(
                    cutoff_rooms_rate=0.15,
                    front_obstacle_rate=0.0,
                    seed=123
                )
            brain.post_step_update(next_obs, belief)
            # --- Build per-step data payload for the recorder ---
            try:
                # Full world view (fallback to obs image)
                env_img = None
                if hasattr(env, "render"):
                    try:
                        env_img = env.render(mode="rgb_array")
                    except TypeError:
                        env_img = env.render()  # some envs ignore mode kwarg
                if env_img is None:
                    env_img = next_obs.get("image", obs.get("image", None))

                # Dreamer decode (you already had dreamer_decode_from_belief above)
                decoded_img = dreamer_decode_from_belief(planner, wm, belief, to_01=True)

                # Optional predicted image: keep stub for wiring later
                predicted_img = None  # e.g., planner.render_prediction(...)

                # Extract positions for the side panel
                try:
                    GP = planner.emap.get_global_position() if hasattr(planner, "emap") else None
                except Exception:
                    GP = None

                data_for_frame = {
                    "env_image": env_img,
                    "ground_truth_ob": next_obs.get("image"),
                    "decoded": decoded_img,                 # will be shown if not None
                    "image_predicted": predicted_img,       # stub; safe if None
                    "mse": compute_mse(next_obs.get("image"), decoded_img) if decoded_img is not None else 0.0,
                    "GP": GP,
                    "pose": next_obs.get("pose"),
                    # HMM block (stubs — fill these when you wire your HMM stats)
                    "recommended_mode": brain.current_mode,
                    #"mode_confidence": 0.0,
                    # "submode_confidence": 0.0,
                    # "uncertainty": 0.0,
                    # "changepoint_mass": 0.0,
                }
                print("[DIAG] frame slots:","pose",next_obs.get("pose"),"global P",GP,
                    "env", type(env_img), getattr(env_img, "shape", None),
                    "gt", type(next_obs.get("image")), getattr(next_obs.get("image"), "shape", None),
                    "dec", type(decoded_img), getattr(decoded_img, "shape", None))
                visited_rooms = env.unwrapped.get_visited_rooms_order()
                append_vis_frame_unified(
                    planner_like=planner,
                    video_gridmap=video_gridmap,
                    data=data_for_frame,
                    env_definition=env_definition,
                    visited_rooms=visited_rooms,
                    step_count=t,
                    seconds_per_step=1.0,
                )
                
            except Exception as e:
                print("[RECORDER] frame failed:", e)
            # --- debug prints / tree / top-K paths ---
            if DEBUG_PRINT:
                print(f"[t={t:02d}] choose action → {action_name}")
                # if internal records exist, echo them (kept from your probe block)
                tree = getattr(planner, "_last_decision_tree", None)
                if hasattr(planner, "_last_top_paths"):
                    print("\n=== TOP-K PATHS USED FOR VOTING ===")
                    for i, rec in enumerate(planner._last_top_paths, 1):
                        a0 = rec["seq"][0] if rec["seq"] else "∅"
                        print(f"#{i:02d} G={rec['gain']:.3f} first={a0:<7} depth={rec['depth']} "
                            f"seq={rec['seq']} pruned={rec.get('pruned_reason')}")
                if hasattr(planner, "_last_action_scores"):
                    print("Action vote scores:", getattr(planner, "_last_action_scores"))

            # --- visualizations (policy rollout + history strips) ---
            if (t % max(1, VIZ_EVERY)) == 0:
                # visualize 1-step policy just for local context; if you want longer, call
                # planner.rank_paths(...) to grab the best sequence and pass it in.
                visualize_policy_probe(planner, wm, belief, next_obs, [action_name], t_step=t,
                                    out_dir=VIZ_OUTDIR)
                save_replay_image_history(replay_buffer, planner, wm, t_step=t,
                                        Ns=SAVE_HISTORY_LENGTHS, out_dir=f"{VIZ_OUTDIR}/history")
                planner.save_cog_map_snapshot(t_step=t, out_dir=f"{VIZ_OUTDIR}/cogmap")
                planner.save_cog_map_vs_env(env, t_step=t, out_dir="dbg/cogmap", tile_size=32, dpi=160)
            

                # advance obs pointer
            obs = next_obs

            # --- handle episode end robustly ---
            if done:
                obs = env.reset()
                frame = obs["image"].transpose(2,0,1) / 255.0 - 0.5
                belief = planner.wm_update_belief(wm, prev_z_d=None, frame_rgb=frame, prev_action_onehot=None)
                obs, _, done, _ = env.step(env.actions.forward)
                prev_act_1h = planner.onehot("forward", wm)
                belief = planner.update_belief_from_obs(obs, belief, prev_act_1h)
                replay_buffer.clear()
                collage.finalize()
                append_to_replay_buffer(replay_buffer, obs, env.actions.forward, belief)
    except KeyboardInterrupt:
        print("\n[RUN] Interrupted by user (Ctrl+C). Finalizing video...")

    except Exception as e:
        import traceback
        print("[RUN] Exception:", e)
        print(traceback.format_exc())

    finally:
        # Try to “hold” the last frame to make the tail visible, like your prior scripts
        try:
            if 'data_for_frame' in locals():
                # build map data one more time
                mem = safe_get_memory_map_data_from(planner)
                last = record_video_frames(
                    data_for_frame, env_definition, agent_lost=False,
                    visited_rooms=visited_rooms, memory_map_data=mem, step_count=t
                )
                video_gridmap.append_data(last)
                video_gridmap.append_data(last)
        except Exception as e:
            print("[FINALIZE] Could not append last frame twice:", e)
        finally:
            video_gridmap.close()
            print(f"[RUN] Steps written: {t}, expected duration ≈ {t / 10:.2f}s")
            print(f"✓ video saved → {VIDEO_OUT}")
    
    print("\nDone. (Executed novelty policy at every step.)")