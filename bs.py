"""def build_pcfg_from_memory(self,debug: bool = True) -> PCFG:
        mg    = self.memory_graph
        emap  = mg.experience_map
        start = mg.get_current_exp_id()

        # ---------- 1. adjacency list ----------------------------------
        graph = {e.id: [l.target.id for l in e.links] for e in emap.exps}
        if debug:
            print("[PCFG DEBUG] graph:", graph)

        # ---------- 2. confidence-weighted goal prior ------------------
        goals, Z_prior = {}, 0.0
        sx, sy, _ = emap.get_pose(start)
        for exp in emap.exps:
            if exp.id == start:
                continue
            conf = getattr(exp, "confidence", 1.0)
            dist = math.hypot(exp.x_m - sx, exp.y_m - sy) + 1e-5
            w    = conf / dist
            goals[exp.id] = w
            Z_prior      += w

        # ---------- 3. helper to enumerate a few paths ----------------
        def all_paths(src, dst, k=12, depth=15):
            out, q = [], deque([[src]])
            while q and len(out) < k:
                p = q.popleft()
                if p[-1] == dst:
                    out.append(p)
                elif len(p) < depth:
                    for nb in graph.get(p[-1], []):
                        if nb not in p:
                            q.append(p + [nb])
            return out

        # ---------- 4. construct production rules ---------------------
        rules = defaultdict(list)

        # 4-A.   NAVPLAN → PATH_t  (goal prior)
        for tgt, w in goals.items():
            rules["NAVPLAN"].append((f"PATH_{tgt}", w / Z_prior))

        # 4-B.   PATH_t  → STEP_*_* STEP_*_* ...
        for tgt in goals:
            for path in all_paths(start, tgt):
                # prepend self-edge STEP_s_s
                step_tokens = []
                if not self._at_node_exact(start):
                    step_tokens.append(f"STEP_{start}_{start}")   # only if we are *away*
                step_tokens += [f"STEP_{u}_{v}" for u, v in zip(path, path[1:])]
                rhs = " ".join(step_tokens)
                rules[f"PATH_{tgt}"].append((rhs, 1.0))       # equal weight

        # 4-C.   every STEP_u_v becomes a *terminal* symbol
        for lhs in list(rules.keys()):
            if lhs.startswith("PATH_"):
                for rhs, _ in rules[lhs]:
                    for tok in rhs.split():
                        if tok not in rules:                  # first encounter
                            rules[tok].append((f"'{tok}'", 1.0))

        # ---------- 5. serialise to NLTK PCFG --------------------------
        lines = []
        for lhs, prods in rules.items():
            Z = sum(p for _, p in prods)
            for rhs, p in prods:
                lines.append(f"{lhs} -> {rhs} [{p/Z:.4f}]")

        grammar_src = "\n".join(lines)
        if debug:
            print("[PCFG DEBUG] Final grammar:\n" + grammar_src)

        return PCFG.fromstring(grammar_src) """