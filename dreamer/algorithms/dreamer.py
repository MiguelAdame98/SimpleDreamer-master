# dreamer/algorithms/dreamer.py
import logging, time, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from dreamer.modules.model   import RSSM, RewardModel, ContinueModel
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.utils.utils     import (
    compute_lambda_values, create_normal_dist, DynamicInfos
)
from dreamer.modules.actor import Actor
from dreamer.utils.buffer    import ReplayBuffer
from pathlib import Path
import random
import gym_minigrid
from gym_minigrid.minigrid import Wall

# --------------------------------------------------------------------------- #
#  nice-looking logger                                                        #
# --------------------------------------------------------------------------- #
log = logging.getLogger("Dreamer")
if not log.handlers:          # only configure once
    h  = logging.StreamHandler()
    fmt = "[%(name)s] %(message)s"
    h.setFormatter(logging.Formatter(fmt))
    log.addHandler(h)
log.setLevel(os.environ.get("DREAMER_LOG_LVL", "INFO").upper())


# --------------------------------------------------------------------------- #
#                           D R E A M E R                                     #
# --------------------------------------------------------------------------- #
class Dreamer:
    # ..................................................................... #
    def __init__(self,
                 observation_shape,
                 discrete_action_bool,
                 action_size,
                 writer,
                 device,
                 config,
                 run_dir):
        t0 = time.time()
        self.run_dir=Path(run_dir) if run_dir else None
        self.device               = device
        self.action_size          = action_size
        self.action_dict = {"left": 0,
                            "right": 1,
                            "forward": 2}
        self.discrete_action_bool = discrete_action_bool
        self.writer               = writer
        self.log_dir = Path(run_dir)
        self.num_total_episode    = 0
        self.global_step = 0
        self.config               = config.parameters.dreamer

        # ---------- networks --------------------------------------------- #
        self.encoder           = Encoder(observation_shape, config).to(device)
        self.decoder           = Decoder(observation_shape, config).to(device)
        self.rssm              = RSSM(action_size, config).to(device)
        self.reward_predictor  = RewardModel(config).to(device)
        #self.actor = Actor( discrete_action_bool,action_size, config).to(device)
        #self.debug_actor = True 
        if self.config.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(device)

        # ---------- replay buffer ---------------------------------------- #
        self.buffer = ReplayBuffer(observation_shape,
                                   action_size,
                                   device,
                                   config)

        # ---------- optimiser ------------------------------------------- #
        self.model_params = (list(self.encoder.parameters())   +
                             list(self.decoder.parameters())   +
                             list(self.rssm.parameters())      +
                             list(self.reward_predictor.parameters()))
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.continue_criterion = nn.BCELoss()

        # ---------- helpers --------------------------------------------- #
        self.dynamic_learning_infos  = DynamicInfos(device)
        self.behavior_learning_infos = DynamicInfos(device)  # not used yet

        # ---------- debug summary ---------------------------------------- #
        def n_params(m): return sum(p.numel() for p in m.parameters())/1e6
        log.info("Initialised Dreamer | encoder %.1f M  rssm %.1f M  total %.1f M",
                 n_params(self.encoder), n_params(self.rssm),
                 sum(p.numel() for p in self.model_params)/1e6)
        log.info("device: %s | buffer capacity: %s",
                 device, f"{self.buffer.capacity:,}")
        log.info("setup done in %.2f s", time.time()-t0)

    # ..................................................................... #
    # placeholder until you plug actor-critic
    def behavior_learning(self, *_):
        return

    # ..................................................................... #
    def train(self, env):
        # seed episodes --------------------------------------------------- #
        if len(self.buffer) < 1:
            self.environment_interaction(env, self.config.seed_episodes)

        ckpt_every = 50                       # iterations
        ckpt_dir   = self.run_dir / "ckpt"
        if ckpt_dir and not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True)

        # resume if a *.pt file already exists ----------------------------------
        latest = sorted(ckpt_dir.glob("iter*.pt"))[-1] if list(ckpt_dir.glob("iter*.pt")) else None
        if latest:
            self._load_ckpt(latest)
            print(f"[Dreamer] resumed from {latest.name}")


        # main loop ------------------------------------------------------- #
        for it in range(1, self.config.train_iterations + 1):
            log.info("⇢ iteration %d/%d", it, self.config.train_iterations)

            # --- model training cycles ---------------------------------- #
            for c in range(1, self.config.collect_interval + 1):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                post, det = self.dynamic_learning(data)
                self.behavior_learning(post, det)

                if c % 10 == 0:
                    log.debug("   collect %d/%d  buffer len: %d",
                              c, self.config.collect_interval, len(self.buffer))
            
                    



            # --- interact with env -------------------------------------- #
            self.environment_interaction(env,
                                         self.config.num_interaction_episodes)
            

            # --- evaluation --------------------------------------------- #
            if it % 10 == 0:
                self.evaluate(env)
            if it % ckpt_every == 0:
                self._save_ckpt(it)

    def _modules(self):
        # everything we want to store
        return dict(
            encoder=self.encoder,
            decoder=self.decoder,
            rssm=self.rssm,
            reward=self.reward_predictor
        )

    def _save_ckpt(self, it):
        state = {
            "iter" : it,
            "rng"  : torch.random.get_rng_state(),
            "opt"  : {k: o.state_dict() for k, o in {
                        "model": self.model_optimizer,
                        "actor": getattr(self, "actor_optimizer", None),
                        "critic":getattr(self,"critic_optimizer",None)}
                    .items() if o},
            "modules": {n: m.state_dict() for n, m in self._modules().items()},
        }
        path = self.run_dir / "ckpt" / f"iter{it:05d}.pt"
        torch.save(state, path)
        print(f"[ckpt] saved → {path.name}")

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for n, m in self._modules().items():
            m.load_state_dict(ckpt["modules"][n])
        for n, o in {"model": self.model_optimizer,
                    "actor": getattr(self, "actor_optimizer", None),
                    "critic":getattr(self,"critic_optimizer", None)}.items():
            if o and n in ckpt["opt"]:
                o.load_state_dict(ckpt["opt"][n])
        torch.random.set_rng_state(ckpt["rng"])
        self.start_iter = ckpt.get("iter", 0) + 1
    # ..................................................................... #
    def evaluate(self, env):
        self.environment_interaction(env,
                                     self.config.num_evaluate,
                                     train=False)

    # ..................................................................... #
    def dynamic_learning(self, data):
        # roll out through time ------------------------------------------ #
        prior, det = self.rssm.recurrent_model_input_init(len(data.action))
        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            det = self.rssm.recurrent_model(prior, data.action[:, t-1], det)
            prior_dist, prior = self.rssm.transition_model(det)
            post_dist, post   = self.rssm.representation_model(
                                    data.embedded_observation[:, t], det)

            self.dynamic_learning_infos.append(
                priors                = prior,
                prior_dist_means      = prior_dist.mean,
                prior_dist_stds       = prior_dist.scale,
                posteriors            = post,
                posterior_dist_means  = post_dist.mean,
                posterior_dist_stds   = post_dist.scale,
                deterministics        = det,
            )
            prior = post

        infos = self.dynamic_learning_infos.get_stacked()
        losses = self._model_update(data, infos)

        log.debug("   dynamic-loss: %.4f  (KL %.3f ‖ recon %.3f ‖ rew %.3f)",
                  losses["model"], losses["kl"], losses["recon"], losses["rew"])
        return infos.posteriors.detach(), infos.deterministics.detach()

    # ..................................................................... #
    def _model_update(self, data, infos):
        # ───────────────────────── reconstruction (image log-likelihood) ──────
        recon_dist  = self.decoder(infos.posteriors, infos.deterministics)
        recon_loss  = recon_dist.log_prob(data.observation[:, 1:])          # <── moved up

        # ───────────────────────── continue flag (optional) ───────────────────
        if self.config.use_continue_flag:
            cont_dist = self.continue_predictor(infos.posteriors,
                                                infos.deterministics)
            cont_loss = self.continue_criterion(cont_dist.probs,
                                                1 - data.done[:, 1:])

        # ───────────────────────── reward model  ───────────────────────────────
        rew_dist  = self.reward_predictor(infos.posteriors, infos.deterministics)
        rew_loss  = rew_dist.log_prob(data.reward[:, 1:])

        # ───────────────────────── KL divergence  ──────────────────────────────
        prior_dist = create_normal_dist(infos.prior_dist_means,
                                        infos.prior_dist_stds, event_shape=1)
        post_dist  = create_normal_dist(infos.posterior_dist_means,
                                        infos.posterior_dist_stds, event_shape=1)

        kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
        kl = torch.max(torch.tensor(self.config.free_nats, device=self.device), kl)

        # ───────────────────────── total model loss  ───────────────────────────
        model_loss = ( self.config.kl_divergence_scale * kl
                    - recon_loss.mean()
                    - rew_loss.mean() )
        if self.config.use_continue_flag:
            model_loss += cont_loss.mean()

        # ───────────────────────── optimisation  ───────────────────────────────
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params,
                                self.config.clip_grad,
                                norm_type=self.config.grad_norm_type)
        self.model_optimizer.step()

        # ───────────────────────── bookkeeping / logging  ──────────────────────
        self.global_step += 1
        if self.writer is not None and self.global_step % 5 == 0:
            recon_img = recon_dist.mean[0, -1].clamp(0, 1)  # (3,H,W)
            self.writer.add_image("reconstruction", recon_img, self.global_step)

        # ─── tiny visual probe every 200 optimisation steps  ---------------
        # save to  runs/<TIMESTAMP>/recon/recon_00012.png  (auto-created dir)
        self._vis_counter = getattr(self, "_vis_counter", 0)
        if getattr(self, "_vis_counter", 0) % 250 == 0:
           
            with torch.no_grad():                       # ← important: no grads!
                outdir = Path(self.run_dir) / "recon"
                outdir.mkdir(parents=True, exist_ok=True)

                # --- 1. ground truth & reconstruction at t = 1 ------------------
                t0_img = data.observation[0, 1]          # (3,64,64)   gt
                recon   = recon_dist.mean[0, 0]          # posterior-recon of the same

                # --- 2. one-step PRIOR (“fantasy”) ------------------------------
                prior_d = infos.deterministics[0, -1]    # deterministic @ last step
                prior_z = infos.priors       [0, -1]     # prior z (no obs)

                fantasy = self.decoder(prior_z[None],    # add batch dim (1, Z)
                                        prior_d[None]    # (1, H)
                                    ).mean.squeeze(0) # → (3,64,64)

                # --- 3. 4-step dreamed rollout under a hand-picked cmd list -----
                cmd = ["forward", "forward", "left", "forward"]
                act_idx = torch.tensor([self.action_dict[a] for a in cmd],
                                    device=self.device)
                onehots = F.one_hot(act_idx, num_classes=self.action_size).float()  # (T,3)

                z = prior_z.unsqueeze(0)     # (1,Z)
                d = prior_d.unsqueeze(0)     # (1,H)
                dreams = []
                for a in onehots:
                    a = a.unsqueeze(0)       # (1,3)
                    d = self.rssm.recurrent_model(z, a, d)
                    _, z = self.rssm.transition_model(d)  # PRIOR z_{t+1}
                    frame = self.decoder(z, d).mean.squeeze(0).cpu()
                    dreams.append(frame)

                # --- save both grids (truth|recon|prior  & dreams) ---------------
                grid1 = torch.stack([t0_img.cpu(), recon.cpu(), fantasy.cpu()])
                save_image(grid1, outdir / f"recon_{self._vis_counter//50:05d}.png",
                        nrow=3, normalize=True)

                grid2 = torch.stack([t0_img.cpu()] + dreams)  # 1+len(cmd) frames
                save_image(grid2, outdir / f"dream_{self._vis_counter//50:05d}.png",
                        nrow=len(cmd)+1, normalize=True)

        self._vis_counter += 1
        
        if self.writer is not None :
            step = self._vis_counter                     # same counter as above
            self.writer.add_scalar("loss/model" ,  model_loss.item(), step)
            self.writer.add_scalar("loss/kl"    ,  kl.item()        , step)
            self.writer.add_scalar("loss/recon" , -recon_loss.mean().item(), step)
            self.writer.add_scalar("loss/reward", -rew_loss.mean().item(), step)

        return dict(model=model_loss.item(),
                    kl=kl.item(),
                    recon=-recon_loss.mean().item(),
                    rew=-rew_loss.mean().item())

    # ..................................................................... #
    @torch.no_grad()
    def environment_interaction(self, env,
                                num_episodes,
                                train: bool = True):

        mode = "train" if train else "eval"
        for epi in range(num_episodes):
            posterior, det = self.rssm.recurrent_model_input_init(1)
            action  = torch.zeros(1, self.action_size, device=self.device)

            obs     = env.reset()
            emb_obs = self.encoder(torch.tensor(obs, dtype=torch.float32,
                                                device=self.device))

            score, steps = 0.0, 0
            done = False
            
            SAFE_STEPS = 30
            while not done:
                det = self.rssm.recurrent_model(posterior, action, det)
                emb_obs = emb_obs.reshape(1, -1)
                _, posterior = self.rssm.representation_model(emb_obs, det)


                front_pos = env.front_pos           # (x, y) tuple
                front_cell = env.grid.get(*front_pos)
                SAFE_STEPS = 30
                # --------------------------------------------------------- choose an action

                if steps < SAFE_STEPS:
                    
                    if isinstance(front_cell, Wall):
                        # there's a wall ahead → turn
                        env_act = random.choice([0, 1])  # 0=left, 1=right
                    else:
                        env_act = 2                      # 2=forward
                else:
                    # after SAFE_STEPS, pure random
                    env_act = random.randrange(0,3)
              
                buffer_act = np.eye(self.action_size, dtype=np.float32)[env_act]
                print(buffer_act)
                next_obs, reward, done, _ = env.step(env_act)
                
                

                if train:
                    self.buffer.add(obs, buffer_act, reward, next_obs, done)

                score  += reward
                steps  += 1
                emb_obs = self.encoder(torch.tensor(next_obs, dtype=torch.float32,
                                                    device=self.device))
                obs     = next_obs

            # ---------- episode finished -------------------------------- #
            if train:
                self.num_total_episode += 1
                self.writer.add_scalar("training score", score,
                                       self.num_total_episode)

            log.info("  episode %d (%s)  score: %.2f  steps: %d",
                     self.num_total_episode if train else epi+1,
                     mode, score, steps)

        # ---------- evaluation summary ---------------------------------- #
        if not train:
            self.writer.add_scalar("test score", score, self.num_total_episode)
            log.info("≈ evaluate mean-score: %.3f over %d episodes",
                     score / num_episodes, num_episodes)

