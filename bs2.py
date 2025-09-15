import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time, math
from scipy import stats

class EnhancedHMMBayesianObserver:
    """
    Enhanced HMM for mode transitions + proper BOCPD.
    Modes: EXPLORE, NAVIGATE, RECOVER, TASK_SOLVING
    """
    def __init__(self, modes: List[str] = None):
        if modes is None:
            modes = ['EXPLORE','NAVIGATE','RECOVER','TASK_SOLVING']
        self.modes       = modes
        self.n_modes     = len(modes)
        self.mode_to_idx = {m:i for i,m in enumerate(modes)}

        # 1) HMM params
        self.transition_matrix = self._init_transition_matrix()
        self.emission_params   = self._init_emission_params()

        # 2) HMM state
        self.mode_beliefs = np.ones(self.n_modes)/self.n_modes
        self.mode_history = deque(maxlen=100)

        # 3) Evidence tracking
        self.last_poses    = deque(maxlen=100)
        self.replay_buffer = deque(maxlen=80)
        self.new_nodes_created = deque(maxlen=50)
        self.plan_progress_history = deque(maxlen=30)
        self.last_significant_progress = time.time()
        self.stagnation_counter = 0

        # 4) attempt counters
        self.exploration_attempts = 0
        self.navigation_attempts  = 0
        self.recovery_attempts    = 0

        # 5) thresholds / params
        self.params = {
            'loop_threshold':20,
            'stagnation_time_threshold':30.0,
            'stagnation_step_threshold':50,
            'max_exploration_cycles':3,
            'max_navigation_cycles':2,
            'max_recovery_cycles':2
        }

        # 6) BOCPD state
        self.max_run_length      = 50
        self.run_length_dist     = np.zeros(self.max_run_length+1)
        self.run_length_dist[0]  = 1.0
        self.hazard_rate         = 1.0/25.0
        self.changepoint_threshold = 0.1

        # 7) history buffers
        self.evidence_buffer  = deque(maxlen=100)
        self.evidence_history = deque(maxlen=self.max_run_length)
        self.last_changepoint = 0

    def _init_transition_matrix(self) -> np.ndarray:
        return np.array([
            [0.75,0.15,0.05,0.05],
            [0.10,0.70,0.15,0.05],
            [0.20,0.20,0.55,0.05],
            [0.05,0.05,0.05,0.85],
        ])

    def _init_emission_params(self) -> Dict:
        return {
            'EXPLORE': {
                'info_gain':{'mean':0.4,'std':0.2},
                'progress': {'mean':0.2,'std':0.15},
                'stagnation':{'mean':0.3,'std':0.2},
                'lost_prob':0.2,
                'loop_prob':0.3,
                'exploration_productivity':{'mean':0.6,'std':0.25}
            },
            'NAVIGATE': {
                'info_gain':{'mean':0.1,'std':0.1},
                'progress': {'mean':0.6,'std':0.2},
                'stagnation':{'mean':0.2,'std':0.15},
                'lost_prob':0.15,
                'loop_prob':0.2,
                'navigation_progress':{'mean':0.7,'std':0.2}
            },
            'RECOVER': {
                'info_gain':{'mean':0.05,'std':0.05},
                'progress': {'mean':0.1,'std':0.1},
                'stagnation':{'mean':0.8,'std':0.2},
                'lost_prob':0.9,
                'loop_prob':0.7,
                'recovery_effectiveness':{'mean':0.4,'std':0.3}
            },
            'TASK_SOLVING': {
                'info_gain':{'mean':0.2,'std':0.1},
                'progress': {'mean':0.5,'std':0.2},
                'stagnation':{'mean':0.1,'std':0.1},
                'lost_prob':0.1,
                'loop_prob':0.1,
                'task_completion':{'mean':0.8,'std':0.2}
            }
        }

    # ————— Evidence metrics —————
    def detect_looping(self) -> float:
        if len(self.last_poses)<self.params['loop_threshold']:
            return 0.0
        counts = defaultdict(int)
        recent = list(self.last_poses)[-self.params['loop_threshold']:]
        for p in recent:
            pose = (round(p[0],1),round(p[1],1)) if len(p)>=2 else p
            counts[pose]+=1
        return min(1.0, max(counts.values())/len(recent))

    def detect_stagnation(self) -> float:
        dt = time.time()-self.last_significant_progress
        return max(min(1.0,dt/self.params['stagnation_time_threshold']),
                   min(1.0,self.stagnation_counter/self.params['stagnation_step_threshold']))

    def assess_exploration_productivity(self) -> float:
        if len(self.new_nodes_created)<5: return 0.5
        recent = list(self.new_nodes_created)[-self.params['exploration_productivity_window']:]
        rate = sum(recent)/len(recent)
        if len(self.last_poses)>=10:
            poses = list(self.last_poses)[-20:]
            unique = len({tuple(round(x,1) for x in p) for p in poses})
            div = unique/len(poses)
        else:
            div = 0.5
        return min(1.0,max(0.0,rate*0.6+div*0.4))

    def assess_navigation_progress(self,fb: Dict) -> float:
        prog=fb.get('plan_progress',0.0)
        self.plan_progress_history.append(prog)
        if len(self.plan_progress_history)<3: return prog
        rp = list(self.plan_progress_history)[-5:]
        trend = np.mean(np.diff(rp)) if len(rp)>1 else 0
        return min(1.0,max(0.0,prog*0.7+max(0,trend)*0.3))

    def assess_recovery_effectiveness(self,fb: Dict) -> float:
        doubt=fb.get('place_doubt_step_count',0)
        if hasattr(self,'last_doubt_count'):
            trend = self.last_doubt_count-doubt
            eff = min(1.0,max(0.0,trend/6.0))
        else:
            eff=0.5
        self.last_doubt_count=doubt
        return eff

    def should_transition_to_task_solving(self) -> float:
        total = (self.exploration_attempts+
                 self.navigation_attempts+
                 self.recovery_attempts)
        if total<3: return 0.1
        if (self.exploration_attempts>=self.params['max_exploration_cycles'] and
            self.navigation_attempts>=self.params['max_navigation_cycles']):
            return 0.8
        if self.exploration_attempts>=2 and len(self.new_nodes_created)>30:
            return 0.4
        return 0.2

    # ————— Emission models —————
    def compute_emission_likelihood(self, evidence: Dict, mode: str) -> float:
        params = self.emission_params[mode]
        ll = 0.0
        # continuous
        for k in ['info_gain','progress','stagnation']:
            if k in evidence and k in params:
                ll += stats.norm.logpdf(evidence[k],
                                        params[k]['mean'],
                                        params[k]['std'])
        # mode-specific
        if mode=='EXPLORE':
            if 'exploration_productivity' in evidence:
                p=evidence['exploration_productivity']
                pm=params['exploration_productivity']
                ll+=stats.norm.logpdf(p,pm['mean'],pm['std'])
            if 'loop_prob' in evidence:
                lp=evidence['loop_prob']
                ll+=stats.norm.logpdf(lp,params['loop_prob'],0.2)
        elif mode=='NAVIGATE':
            if 'navigation_progress' in evidence:
                npv=evidence['navigation_progress']
                pm=params['navigation_progress']
                ll+=stats.norm.logpdf(npv,pm['mean'],pm['std'])
        elif mode=='RECOVER':
            if 'recovery_effectiveness' in evidence:
                rev=evidence['recovery_effectiveness']
                pm=params['recovery_effectiveness']
                ll+=stats.norm.logpdf(rev,pm['mean'],pm['std'])
        elif mode=='TASK_SOLVING':
            if 'task_solving_readiness' in evidence:
                tr=evidence['task_solving_readiness']
                pm=params['task_completion']
                ll+=stats.norm.logpdf(tr,pm['mean'],pm['std'])
        # binary
        if 'agent_lost' in evidence:
            pl=params['lost_prob']
            ll+= (np.log(pl)   if evidence['agent_lost']
                  else np.log(1-pl))
        return np.exp(ll)

    # ————— BOCPD helpers —————
    def compute_run_length_specific_likelihood(self,
                                               evidence: Dict,
                                               run_length: int) -> float:
        if run_length==0:
            return self._compute_prior_likelihood(evidence)
        hist = list(self.evidence_history)
        if len(hist)<run_length:
            relevant = hist
        else:
            relevant = hist[-run_length:]
        if not relevant:
            return self._compute_prior_likelihood(evidence)

        logl = 0.0
        for var in ['info_gain','progress','stagnation',
                    'exploration_productivity',
                    'navigation_progress',
                    'recovery_effectiveness']:
            if var in evidence:
                vals = [e.get(var,0) for e in relevant if var in e]
                if len(vals)>=2:
                    m=np.mean(vals)
                    s=max(np.std(vals),1e-2)
                    s *= (1+1.0/run_length)
                    logl += stats.norm.logpdf(evidence[var],m,s)
                else:
                    # fallback to mode-weighted static
                    logl += np.log(self._compute_mode_weighted_likelihood(evidence,var)+1e-10)
        # combine with static emission
        static_mix = sum(self.mode_beliefs[i]*
                         self.compute_emission_likelihood(evidence,mode)
                         for i,mode in enumerate(self.modes))
        comb = 0.7*np.exp(logl) + 0.3*static_mix
        return max(comb,1e-10)

    def _compute_prior_likelihood(self,evidence:Dict)->float:
        # equally-weighted prior over modes
        return sum(0.25*self.compute_emission_likelihood(evidence,m)
                   for m in self.modes)

    def _compute_mode_weighted_likelihood(self,
                                          evidence:Dict,var:str)->float:
        tot = 0.0
        for i,mode in enumerate(self.modes):
            p=self.mode_beliefs[i]
            prm=self.emission_params[mode].get(var)
            if isinstance(prm,dict):
                tot += p*stats.norm.pdf(evidence[var],
                                        prm['mean'],prm['std'])
        return tot

    # ————— Buffer updates & HMM/BOCPD —————
    def update_mode_attempts(self,current_mode:str):
        if current_mode=='EXPLORE':   self.exploration_attempts+=1
        if current_mode=='NAVIGATE':  self.navigation_attempts+=1
        if current_mode=='RECOVER':   self.recovery_attempts+=1

    def update_evidence_buffers(self,evidence:Dict):
        if 'current_pose' in evidence:
            self.last_poses.append(evidence['current_pose'])
        if 'replay_step' in evidence:
            self.replay_buffer.append(evidence['replay_step'])
        if 'new_nodes' in evidence:
            self.new_nodes_created.append(evidence['new_nodes'])
        if evidence.get('progress',0)>0.1:
            self.last_significant_progress=time.time()
            self.stagnation_counter=0
        else:
            self.stagnation_counter+=1
        # **CRITICAL** push into evidence_history for BOCPD
        self.evidence_history.append(evidence.copy())

    def hmm_forward_step(self,evidence:Dict)->np.ndarray:
        # standard HMM predict+update
        e_likes = np.array([self.compute_emission_likelihood(evidence,m)
                            for m in self.modes])
        # Task-solving bump
        tsp = self.should_transition_to_task_solving()
        if tsp>0.5:
            idx=self.mode_to_idx['TASK_SOLVING']
            e_likes[idx]*=(1.0+tsp)
        pred = self.transition_matrix.T @ self.mode_beliefs
        newb = e_likes * pred
        return (newb/newb.sum()) if newb.sum()>0 else np.ones(self.n_modes)/self.n_modes

    def bocpd_update(self,evidence:Dict)->Tuple[bool,float]:
        pred = np.zeros(self.max_run_length+1)
        for r,mass in enumerate(self.run_length_dist):
            if mass>1e-10:
                pred[r] = self.compute_run_length_specific_likelihood(evidence,r)
        growth = self.run_length_dist[:-1]*(1-self.hazard_rate)*pred[:-1]
        cp_mass= np.sum(self.run_length_dist*self.hazard_rate*pred)
        newdist=np.zeros_like(self.run_length_dist)
        newdist[0]=cp_mass
        newdist[1:]=growth
        if newdist.sum()>0:
            newdist/=newdist.sum()
        self.run_length_dist=newdist
        return (cp_mass>self.changepoint_threshold), cp_mass

    def update(self,evidence:Dict)->Tuple[np.ndarray,bool]:
        self.update_evidence_buffers(evidence)
        enhanced = evidence.copy()
        enhanced.update({
            'loop_prob':self.detect_looping(),
            'stagnation_prob':self.detect_stagnation(),
            'exploration_productivity':self.assess_exploration_productivity(),
            'navigation_progress':self.assess_navigation_progress(evidence),
            'recovery_effectiveness':self.assess_recovery_effectiveness(evidence),
            'task_solving_readiness':self.should_transition_to_task_solving()
        })
        self.evidence_buffer.append(enhanced)
        # track attempts
        cur = self.modes[np.argmax(self.mode_beliefs)]
        self.update_mode_attempts(cur)
        # BOCPD
        cp_flag, cp_prob = self.bocpd_update(enhanced)
        if cp_flag:
            print(f"[BOCPD] Changepoint! p={cp_prob:.3f}")
            self.mode_beliefs = 0.5*self.mode_beliefs + 0.5*(np.ones(self.n_modes)/self.n_modes)
            self.last_changepoint = len(self.evidence_buffer)
        # HMM forward
        self.mode_beliefs = self.hmm_forward_step(enhanced)
        self.mode_history.append((time.time(),self.mode_beliefs.copy()))
        return self.mode_beliefs, cp_flag

    def get_mode_probabilities(self)->Dict[str,float]:
        return {m: self.mode_beliefs[i] for i,m in enumerate(self.modes)}

    def get_diagnostics(self)->Dict:
        recent = self.evidence_buffer[-1] if self.evidence_buffer else {}
        return {
            'mode_attempts':{
                'explore':self.exploration_attempts,
                'nav':self.navigation_attempts,
                'recov':self.recovery_attempts
            },
            'evidence_metrics':{
                'loop':recent.get('loop_prob',0),
                'stagnation':recent.get('stagnation_prob',0),
                'expl_prod':recent.get('exploration_productivity',0),
                'task_ready':recent.get('task_solving_readiness',0)
            },
            'buffer_stats':{
                'poses':len(self.last_poses),
                'replay':len(self.replay_buffer),
                'nodes':sum(self.new_nodes_created)
            },
            'bocpd_stats':{
                'run_length_dist':self.run_length_dist.copy(),
                'most_likely_r':int(np.argmax(self.run_length_dist)),
                'history_size':len(self.evidence_history)
            }
        }


# --------------------------------------------------
# PCFG Builder & Controller (unchanged from your skeleton)
# --------------------------------------------------
class EnhancedMixturePCFGBuilder:
    def __init__(self, key, hmm_observer: EnhancedHMMBayesianObserver):
        self.key          = key
        self.hmm_observer = hmm_observer
        self.memory_graph = key.models_manager.memory_graph
        self.emap         = self.memory_graph.experience_map
        self.submode_beliefs = {
            'EXPLORE': {'ego_allo':0.4,'ego_allo_lookahead':0.3,'short_term_memory':0.2,'astar_directed':0.1},
            'NAVIGATE':{'distant_node':0.4,'unvisited_priority':0.3,'plan_following':0.3},
            'RECOVER': {'solve_doubt':0.6,'backtrack_safe':0.4},
            'TASK_SOLVING':{'goal_directed':0.5,'systematic_search':0.3,'task_completion':0.2}
        }

    def update_submode_beliefs(self,evidence:Dict):
        # your RL-style update from before...
        current = max(self.hmm_observer.get_mode_probabilities().items(),
                      key=lambda x:x[1])[0]
        perf = evidence.get('performance_score',0.5)
        sub = evidence.get('active_submode')
        if sub and sub in self.submode_beliefs[current]:
            lr=0.1; cb=self.submode_beliefs[current][sub]
            if perf>0.6:   self.submode_beliefs[current][sub]=min(0.9,cb+lr*(1-cb))
            elif perf<0.3: self.submode_beliefs[current][sub]=max(0.1,cb-lr*cb)
            # renormalize
            tot=sum(self.submode_beliefs[current].values())
            for k in self.submode_beliefs[current]:
                self.submode_beliefs[current][k]/=tot

    def build_mixture_pcfg(self,use_soft:bool=True)->PCFG:
        mode_probs=self.hmm_observer.get_mode_probabilities()
        if use_soft:
            return self._build_soft_mixture_pcfg(mode_probs)
        else:
            best=max(mode_probs,key=mode_probs.get)
            return self._build_single_mode_pcfg(best)

    def _build_soft_mixture_pcfg(self,mode_probs):
        rules=[]
        s=sum(mode_probs.values())
        if s>0:
            for m,p in mode_probs.items():
                rules.append(f"START -> {m}_ROOT [{p/s:.4f}]")
        else:
            for m in mode_probs: rules.append(f"START -> {m}_ROOT [0.25]")
        rules+=self._build_explore_rules()
        rules+=self._build_navigate_rules()
        rules+=self._build_recover_rules()
        rules+=self._build_task_solving_rules()
        return PCFG.fromstring("\n".join(rules))

    def _build_single_mode_pcfg(self,mode):
        rules=[f"START -> {mode}_ROOT [1.0]"]
        if mode=='EXPLORE':   rules+=self._build_explore_rules()
        if mode=='NAVIGATE':  rules+=self._build_navigate_rules()
        if mode=='RECOVER':   rules+=self._build_recover_rules()
        if mode=='TASK_SOLVING':rules+=self._build_task_solving_rules()
        return PCFG.fromstring("\n".join(rules))

    def _build_explore_rules(self):
        p=self.submode_beliefs['EXPLORE']; s=sum(p.values())
        r=[f"EXPLORE_ROOT -> EXPLORE_{k.upper()} [{(v/s if s>0 else 0.25):.4f}]" for k,v in p.items()]
        r+=[
            "EXPLORE_EGO_ALLO -> 'forward' [0.6] | 'left' [0.2] | 'right' [0.2]",
            "EXPLORE_EGO_ALLO_LOOKAHEAD -> 'forward' 'forward' [0.4] | 'left' 'forward' [0.3] | 'right' 'forward' [0.3]",
            "EXPLORE_SHORT_TERM_MEMORY -> 'scan' [0.3] | 'forward' [0.4] | 'backtrack' [0.3]",
            "EXPLORE_ASTAR_DIRECTED -> 'plan_to_frontier' [1.0]"
        ]
        return r

    def _build_navigate_rules(self):
        p=self.submode_beliefs['NAVIGATE']; s=sum(p.values())
        r=[f"NAVIGATE_ROOT -> NAVIGATE_{k.upper()} [{(v/s if s>0 else 0.33):.4f}]" for k,v in p.items()]
        r+=[
            "NAVIGATE_DISTANT_NODE -> 'goto_distant' [1.0]",
            "NAVIGATE_UNVISITED_PRIORITY -> 'goto_unvisited' [1.0]",
            "NAVIGATE_PLAN_FOLLOWING -> 'follow_plan' [0.8] | 'replan' [0.2]"
        ]
        return r

    def _build_recover_rules(self):
        p=self.submode_beliefs['RECOVER']; s=sum(p.values())
        r=[f"RECOVER_ROOT -> RECOVER_{k.upper()} [{(v/s if s>0 else 0.5):.4f}]" for k,v in p.items()]
        r+=[
            "RECOVER_SOLVE_DOUBT -> 'scan' [0.4] | 'relocalize' [0.6]",
            "RECOVER_BACKTRACK_SAFE -> 'backtrack' [0.7] | 'return_to_known' [0.3]"
        ]
        return r

    def _build_task_solving_rules(self):
        p=self.submode_beliefs['TASK_SOLVING']; s=sum(p.values())
        r=[f"TASK_SOLVING_ROOT -> TASK_{k.upper()} [{(v/s if s>0 else 0.33):.4f}]" for k,v in p.items()]
        r+=[
            "TASK_GOAL_DIRECTED -> 'execute_task_plan' [0.8] | 'refine_task_plan' [0.2]",
            "TASK_SYSTEMATIC_SEARCH -> 'systematic_exploration' [0.6] | 'check_all_rooms' [0.4]",
            "TASK_TASK_COMPLETION -> 'complete_objective' [0.9] | 'verify_completion' [0.1]"
        ]
        return r


class EnhancedHybridBayesianController:
    def __init__(self, key, buffer_size:int=50):
        self.key           = key
        self.hmm_observer  = EnhancedHMMBayesianObserver()
        self.pcfg_builder  = EnhancedMixturePCFGBuilder(key,self.hmm_observer)
        self.performance_buffer = deque(maxlen=buffer_size)
        self.last_evidence     = {}
        self.use_soft_mixture  = True
        self.adaptation_enabled = True

    def extract_enhanced_evidence(self,agent_state,env_state,perf):
        e = {
            'timestamp':time.time(),
            'performance_score':perf.get('reward',0.5),
            'active_submode':agent_state.get('active_submode'),
            'info_gain':perf.get('info_gain',0.0),
            'progress':perf.get('plan_progress',0.0),
            'agent_lost':agent_state.get('place_doubt_step_count',0)>6,
            'place_doubt_step_count':agent_state.get('place_doubt_step_count',0),
            'new_nodes':env_state.get('new_nodes',0),
            'nodes_created':env_state.get('nodes_created_total',0),
            'plan_progress':perf.get('plan_progress',0.0),
            'has_navigation_goal':agent_state.get('has_navigation_goal',False),
            'path_blocked':env_state.get('path_blocked',False),
            'task_defined':agent_state.get('task_defined',False),
            'exploration_completeness':env_state.get('exploration_completeness',0.0)
        }
        return e

    def update(self,agent_state,env_state,perf):
        evidence = self.extract_enhanced_evidence(agent_state,env_state,perf)
        self.last_evidence = evidence
        self.performance_buffer.append(perf.get('reward',0.0))

        # 1) HMM+BOCPD
        beliefs, cp = self.hmm_observer.update(evidence)
        # 2) Submode adaptation
        if self.adaptation_enabled:
            self.pcfg_builder.update_submode_beliefs(evidence)
        # 3) PCFG
        pcfg = self.pcfg_builder.build_mixture_pcfg(self.use_soft_mixture)
        # 4) Diagnostics
        diag = {
            'mode_beliefs':self.hmm_observer.get_mode_probabilities(),
            'changepoint': cp,
            'last_evidence': evidence,
            **self.hmm_observer.get_diagnostics()
        }
        return pcfg, diag

    def get_current_strategy(self)->Tuple[str,float]:
        mp = self.hmm_observer.get_mode_probabilities()
        best,conf = max(mp.items(),key=lambda x:x[1])
        return best,conf

    def toggle_mixture_mode(self,use_soft:Optional[bool]=None):
        if use_soft is not None: self.use_soft_mixture = use_soft
        else:                   self.use_soft_mixture = not self.use_soft_mixture
        print(f"[Controller] {'soft' if self.use_soft_mixture else 'hard'} mixture")

    def print_status(self):
        best,conf = self.get_current_strategy()
        stats = self.hmm_observer.get_diagnostics()['buffer_stats']
        print(f"Dominant: {best} ({conf:.2f}); poses={stats['poses_tracked']}, replay={stats['replay_buffer_size']}")
