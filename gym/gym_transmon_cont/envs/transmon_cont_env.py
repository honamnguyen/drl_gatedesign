'''transmon_cont_env.py'''
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from qtool.simulators import *
from qtool.utility import *

class ContinuousTransmonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kw):
        super(ContinuousTransmonEnv, self).__init__()
        print('---Initiating Transmon Environment with Continuous Action---')
        self.init_state = kw['init_state']
        self.init_ket = None if kw['init_ket'] is None else kw['init_ket']
        self.rl_state = kw['rl_state']
        self.pca_order = kw['pca_order']
        # self.sim_name = kw['sim_name']
        self.sim_params = kw['qsim_params']
        self.step_params = kw['step_params']
        _,self.qubit_indices, self.qubit_proj = qubit_subspace(kw['qsim_params']['num_level'],kw['qsim_params']['num_transmon'])
        self.channels = kw['channels']
        self.sub_action_scale = kw['sub_action_scale']
        self.end_amp_window = kw['end_amp_window']
        self.tnow = 0
                
        self.action_space = spaces.Box(-1,1,[len(self.channels)])
        self.update_simulator(kw['sim_name'])           
        
        # rotating target unitary to simulation frame
        S = self.sim.dressed_to_sim
        self.target_unitary = S@kw['target_unitary']@S.T.conj()
        self.target_state = tensor([self.target_unitary,self.target_unitary.conj()]) @ kw['init_state']
        
        self.correction_angle = True if self.sim.L == 2 else False
        self.state = self.init_state
        if self.init_ket is not None:
            self.ket = self.init_ket
            self.target_ket = self.target_unitary@self.init_ket
        self.observation_space = spaces.Box(-2,2,[len(self.reset())])
        
        # if kw['seed']:
        #     seed = kw['seed']
        #     num_workers = kw['num_workers']
        #     worker_index = kw['worker_index']
        #     vector_index = kw['vector_index']
        #     total_seed = seed + num_workers + worker_index + vector_index
        #     self.seed(total_seed)
        #     print(f'-   Seed: {total_seed} = {seed} + {num_workers} + {worker_index} + {vector_index}')
        # else:
        #     print('-   Random seed')
            
    def evolve(self, input_action, evolve_method='exact'):
        if self.sub_action_scale is not None:
            action = input_action*self.sub_action_scale + self.prev_action
            action[action>1] = 1
            action[action<-1] = -1
        else:
            action = input_action
            
        assert abs(action).max()<=1
        if evolve_method == 'exact':
            if len(self.channels) == self.sim.num_channel:
                expmap = self.sim.get_expmap(action,self.tnow)
            else:
                expanded_action = np.zeros(self.sim.num_channel)
                expanded_action[self.channels] = action
                expmap = self.sim.get_expmap(expanded_action,self.tnow)
            expmap_super = tensor([expmap,expmap.conj()])

        else:
            raise NotImplementedError
            
        self.state = expmap_super@self.state
        self.map_super = expmap_super@self.map_super
        self.leakage = compute_leakage(self)
        if 'ket' in self.rl_state:
            self.ket = expmap@self.ket
            self.map = expmap@self.map
        self.tnow += 1
        return action
    
    def step(self, action):
        
        evolve_method,num_seg,fid_threshold,neg_reward_scale,reward_scheme,reward_type,method = self.step_params.values()
        self.prev_action = self.evolve(action,evolve_method)
        
        current_map = self.map if 'ket' in self.rl_state else self.map_super
        M_qubit,theta = projected_overlap(current_map,
                                          self.target_unitary,
                                          self.qubit_indices,
                                          self.correction_angle)
        self.avg_fid = avg_fid = average_over_pure_states(M_qubit)

        if 'local-fidelity-difference' in reward_scheme:
            if reward_type == 'worst':
                _,worst_fid = worst_fidelity(self,method,overlap_args=(M_qubit,theta))
                fid = worst_fid
            elif reward_type == 'average':
                fid = avg_fid
            if 'nli' in  reward_scheme:
                reward = NLI(fid) - NLI(self.fid)
            else:
                reward = fid - self.fid
            done = True if (fid > fid_threshold or self.tnow>=num_seg) else False
            self.fid = fid
            
            
        elif 'only-final-step' in reward_scheme:
            self.fid = fid = worst_fidelity(self,method,overlap_args=(M_qubit,theta))[1] if (avg_fid > fid_threshold or self.tnow>=num_seg) else avg_fid
            done = True if (fid > fid_threshold or self.tnow>=num_seg) else False
            if 'nli' in reward_scheme:
                reward = NLI(fid) if done else -1/num_seg*neg_reward_scale
            else:
                reward = fid if done else -1/num_seg*neg_reward_scale
        
        # enforce small amplitude at the end
        if self.end_amp_window:
            if done and abs(self.prev_action).max()>self.end_amp_window:
                reward = 0
            
        state = np.hstack([self.get_state(self.rl_state).flatten().view(np.float64),
                           self.prev_action])
        return state, reward, done, {}
    
    def reset(self):
        self.state = self.init_state
        if self.init_ket is not None:
            self.ket = self.init_ket
        N = self.sim.num_level**self.sim.L
        self.map = np.eye(N)
        self.map_super = np.eye(N**2)
        self.tnow = 0
        self.prev_action = np.zeros(len(self.channels))
        
        # get fidelity of the unevolved basis states
        _,_,_,_,reward_scheme,reward_type,method = self.step_params.values()
        current_map = self.map if 'ket' in self.rl_state else self.map_super
        M_qubit,theta = projected_overlap(current_map,
                                          self.target_unitary,
                                          self.qubit_indices,
                                          self.correction_angle)
        self.avg_fid = avg_fid = average_over_pure_states(M_qubit)
        self.leakage = [0,0]
        if 'local-fidelity-difference' in self.step_params['reward_scheme']:
            if self.step_params['reward_type'] == 'worst':
                _,worst_fid = worst_fidelity(self,method,overlap_args=(M_qubit,theta))
                fid = worst_fid
            elif self.step_params['reward_type'] == 'average':
                fid = avg_fid
            self.fid = fid
        else:
            self.fid = None
        
        state = np.hstack([self.get_state(self.rl_state).flatten().view(np.float64),
                           np.zeros(len(self.channels))])
        return state
    
    def render(self, mode='human'):
        return 0
    
    def get_state(self,rl_state='full_dm'):
        if rl_state == 'full_dm':
            return self.state
        elif rl_state == 'pca_dm':
            return pca(self.state,self.sim.num_level,self.sim.L,order=self.pca_order,test=False)
        elif rl_state == 'ket':
            return self.ket
        
    def update_init_target_state(self,init_state,target_state):
        self.init_state = init_state
        self.target_state = target_state
        
    def update_simulator(self,sim_name):
        if sim_name == 'quspin':
            self.sim = quspin_duffing_simulator(self.sim_params)            
        elif sim_name == 'TransmonDuffingSimulator':
            self.sim = TransmonDuffingSimulator(self.sim_params)
        self.sim_name = sim_name