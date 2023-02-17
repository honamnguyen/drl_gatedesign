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
                
        self.action_space = spaces.Box(-1,1,[2*len(self.channels)])
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
        
        # additional information for rl_state
        self.rl_state_len = len(self.rl_state.split('_'))
        if self.rl_state_len == 3:
            # eg: ket_anharm_0
            _, self.param, self.index = self.rl_state.split('_')
        elif self.rl_state_len == 2 and 'ctrl' not in self.rl_state:
            # eg: ket_anharm
            _, self.param = self.rl_state.split('_')
            
        # 'concat' observation space for old runs before changing to dict
        if 'concat' in self.rl_state:
            self.observation_space = spaces.Box(-2,2,[len(self.reset())])
        else:
            state = self.reset()
            obs_space_dict = {
                'quantum_state': spaces.Box(-1,1,[len(state['quantum_state'])]),
                'prev_action': spaces.Box(-1,1,[len(state['prev_action'])]),
            }
            if 'ctrl' in self.rl_state:
                obs_space_dict['drive'] = spaces.Box(-1,1,[len(state['drive'])])
                obs_space_dict['detune'] = spaces.Box(-1,1,[len(state['detune'])])
                obs_space_dict['anharm'] = spaces.Box(-1,1,[len(state['anharm'])])
                obs_space_dict['coupling'] = spaces.Box(-1,1,[len(state['coupling'])])
                
            elif self.rl_state_len == 3:
                _, self.param, self.index = self.rl_state.split('_')
                obs_space_dict[self.param+self.index] = spaces.Box(-1,1,[1])
                
            elif self.rl_state_len == 2:
                _, self.param = self.rl_state.split('_')
                obs_space_dict[self.param] = spaces.Box(-1,1,[len(state[self.param])])

            self.observation_space = spaces.Dict(obs_space_dict)
        print(f'\n{self.observation_space}\n')
            
    def evolve(self, input_action, evolve_method): #='exact'):
        if self.sub_action_scale is not None:
            action = input_action*self.sub_action_scale + self.prev_action
            action[action>1] = 1
            action[action<-1] = -1
        else:
            action = input_action
            
        assert abs(action).max()<=1
        # if evolve_method == 'exact':
        complex_action = action.view(np.complex128)
        if len(self.channels) == self.sim.num_channel:
            expmap = self.sim.get_expmap(complex_action,self.tnow,evolve_method)
        else:
            expanded_action = np.zeros(self.sim.num_channel, dtype=np.complex128)
            expanded_action[self.channels] = complex_action
            expmap = self.sim.get_expmap(expanded_action,self.tnow,evolve_method)
        expmap_super = tensor([expmap,expmap.conj()])

        # else:
        #     raise NotImplementedError
            
        self.state = expmap_super@self.init_state #@self.state
        self.map_super = expmap_super #@self.map_super
        self.leakage = compute_leakage(self)
        if 'ket' in self.rl_state:
            self.ket = expmap@self.init_ket #@self.ket
            self.map = expmap #@self.map
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
#             print(avg_fid > fid_threshold or self.tnow>=num_seg,avg_fid,self.tnow)
            # if avg_fid crosses threshold, check if worst_fid also crosses threshold
            if reward_type == 'worst':
                if avg_fid > fid_threshold or self.tnow>=num_seg:
                    _,worst_fid = worst_fidelity(self,method,overlap_args=(M_qubit,theta))
                    fid = worst_fid
                else:
                    fid = avg_fid
            elif reward_type == 'average':
                fid = avg_fid                
#             self.fid = fid = worst_fidelity(self,method,overlap_args=(M_qubit,theta))[1] if (avg_fid > fid_threshold or self.tnow>=num_seg) else avg_fid
            done = True if (fid > fid_threshold or self.tnow>=num_seg) else False
            if 'nli' in reward_scheme:
                reward = NLI(fid) if done else -1/num_seg*neg_reward_scale
            else:
                reward = fid if done else -1/num_seg*neg_reward_scale
            self.fid = fid if done else None
        
        # enforce small amplitude at the end
        if self.end_amp_window:
            if done and abs(self.prev_action).max()>self.end_amp_window:
                reward = 0
            
        # Dict observation space 
        if 'concat' in self.rl_state:
            state = np.hstack([self.get_state(self.rl_state).flatten().view(np.float64),self.prev_action])
        else:
            state = self.get_state_dict()

        return state, reward, done, {}
    
    def reset(self):
        self.sim.reset()
        self.state = self.init_state
        if self.init_ket is not None:
            self.ket = self.init_ket
        N = self.sim.num_level**self.sim.L
        self.map = np.eye(N)
        self.map_super = np.eye(N**2)
        self.tnow = 0
        self.prev_action = np.zeros(2*len(self.channels))
        
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
        
        # Dict observation space 
        if 'concat' in self.rl_state:
            state = np.hstack([self.get_state(self.rl_state),self.prev_action])
        else:
            state = self.get_state_dict()
    
        return state
    
    def render(self, mode='human'):
        return 0
    
    def _add_ctrl_to_state(self, param, state):
        if param == 'detune':
            nonzero_ind = abs(self.sim.ctrl[param])>1e-10
            state[param] = self.sim.current_ctrl[param][nonzero_ind]/self.sim.ctrl[param][nonzero_ind] - 1
        else:
            state[param] = self.sim.current_ctrl[param]/self.sim.ctrl[param] - 1  
        return state
    
    def get_state_dict(self):
        state = {
            'quantum_state': self.get_quantum_state(self.rl_state),
            'prev_action': self.prev_action,
        }
        if 'ctrl' in self.rl_state:
            for param in ['drive','detune','anharm','coupling']:
                state = self._add_ctrl_to_state(param, state)
            # for key,val in self.sim.current_ctrl.items():
            #     if key == 'freq':
            #         continue
            #     elif key == 'detune':
            #         nonzero_ind = abs(self.sim.ctrl[key])>1e-10
            #         state[key] = val[nonzero_ind]/self.sim.ctrl[key][nonzero_ind] - 1
            #     else:
            #         state[key] = val/self.sim.ctrl[key] - 1
        elif self.rl_state_len == 3:
            current = self.sim.current_ctrl[self.param][[int(self.index)]]
            fiducial = self.sim.ctrl[self.param][[int(self.index)]]
            state[self.param+self.index] = current/fiducial - 1 
        elif self.rl_state_len == 2:
            state = self._add_ctrl_to_state(self.param, state)

        return state
        
    def get_quantum_state(self, rl_state='full_dm'):
        if 'full_dm' in rl_state:
            state = self.state
        elif 'pca_dm' in rl_state:
            state = pca(self.state,self.sim.num_level,self.sim.L,order=self.pca_order,test=False)
        elif 'ket' in rl_state:
            state = self.ket
        return state.flatten().view(np.float64)
    
    def get_state(self,rl_state='full_dm'):
        if 'full_dm' in rl_state:
            state = self.state
        elif 'pca_dm' in rl_state:
            state = pca(self.state,self.sim.num_level,self.sim.L,order=self.pca_order,test=False)
        elif 'ket' in rl_state:
            state = self.ket
        state = state.flatten().view(np.float64)
        
        return state
        
    def update_init_target_state(self,init_state,target_state):
        self.init_state = init_state
        self.target_state = target_state
        
    def update_simulator(self,sim_name):
        if sim_name == 'quspin':
            self.sim = quspin_duffing_simulator(self.sim_params)            
        elif sim_name == 'TransmonDuffingSimulator':
            self.sim = TransmonDuffingSimulator(self.sim_params)
        self.sim_name = sim_name