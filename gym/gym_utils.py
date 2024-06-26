import numpy as np
from qtool.utility import *

def initialize_transmon_env_sqrtZX():
    num_transmon = 2
    num_level = 3
    sim_frame_rotation = False
    anharm   = 2*np.pi * np.array([-319.7,-320.2])*MHz
    drive    = 2*np.pi * np.array([30,300,30,300])*MHz
    detune   = 2*np.pi * np.array([115,0])*MHz
    coupling = 2*np.pi * np.array([args.coupling])*MHz

def initialize_transmon_env(sim_name, num_transmon, num_level, sim_frame_rotation,
                            ctrl, ctrl_noise, ctrl_noise_param, ctrl_update_freq,
                            num_seg, dt, target_gate,
                            rl_state, pca_order,
                            reward_type, reward_scheme, fid_threshold, worstfid_method,
                            channels, sub_action_scale, end_amp_window, evolve_method):
    kw = {}
    kw['qsim_params'] = {'num_transmon': num_transmon,
                         'num_level': num_level,
                         'sim_frame_rotation':sim_frame_rotation,
                         'dt':dt,
                         'ctrl': ctrl,
                         'ctrl_noise': ctrl_noise,
                         'ctrl_noise_param': ctrl_noise_param,
                         'ctrl_update_freq': ctrl_update_freq,
                         }
    kw['sim_name'] = sim_name
    kw['rl_state'] = rl_state
    kw['pca_order'] = pca_order

    # Track evolution of basis elements
    dm_basis = get_reduced_basis(num_level,num_transmon)
    _,qubit_indices,qubit_proj = qubit_subspace(num_level,num_transmon)
    basis_size = len(dm_basis)

    gate = common_gate(target_gate)
    if num_level == 2:
        target_unitary = gate
    elif num_level == 3:
        target_unitary = np.eye(num_level**num_transmon,dtype=np.complex128)
        target_unitary[qubit_indices] = gate
    
    kw['init_state'] = (dm_basis).reshape([basis_size,-1]).T
    kw['target_unitary'] = target_unitary
    kw['qubit_indices'] = qubit_indices
    # kw['qubit_proj'] = qubit_proj
    
    # Keep ket
    kw['init_ket'] = get_ket_basis(num_level,num_transmon).T if 'ket' in rl_state else None
    
    # Continuous actions
    kw['channels'] = channels
    kw['sub_action_scale'] = sub_action_scale
    kw['end_amp_window'] = end_amp_window
    
    kw['step_params'] = {'evolve_method':evolve_method,
                         'max_step':num_seg,
                         'fid_threshold':fid_threshold,
                         'neg_reward_scale':0,
                         'reward_scheme':reward_scheme,
                         'reward_type': reward_type,
                         'worstfid_method':worstfid_method}
    return kw

MHz = 1e6
nanosec = 1e-9