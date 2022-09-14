from typing import Dict, List, Tuple
from qc_utils import *

PI = np.pi

class TransmonSimulator(object):
    '''
    - Simulation in SI units (Hz, sec, etc.)
    '''
    def __init__(self, params):
        num_transmon = self.num_transmon = self.L= params['num_transmon']
        num_level = self.num_level = params['num_level']
        self.dt = params['dt']
        self.qubit_indices,_ = qubit_subspace(self.num_level,self.num_transmon)
        
        # Set up raising, lowering, and occupancy operators
        b = np.diag(np.sqrt(np.arange(1, num_level)), 1)
        n = np.diag(np.arange(num_level))
        Id = np.eye(num_level)
        self.bs, self.ns = [], []
        for i  in range(num_transmon):
            b_list, n_list = [Id]*num_transmon, [Id]*num_transmon
            b_list[i], n_list[i] = b, n
            self.bs.append(tensor(b_list))
            self.ns.append(tensor(n_list))
        
        # System control + control noise
        self.ctrl = params['ctrl']
        self.ctrl_noise = params['ctrl_noise']
        self.current_ctrl = {}
        
        if self.ctrl_noise:  
            if self.ctrl_noise > 1:
                print(f'-   Noisy control with variance: {self.ctrl_noise/MHz} MHZ')
            else:
                print(f'-   Noisy control with variance: {self.ctrl_noise*100}%')
        else:
            print('-   Noiseless control')
            
        if params['sim_frame_rotation']:
            raise NotImplementedError('Check implementation in quspin simulator')
        else:
            self.dressed_to_sim = np.eye(num_level**num_transmon)
            
    def get_expmap(self,pulse,t_step) -> List[np.ndarray]:     
        
        # Perturb control params if specified
        if self.ctrl_noise:
            for param in self.ctrl.keys():
                noise_var = self.ctrl_noise if self.ctrl_noise >= 1 else self.ctrl_noise*abs(self.ctrl[param])
                self.current_ctrl =  np.random.normal(self.ctrl[param],noise_var)
        else:
            self.current_ctrl = self.ctrl.copy()
            
        # For readability
        b, n = self.bs, self.ns
        coupling = self.current_ctrl['coupling']
        anharm = self.current_ctrl['anharm']
        detune = self.current_ctrl['detune']
        drive = self.current_ctrl['drive']
        
        # System hamiltonian        
        H_sys = 0
        for i in range(self.num_transmon-1):
            H_sys += coupling[i]*( b[i].T @ b[i+1] + b[i] @ b[i+1].T )
        for i in range(self.num_transmon):
            H_sys += detune[i]*n[i] + anharm[i]/2*( n[i] @ n[i] - n[i])
            
        # Control hamiltonian
        phase = np.exp(1j*self.dt*(t_step+1)*detune)
        H_ctrl = 0
        for i in range(self.num_transmon):
            if self.num_transmon == 1:
                term = b[i]/2 * drive[i] * phase[i] * (pulse[i] + 1j*pulse[i+1])
            elif self.num_transmon == 2:
                term = b[i]/2 * ( drive[2*i]  *phase[i]      *(pulse[4*i]   + 1j*pulse[4*i+1]) + 
                                  drive[2*i+1]*phase[(i+1)%2]*(pulse[4*i+2] + 1j*pulse[4*i+3]) )
            else:
                raise NotImplementedError(f'Drive hamiltonian not implemented for {self.num_transmon} transmons! What are the channels?')
            H_ctrl += term + term.conj().T
            
        expmap = expm(-1j*(H_sys+H_ctrl)*self.dt)
        return expmap