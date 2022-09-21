import numpy as np
from typing import Dict, List, Tuple
from scipy.linalg import expm
from qtool.utility import *

PI = np.pi

class TransmonDuffingSimulator(object):
    '''
    - Simulation in SI units (Hz, sec, etc.)
    '''
    def __init__(self, params):
        num_transmon = self.num_transmon = self.L= params['num_transmon']
        num_level = self.num_level = params['num_level']
        if self.num_transmon == 1:
            self.num_channel = 2
        elif self.num_transmon == 2:
            self.num_channel = 8
        else:
            raise NotImplementedError(f'Drive hamiltonian not implemented for {self.num_transmon} transmons! What are the channels?')

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
    
    def pulse_average_fidelity(self, full_pulse, U_target, qubit_indices,
                               initial_Z_thetas=[], correct_Z_after=True) -> float:
        U = np.eye(len(U_target))        
        for t_step,pulse in enumerate(full_pulse):
            U = self.get_expmap(pulse,t_step)@U
            
        if len(initial_Z_thetas) > 0:
            initial_Z = Zgate_on_all(initial_Z_thetas,self.num_level)
            M,theta = projected_overlap(U@initial_Z,U_target,qubit_indices,correct_Z_after)
        else:
            M,theta = projected_overlap(U,U_target,qubit_indices,correct_Z_after)
            
        return average_over_pure_states(M)
    
    def pulse_average_fidelities(self, full_pulse, U_target, qubit_indices,
                                 initial_Z_thetas=[], correct_Z_after=True):
        U = np.eye(len(U_target))
        avg_fids = []
        if len(initial_Z_thetas) > 0:
            initial_Z = Zgate_on_all(initial_Z_thetas,self.num_level)
            
        for t_step,pulse in enumerate(full_pulse):
            U = self.get_expmap(pulse,t_step)@U
            if len(initial_Z_thetas) > 0:
                M,theta = projected_overlap(U@initial_Z,U_target,qubit_indices,correct_Z_after)
            else:
                M,theta = projected_overlap(U,U_target,qubit_indices,correct_Z_after)
            avg_fids.append(average_over_pure_states(M))
        return np.array(avg_fids),theta,U
    
#############################################################################
############################## OLD QUSPIN SIM ###############################
#############################################################################
from quspin.operators import hamiltonian 
from quspin.basis import boson_basis_1d

class quspin_duffing_simulator():
    '''
    Input: params
        - ctrl: control parameters in MHz
        - dt: time segment size in us 
        - error:
            + If within [0,1], treat as percentage of parameter's amplitude
            + If larger than 1, treat as an absolute uncertainty in Hz
    '''
    def __init__(self, params):
        L = self.L = params['num_transmon']
        self.num_level = params['num_level']
        self.basis = boson_basis_1d(L=self.L, sps=self.num_level)

        self.ctrl = params['ctrl']
        self.ctrl_noise = params['ctrl_noise']
        self.current_ctrl = {}
        dt = self.dt = params['dt']
        self.coupling_list = [[i,j] for i in range(self.L) for j in range(i+1,self.L)]
        
        # fixed error in Hz
        if self.ctrl_noise:  
            if self.ctrl_noise > 1:
                print(f'-   Noisy control with variance: {self.ctrl_noise/MHz} MHZ')
            else:
                print(f'-   Noisy control with variance: {self.ctrl_noise*100}%')
        else:
            print('-   Noiseless control')

            
        detune_coeff   = [[2*PI*coeff*dt,i]       for i,coeff in zip(range(L),self.ctrl['detune'])]
        anharm_coeff   = [[  PI*coeff*dt,i,i,i,i] for i,coeff in zip(range(L),self.ctrl['anharm'])]
        coupling_coeff = [[2*PI*coeff*dt,*ij]     for ij,coeff in zip(self.coupling_list,self.ctrl['coupling'])]
            
        static = [['n',detune_coeff],
                  ['++--',anharm_coeff],
                  ['+-',coupling_coeff],
                  ['-+',coupling_coeff]]
        
        if params['sim_frame_rotation']:
            ham_no_drive = np.flip(hamiltonian(static,[],dtype=np.complex128,basis=self.basis,
                                    check_symm=False,check_herm=False).toarray())
            self.dressed_to_sim = block_diag_transf_mat(ham_no_drive,self.num_level)
        else:
            self.dressed_to_sim = np.eye(self.num_level**self.L)
        # _,self.dressed_to_sim = np.linalg.eigh(ham_no_drive.toarray())

    def get_expmap(self, pulse, t_step):
        expmap, expmap_super = [],[]
        L, dt = self.L, self.dt
        # Perturb control params if specified
        if self.ctrl_noise:
            for param in self.ctrl.keys():
                noise_var = self.ctrl_noise/MHz if self.ctrl_noise >= 1 else self.ctrl_noise*abs(self.ctrl[param])
                self.current_ctrl =  np.random.normal(self.ctrl[param],noise_var)
        else:
            self.current_ctrl = self.ctrl.copy()
            
        self.detune_coeff   = [[2*PI*coeff*dt,i]       for i,coeff in zip(range(L),self.current_ctrl['detune'])]
        self.anharm_coeff   = [[  PI*coeff*dt,i,i,i,i] for i,coeff in zip(range(L),self.current_ctrl['anharm'])]
        self.coupling_coeff = [[2*PI*coeff*dt,*ij]     for ij,coeff in zip(self.coupling_list,self.current_ctrl['coupling'])]
            
        # for pulse in actions:
        drive_coeff_p,drive_coeff_m = [],[]
        for i in range(L):
            # D(t)b + D(t).conj()b^dag
            drive_coeff_p.append([(pulse[2*i]-1j*pulse[2*i+1]) * 2*PI*dt*self.current_ctrl['drive'][i],i])
            drive_coeff_m.append([(pulse[2*i]+1j*pulse[2*i+1]) * 2*PI*dt*self.current_ctrl['drive'][i],i])
        static = [['+',drive_coeff_p],
                  ['-',drive_coeff_m],
                  ['n',self.detune_coeff],
                  ['++--',self.anharm_coeff],
                  ['+-',self.coupling_coeff],
                  ['-+',self.coupling_coeff]]
        H = np.flip(hamiltonian(static,[],dtype=np.complex128,basis=self.basis,
                                check_symm=False,check_herm=False).toarray()) 
        expmap = expm(-1j*H)
        return expmap      
