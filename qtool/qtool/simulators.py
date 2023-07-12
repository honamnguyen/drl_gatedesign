# copy file here to modify
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple
from scipy.linalg import expm
from qtool.utility import *

import qutip as qt

PI = np.pi

class TransmonDuffingSimulator(object):
    '''
    - Simulation in SI units (Hz, sec, etc.)
    '''
    def __init__(self, params):
        num_transmon = self.num_transmon = self.L= params['num_transmon']
        num_level = self.num_level = params['num_level']
        self.dim = num_level**num_transmon
        if num_transmon == 1:
            self.num_channel = 1
        elif num_transmon == 2:
            self.num_channel = 4
        else:
            raise NotImplementedError(f'Drive hamiltonian not implemented for {num_transmon} transmons! What are the channels?')

        self.dt = params['dt']
        # self.qubit_indices,_ = qubit_subspace(self.num_level,self.num_transmon)
        
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
      
        self.reset_ctrl(params)   
        
        if params['sim_frame_rotation']:
            raise NotImplementedError('Check implementation in quspin simulator')
        else:
            self.dressed_to_sim = np.eye(num_level**num_transmon)     
            
    def reset_ctrl(self, params, verbose=True):
        # System control + control noise
        self.ctrl = params['ctrl']
        self.ctrl_noise = params['ctrl_noise']
        self.ctrl_noise_param = self.ctrl.keys() if params['ctrl_noise_param'] == 'all' else params['ctrl_noise_param'].split('_')
        self.ctrl_update_freq = params['ctrl_update_freq']
        assert self.ctrl_update_freq in ['everystep','everyepisode']
        self.current_ctrl = deepcopy(self.ctrl)
        
        if verbose:
            if self.ctrl_noise:
                assert self.ctrl_noise <= 1
                print(f"-   {self.ctrl_update_freq}: Noisy control for '{params['ctrl_noise_param']}' with variance {self.ctrl_noise*100}%")
                # if self.ctrl_noise > 1:
                #     print(f'-   {self.ctrl_update_freq}: Noisy control with variance: {self.ctrl_noise/MHz} MHZ')
                # else:
            else:
                print('-   Noiseless control')
        self.calculate_H_sys(self.ctrl) # Save system hamiltonian for no noise case
    
    def calculate_H_sys(self, ctrl):
        b, n = self.bs, self.ns
        coupling = ctrl['coupling']
        anharm = ctrl['anharm']
        detune = ctrl['detune']
        
        # Save system hamiltonian for no noise case
        self.H_sys = 0j
        for i in range(self.num_transmon-1):
            self.H_sys += coupling[i]*( b[i].T @ b[i+1] + b[i] @ b[i+1].T )
        for i in range(self.num_transmon):
            self.H_sys += detune[i]*n[i] + anharm[i]/2*( n[i] @ n[i] - n[i])

    def ctrl_update(self):
        '''
        Update current control parameters with noise: 
        
        Output: current_ctrl, H_sys
        '''
        current_ctrl = deepcopy(self.ctrl)
        self.current_variation = {}
        # support updating a type of param or a single param at a time
        for param in self.ctrl_noise_param:
            if param[-1].isdigit():
                ind = int(param[-1])
                self.current_variation[param] = np.random.uniform(-self.ctrl_noise,self.ctrl_noise,1)
                current_ctrl[param[:-1]][ind] = self.ctrl[param[:-1]][ind]*(1 + self.current_variation[param][0])
                
                # mean = self.ctrl[param[:-1]][int(param[-1])]
                # noise_var = self.ctrl_noise if self.ctrl_noise >= 1 else self.ctrl_noise*abs(mean)
                # current_ctrl[param[:-1]][int(param[-1])] =  np.random.uniform(mean-noise_var,mean+noise_var)
                # current_ctrl[param[:-1]][int(param[-1])] =  np.random.normal(mean,noise_var)    
            else:
                self.current_variation[param] = np.random.uniform(-self.ctrl_noise,self.ctrl_noise,len(self.ctrl[param]))
                current_ctrl[param] = self.ctrl[param]*(1 + self.current_variation[param])
                
                # mean = self.ctrl[param]
                # noise_var = self.ctrl_noise if self.ctrl_noise >= 1 else self.ctrl_noise*abs(mean)
                # current_ctrl[param] =  np.random.uniform(mean-noise_var,mean+noise_var)
                # current_ctrl[param] =  np.random.normal(mean,noise_var)
        
        self.current_ctrl = current_ctrl
        self.calculate_H_sys(current_ctrl)            
            
    def get_expmap(self,pulse,t_step,method='TDSE') -> List[np.ndarray]:     
        '''Need to reset self.H (method=sum) or self.U (method=prod) at the beginning of pulse'''
        b, n = self.bs, self.ns
        num_transmon = self.num_transmon
#         ####
#         # Perturb control params if specified
#         if self.ctrl_noise:
#             for param in self.ctrl.keys():
#                 noise_var = self.ctrl_noise if self.ctrl_noise >= 1 else self.ctrl_noise*abs(self.ctrl[param])
#                 self.current_ctrl =  np.random.normal(self.ctrl[param],noise_var)
                
#             coupling = self.current_ctrl['coupling']
#             anharm = self.current_ctrl['anharm']
#             detune = self.current_ctrl['detune']

#             # System hamiltonian        
#             H_sys = 0j
#             for i in range(num_transmon-1):
#                 H_sys += coupling[i]*( b[i].T @ b[i+1] + b[i] @ b[i+1].T )
#             for i in range(num_transmon):
#                 H_sys += detune[i]*n[i] + anharm[i]/2*( n[i] @ n[i] - n[i])
#         else:
#             self.current_ctrl = self.ctrl.copy()
#             H_sys = self.H_sys.copy()
#         #####
        
        # Update current_ctrl and H_sys if necessary
        if self.ctrl_noise and self.ctrl_update_freq == 'everystep': self.ctrl_update()
        current_ctrl, H_sys = self.current_ctrl.copy(), self.H_sys.copy()
        
        detune = current_ctrl['detune']
        drive = current_ctrl['drive']

        # Control hamiltonian
        H_ctrl = 0j*np.zeros([num_transmon,*H_sys.shape])
        for i in range(num_transmon):
            if num_transmon == 1:
                H_ctrl[i] += b[i]/2 * drive[i] * pulse[i]
            elif num_transmon == 2:
                H_ctrl[i]       += b[i]/2 * drive[2*i]   * pulse[2*i] 
                H_ctrl[(i+1)%2] += b[i]/2 * drive[2*i+1] * pulse[2*i+1]
            else:
                raise NotImplementedError(f'Drive hamiltonian not implemented for {num_transmon} transmons! What are the channels?')
                    
        if method == 'TDSE':
            static = H_sys*self.dt
            dynamic = []
            # detune *=0
            Hargs = {}
            for i in range(len(detune)):
                if abs(detune[i])>1 and abs(H_ctrl[i]*self.dt).max()>1e-8:
                    # print(f'{i}, detune={detune[i]/MHz/2/np.pi:.2f}, Hctrl~{np.linalg.norm(H_ctrl[i]*self.dt):.6f}, Hsys~{np.linalg.norm(static):.6f}')
                    # dynamic.append([qt.Qobj(H_ctrl[i]         *self.dt), lambda t, args: np.exp(1j*detune[i]*t*self.dt)])
                    # dynamic.append([qt.Qobj(H_ctrl[i].T.conj()*self.dt), lambda t, args: np.exp(-1j*detune[i]*t*self.dt)])
                    dynamic.append([qt.Qobj(H_ctrl[i]         *self.dt), [detune0_p, detune1_p][i]])
                    dynamic.append([qt.Qobj(H_ctrl[i].T.conj()*self.dt), [detune0_m, detune1_m][i]])
                    Hargs[f'detune{i}_dt'] = detune[i]*self.dt
                else:
                    static += (H_ctrl[i] + H_ctrl[i].T.conj())*self.dt
            if len(dynamic) == 0:
                # print('Static')
                expmap = self.TDSE_U = qt.propagator(qt.Qobj(static), [t_step,t_step+1])*self.TDSE_U
            else:
                # print('Dynamic')
                expmap = self.TDSE_U = qt.propagator([qt.Qobj(static)]+dynamic, 
                                                     [t_step,t_step+1], 
                                                     args = Hargs)*self.TDSE_U

        elif method == 'TISE':
            phase = np.where(abs(detune)>1,
                             np.exp(1j*self.dt*t_step*detune)/(1j*(detune+1e-5))*(np.exp(1j*self.dt*detune)-1), 
                             self.dt).reshape(-1,1,1)
            H_ctrl = self.H_ctrl_dt = (H_ctrl*phase).sum(0)
            H_sys *= self.dt
            expmap = self.TISE_U = expm(-1j*( H_sys + H_ctrl + H_ctrl.T.conj() ))@self.TISE_U
        

        else:
            raise NotImplementedError(f'Method {method} is not implemented')
            
        return np.array(expmap)
    
    def reset(self):
        self.H_ctrl_dt = 0
        self.TISE_U = np.eye(self.num_level**self.num_transmon)
        self.TDSE_U = qt.qeye(self.num_level**self.num_transmon)   
        # if self.ctrl_noise and self.ctrl_update_freq == 'everyepisode': self.ctrl_update()
        if self.ctrl_noise: self.ctrl_update()
        
    def evolve(self, full_pulse, method):
        assert full_pulse.dtype == np.complex128
        self.reset()
        Us = []   
        for t_step,pulse in enumerate(full_pulse):
            Us.append(self.get_expmap(pulse,t_step,method))
        return np.array(Us)

    def pulse_average_fidelity(self, full_pulse, U_target, qubit_indices,
                               initial_Z_thetas=[], correct_Z_after=True, method='TDSE') -> float:
        U = self.evolve(full_pulse,method)[-1]
            
        if len(initial_Z_thetas) > 0:
            initial_Z = Zgate_on_all(initial_Z_thetas,self.num_level)
            M,theta = projected_overlap(U@initial_Z,U_target,qubit_indices,correct_Z_after)
        else:
            M,theta = projected_overlap(U,U_target,qubit_indices,correct_Z_after)
            
        return average_over_pure_states(M)
    
    def pulse_average_fidelities(self, full_pulse, U_target, qubit_indices,
                                 initial_Z_thetas=[], correct_Z_after=True, method='TDSE'):
        Us = self.evolve(full_pulse,method)

        avg_fids = []
        if len(initial_Z_thetas) > 0:
            initial_Z = Zgate_on_all(initial_Z_thetas,self.num_level)
        for U in Us:
            if len(initial_Z_thetas) > 0:
                M,theta = projected_overlap(U@initial_Z,U_target,qubit_indices,correct_Z_after)
            else:
                M,theta = projected_overlap(U,U_target,qubit_indices,correct_Z_after)
            avg_fids.append(average_over_pure_states(M))
        return np.array(avg_fids),theta,Us
    
def detune0_p(t, args):
    return np.exp(1j*t*args['detune0_dt'])
def detune1_p(t, args):
    return np.exp(1j*t*args['detune1_dt'])
def detune0_m(t, args):
    return np.exp(-1j*t*args['detune0_dt'])
def detune1_m(t, args):
    return np.exp(-1j*t*args['detune1_dt'])
    
# class TransmonDuffingSimulator(object):
#     '''
#     - Simulation in SI units (Hz, sec, etc.)
#     '''
#     def __init__(self, params):
#         num_transmon = self.num_transmon = self.L= params['num_transmon']
#         num_level = self.num_level = params['num_level']
#         self.dim = num_level**num_transmon
#         if self.num_transmon == 1:
#             self.num_channel = 2
#         elif self.num_transmon == 2:
#             self.num_channel = 8
#         else:
#             raise NotImplementedError(f'Drive hamiltonian not implemented for {self.num_transmon} transmons! What are the channels?')

#         self.dt = params['dt']
#         # self.qubit_indices,_ = qubit_subspace(self.num_level,self.num_transmon)
        
#         # Set up raising, lowering, and occupancy operators
#         b = np.diag(np.sqrt(np.arange(1, num_level)), 1)
#         n = np.diag(np.arange(num_level))
#         Id = np.eye(num_level)
#         self.bs, self.ns = [], []
#         for i  in range(num_transmon):
#             b_list, n_list = [Id]*num_transmon, [Id]*num_transmon
#             b_list[i], n_list[i] = b, n
#             self.bs.append(tensor(b_list))
#             self.ns.append(tensor(n_list))
        
#         # System control + control noise
#         self.ctrl = params['ctrl']
#         self.ctrl_noise = params['ctrl_noise']
#         self.current_ctrl = {}
        
#         if self.ctrl_noise:  
#             if self.ctrl_noise > 1:
#                 print(f'-   Noisy control with variance: {self.ctrl_noise/MHz} MHZ')
#             else:
#                 print(f'-   Noisy control with variance: {self.ctrl_noise*100}%')
#         else:
#             print('-   Noiseless control')
            
#         if params['sim_frame_rotation']:
#             raise NotImplementedError('Check implementation in quspin simulator')
#         else:
#             self.dressed_to_sim = np.eye(num_level**num_transmon)
            
#     def get_expmap(self,pulse,t_step) -> List[np.ndarray]:     
        
#         # Perturb control params if specified
#         if self.ctrl_noise:
#             for param in self.ctrl.keys():
#                 noise_var = self.ctrl_noise if self.ctrl_noise >= 1 else self.ctrl_noise*abs(self.ctrl[param])
#                 self.current_ctrl =  np.random.normal(self.ctrl[param],noise_var)
#         else:
#             self.current_ctrl = self.ctrl.copy()
            
#         # For readability
#         b, n = self.bs, self.ns
#         coupling = self.current_ctrl['coupling']
#         anharm = self.current_ctrl['anharm']
#         detune = self.current_ctrl['detune']
#         drive = self.current_ctrl['drive']
        
#         # System hamiltonian        
#         H_sys = 0
#         for i in range(self.num_transmon-1):
#             H_sys += coupling[i]*( b[i].T @ b[i+1] + b[i] @ b[i+1].T )
#         for i in range(self.num_transmon):
#             H_sys += detune[i]*n[i] + anharm[i]/2*( n[i] @ n[i] - n[i])
            
#         # Control hamiltonian
#         phase = np.exp(1j*self.dt*(t_step+1)*detune)
#         H_ctrl = 0
#         for i in range(self.num_transmon):
#             if self.num_transmon == 1:
#                 term = b[i]/2 * drive[i] * phase[i] * (pulse[i] + 1j*pulse[i+1])
#             elif self.num_transmon == 2:
#                 term = b[i]/2 * ( drive[2*i]  *phase[i]      *(pulse[4*i]   + 1j*pulse[4*i+1]) + 
#                                   drive[2*i+1]*phase[(i+1)%2]*(pulse[4*i+2] + 1j*pulse[4*i+3]) )
#             else:
#                 raise NotImplementedError(f'Drive hamiltonian not implemented for {self.num_transmon} transmons! What are the channels?')
#             H_ctrl += term + term.conj().T
            
#         expmap = expm(-1j*(H_sys+H_ctrl)*self.dt)
#         return expmap
    
#     def evolve(self, full_pulse):
#         Us = [np.eye(self.num_level**self.num_transmon)]   
#         for t_step,pulse in enumerate(full_pulse):
#             Us.append(self.get_expmap(pulse,t_step)@Us[-1])
#         return np.array(Us)
    
#     def pulse_average_fidelity(self, full_pulse, U_target, qubit_indices,
#                                initial_Z_thetas=[], correct_Z_after=True) -> float:
#         U = np.eye(len(U_target))        
#         for t_step,pulse in enumerate(full_pulse):
#             U = self.get_expmap(pulse,t_step)@U
            
#         if len(initial_Z_thetas) > 0:
#             initial_Z = Zgate_on_all(initial_Z_thetas,self.num_level)
#             M,theta = projected_overlap(U@initial_Z,U_target,qubit_indices,correct_Z_after)
#         else:
#             M,theta = projected_overlap(U,U_target,qubit_indices,correct_Z_after)
            
#         return average_over_pure_states(M)
    
#     def pulse_average_fidelities(self, full_pulse, U_target, qubit_indices,
#                                  initial_Z_thetas=[], correct_Z_after=True):
#         U = np.eye(len(U_target))
#         avg_fids = []
#         if len(initial_Z_thetas) > 0:
#             initial_Z = Zgate_on_all(initial_Z_thetas,self.num_level)
            
#         for t_step,pulse in enumerate(full_pulse):
#             U = self.get_expmap(pulse,t_step)@U
#             if len(initial_Z_thetas) > 0:
#                 M,theta = projected_overlap(U@initial_Z,U_target,qubit_indices,correct_Z_after)
#             else:
#                 M,theta = projected_overlap(U,U_target,qubit_indices,correct_Z_after)
#             avg_fids.append(average_over_pure_states(M))
#         return np.array(avg_fids),theta,U
    
#############################################################################
############################## OLD QUSPIN SIM ###############################
#############################################################################
# from quspin.operators import hamiltonian 
# from quspin.basis import boson_basis_1d

# class quspin_duffing_simulator():
#     '''
#     Input: params
#         - ctrl: control parameters in MHz
#         - dt: time segment size in us 
#         - error:
#             + If within [0,1], treat as percentage of parameter's amplitude
#             + If larger than 1, treat as an absolute uncertainty in Hz
#     '''
#     def __init__(self, params):
#         L = self.L = params['num_transmon']
#         self.num_level = params['num_level']
#         if self.L == 1:
#             self.num_channel = 2
#         elif self.L == 2:
#             self.num_channel = 8
#         self.basis = boson_basis_1d(L=self.L, sps=self.num_level)

#         self.ctrl = params['ctrl']
#         self.ctrl_noise = params['ctrl_noise']
#         self.current_ctrl = {}
#         dt = self.dt = params['dt']
#         self.coupling_list = [[i,j] for i in range(self.L) for j in range(i+1,self.L)]
        
#         # fixed error in Hz
#         if self.ctrl_noise:  
#             if self.ctrl_noise > 1:
#                 print(f'-   Noisy control with variance: {self.ctrl_noise/MHz} MHZ')
#             else:
#                 print(f'-   Noisy control with variance: {self.ctrl_noise*100}%')
#         else:
#             print('-   Noiseless control')

            
#         detune_coeff   = [[2*PI*coeff*dt,i]       for i,coeff in zip(range(L),self.ctrl['detune'])]
#         anharm_coeff   = [[  PI*coeff*dt,i,i,i,i] for i,coeff in zip(range(L),self.ctrl['anharm'])]
#         coupling_coeff = [[2*PI*coeff*dt,*ij]     for ij,coeff in zip(self.coupling_list,self.ctrl['coupling'])]
            
#         static = [['n',detune_coeff],
#                   ['++--',anharm_coeff],
#                   ['+-',coupling_coeff],
#                   ['-+',coupling_coeff]]
        
#         if params['sim_frame_rotation']:
#             ham_no_drive = np.flip(hamiltonian(static,[],dtype=np.complex128,basis=self.basis,
#                                     check_symm=False,check_herm=False).toarray())
#             self.dressed_to_sim = block_diag_transf_mat(ham_no_drive,self.num_level)
#         else:
#             self.dressed_to_sim = np.eye(self.num_level**self.L)
#         # _,self.dressed_to_sim = np.linalg.eigh(ham_no_drive.toarray())

#     def get_expmap(self, pulse, t_step):
#         expmap, expmap_super = [],[]
#         L, dt = self.L, self.dt
#         # Perturb control params if specified
#         if self.ctrl_noise:
#             for param in self.ctrl.keys():
#                 noise_var = self.ctrl_noise/MHz if self.ctrl_noise >= 1 else self.ctrl_noise*abs(self.ctrl[param])
#                 self.current_ctrl =  np.random.normal(self.ctrl[param],noise_var)
#         else:
#             self.current_ctrl = self.ctrl.copy()
            
#         self.detune_coeff   = [[2*PI*coeff*dt,i]       for i,coeff in zip(range(L),self.current_ctrl['detune'])]
#         self.anharm_coeff   = [[  PI*coeff*dt,i,i,i,i] for i,coeff in zip(range(L),self.current_ctrl['anharm'])]
#         self.coupling_coeff = [[2*PI*coeff*dt,*ij]     for ij,coeff in zip(self.coupling_list,self.current_ctrl['coupling'])]
            
#         # for pulse in actions:
#         drive_coeff_p,drive_coeff_m = [],[]
#         for i in range(L):
#             # D(t)b + D(t).conj()b^dag
#             drive_coeff_p.append([(pulse[2*i]-1j*pulse[2*i+1]) * 2*PI*dt*self.current_ctrl['drive'][i],i])
#             drive_coeff_m.append([(pulse[2*i]+1j*pulse[2*i+1]) * 2*PI*dt*self.current_ctrl['drive'][i],i])
#         static = [['+',drive_coeff_p],
#                   ['-',drive_coeff_m],
#                   ['n',self.detune_coeff],
#                   ['++--',self.anharm_coeff],
#                   ['+-',self.coupling_coeff],
#                   ['-+',self.coupling_coeff]]
#         H = np.flip(hamiltonian(static,[],dtype=np.complex128,basis=self.basis,
#                                 check_symm=False,check_herm=False).toarray()) 
#         expmap = expm(-1j*H)
#         return expmap      
