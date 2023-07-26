#####################################################################################
###################################### CORE SIM ######################################

import numpy as np, itertools, scipy, glob, pickle
from tqdm import tqdm

try:
    import tensorflow as tf
except:
    print('Tensorflow is not installed!')

class PQC():
    '''
    Parameterzed Quantum Circuit
    '''
    def __init__(self, params):
        self.num_qubits = params['num_qubits']
        self.N = 2**self.num_qubits
        self.gateset_1q, self.gateset_2q = params['gateset']
        self.method = params['method'] if 'method' in params else 'precalc'
        self.dtype = params['dtype'] if 'dtype' in params else 'complex64'
        self.special_type = params['special_type'] if 'special_type' in params else None
        
        # Set up gateset, 1q gate is its own class, 2q gate is split to control and target
        self.gatelist = []
        self.gateclass = []
        self.gatedict = {}
        for gate in self.gateset_1q:
            self.gateclass.append(gate)
            for qubit in range(self.num_qubits):
                self.update_gateset(gate, [qubit])
        for gate in self.gateset_2q:
            self.gateclass.append(gate+'_ctrl')
            self.gateclass.append(gate+'_targ')
            for qubits in itertools.permutations(range(self.num_qubits),2):
                self.update_gateset(gate, list(qubits))
                
        if 'precalc' in self.method:
            self.precalc_init()

        self.reset()
    
    def precalc_init(self):
        print('Precalculating gate combinations')
        
        # Add base_gateset to gatedict
        self.base_gateset_1q = [g.replace('R','') for g in self.gateset_1q] 
        self.base_gateset_2q = self.gateset_2q
        
        for gate in self.base_gateset_1q: 
            if gate not in self.gateset_1q:
                for qubit in range(self.num_qubits):
                    gatename, unitary = gate_to_unitary(gate, [qubit], self.num_qubits)
                    self.gatedict[gatename] = unitary.astype(self.dtype)
                    
        self.gate_combinations = get_gate_combinations(self.num_qubits, 
                                                       self.base_gateset_1q, 
                                                       self.base_gateset_2q, 
                                                       self.gatedict,
                                                       self.special_type)
#         all_loc = [str(q) for q in range(self.num_qubits)]
#         for num_2q in range(self.num_qubits//2+1):
#             num_1q = self.num_qubits - 2*num_2q
#             gate_1q_products = list(itertools.product(['I']+self.base_gateset_1q, repeat=num_1q))
#             gate_2q_products = list(itertools.product(self.base_gateset_2q, repeat=num_2q))
            
#             for ctrl_loc in itertools.combinations(all_loc, num_2q):
#                 remain_loc = [loc for loc in all_loc if loc not in ctrl_loc]
                
#                 for targ_loc in itertools.permutations(remain_loc, num_2q):
#                     loc_1q = [loc for loc in remain_loc if loc not in targ_loc]
                    
#                     for gate_2q_product in gate_2q_products:
#                         layer_2q = [''.join(ind) for ind in zip(gate_2q_product,ctrl_loc,targ_loc)]

#                         for gate_1q_product in gate_1q_products:
#                             layer = layer_2q + [''.join(ind) for ind in zip(gate_1q_product,loc_1q) if ind[0] != 'I']
#                             layer.sort() # to ensure a consistent order/unique key
                            
#                             # Calculate tensor product for each layer
#                             U = np.eye(2**self.num_qubits) # sp.eye(2**self.num_qubits)
#                             for gatename in layer:                                    
#                                 U = self.gatedict[gatename] @ U # U = pqc.gatedict[gatename] @ sp.csr_matrix(U)
#                             self.gate_combinations['_'.join(layer)] = U #U.astype(np.float32)
                            
#         # verify that there are no duplicates
#         assert len(np.unique(list(self.gate_combinations.keys()))) == len(list(self.gate_combinations.keys()))
                        
                ######## old combinations ##########
                # for targ_loc in itertools.combinations(remain_locs, num_2q):
                #     locs_1q = [loc for loc in remain_locs if loc not in targ_loc]
                #     for targ_ind in itertools.permutations(targ_loc):
                #         print(targ_loc,targ_ind)
                #         layer_2q = [self.base_gateset_2q[0]+''.join(ind) for ind in zip(ctrl_loc,targ_ind)]
                #         print(layer_2q)
                #         for gatecomb in gatecombs_1q:
                #             layer = layer_2q + [''.join(ind) for ind in zip(gatecomb,locs_1q) if ind[0] != 'I']
                #             # layer = layer_2q + [''.join(ind) for ind in zip(gatecomb,locs_1q)]
                #             # Calculate tensor product for each layer
                #             U = np.eye(2**self.num_qubits)
                #             # U = sp.eye(2**self.num_qubits)
                #             for gatename in layer:
                #                 # if 'I' not in gatename:
                #                     # U = pqc.gatedict[gatename] @ sp.csr_matrix(U)
                #                 U = self.gatedict[gatename] @ U
                #             layer.sort()
                #             self.gate_combinations['_'.join(layer)] = U
                #             # Us.append(U.astype(np.float32))
                ########################################

            
    def update_gateset(self, gate, qubits):
        if 'R' not in gate:
            gatename, unitary = gate_to_unitary(gate, qubits, self.num_qubits)
            self.gatedict[gatename] = unitary.astype(self.dtype)

        self.gatelist.append((gate,qubits))

    def reset(self):
        # state dim: [layer, qubit, gate class]
        self.state = np.zeros([1,self.num_qubits,len(self.gateclass)],dtype=int)
        self.avail_layer = np.zeros(self.num_qubits,dtype=int) # he next avail layer for each qubit
        self.param_indices = [[]]
        if 'standard' in self.method:
            self.fixed_Us = [np.eye(self.N)]
            self.param_layers = [[]] # only has nontrivial gates
        elif 'precalc' in self.method:
            self.layers = [[]]
        self.num_params = 0
        self.kets = []
        self.gateseq = []
        
    def undo(self):
        '''Undo one step'''
        self.gateseq = self.previous_step_dict['gateseq']
        self.avail_layer = self.previous_step_dict['avail_layer']
        self.state = self.previous_step_dict['state']
        self.param_indices = self.previous_step_dict['param_indices']
        self.num_params = self.previous_step_dict['num_params']
        
        if 'standard' in self.method:
            self.fixed_Us = self.previous_step_dict['fixed_Us']
            self.param_layers = self.previous_step_dict['param_layers']
        elif 'precalc' in self.method:
            self.layers = self.previous_step_dict['layers']
           
    def append(self, gate, qubits):
        '''
        Parallelize new gate, otherwise add new layer 
        
        Currently only support single-qubit parameterized gates
        '''
        # Save the state in previous step for undoing
        self.previous_step_dict = {
            'gateseq': self.gateseq.copy(),
            'avail_layer': self.avail_layer.copy(),
            'state': self.state.copy(),
            'param_indices': self.param_indices.copy(),
            'num_params': self.num_params,
        }
        if 'standard' in self.method:
            self.previous_step_dict['fixed_Us'] = self.fixed_Us.copy()
            self.previous_step_dict['param_layers'] = self.param_layers.copy()
        elif 'precalc' in self.method:
            self.previous_step_dict['layers'] = self.layers.copy()        
        
        self.gateseq.append([gate,qubits])
        # if qubit already acted on, move to next layer
        if (self.avail_layer[qubits]==self.state.shape[0]).any(): 
            self.state = np.vstack([self.state,np.zeros([1,self.num_qubits,len(self.gateclass)],dtype=int)])
            self.param_indices.append([])
            if 'standard' in self.method:
                self.fixed_Us.append(np.eye(self.N))
                self.param_layers.append([])
            elif 'precalc' in self.method:
                self.layers.append([])

        # update state
        layer_pos = max(self.avail_layer[qubits])
        self.avail_layer[qubits] = layer_pos + 1
        if len(qubits) == 1:
            self.state[layer_pos,qubits[0],self.gateclass.index(gate)] = 1
        elif len(qubits) == 2:
            self.state[layer_pos,qubits[0],self.gateclass.index(gate+'_ctrl')] = 1
            self.state[layer_pos,qubits[1],self.gateclass.index(gate+'_targ')] = 1
        
        # update layer
        if 'standard' in self.method:
            if 'R' in gate:
                self.param_layers[layer_pos].append([gate,qubits])
                self.param_indices[layer_pos].append(self.num_params)
                self.num_params += 1
            else:
                U = self.gatedict[gate+''.join([str(q) for q in qubits])]
                self.fixed_Us[layer_pos] = U @ self.fixed_Us[layer_pos] 
        elif 'precalc' in self.method:
            self.layers[layer_pos].append(''.join([str(x) for x in [gate,*qubits]]))
            if 'R' in gate: 
                self.param_indices[layer_pos].append(self.num_params)
                self.num_params += 1
            
    def get_state(self, state_type):
        if '2D' in state_type:
            '''Convert from 3D state to 2D state of shape [gatedepth,qubits]'''
            self.state2D = self.state@np.arange(1,len(self.gateclass)+1)
            return self.state2D
        elif '3D' in state_type:
            return self.state
        
    def update_circuit_from_state(self, state, state_type='2D'):
        if '2D' in state_type:
            pass
        else:
            raise NotImplementedError
        
    def _get_tensorprod_coeff(self, layer, thetas, kind='np'):
        '''
        Break a layer into lin comb of precalculated tensor products
        args:
            thetas: only the parameters for that layer
        '''
        param_i = 0
        if kind == 'np':
            prods = np.array([['NA']])
            coeffs = np.array([[1]])
            for gate_qubits in layer:
                # break RX to I and X
                if 'R' in gate_qubits:
                    gate = gate_qubits[:-1]
                    prods = np.hstack([np.vstack([prods,[gate_qubits.replace(gate,'I')]*prods.shape[1]]),
                                        np.vstack([prods,[gate_qubits.replace('R','')]*prods.shape[1]])])
                    coeffs = np.vstack([    np.cos(thetas[param_i]/2)*coeffs,
                                        -1j*np.sin(thetas[param_i]/2)*coeffs])
                    param_i += 1
                else:
                    prods = np.vstack([prods,[gate_qubits]*prods.shape[1]])
            prods = prods[1:].T.tolist()
            
        elif kind == 'tf':
            prods = np.array([['NA']])
            coeffs = tf.constant([[1]],dtype=tf.complex64)
            for gate_qubits in layer:
                # break RX to I and X
                if 'R' in gate_qubits:
                    gate = gate_qubits[:-1]
                    prods = np.hstack([np.vstack([prods,[gate_qubits.replace(gate,'I')]*prods.shape[1]]),
                                        np.vstack([prods,[gate_qubits.replace('R','')]*prods.shape[1]])])
                    coeffs = tf.concat([    tf.cos(thetas[param_i]/2)*coeffs,
                                        -1j*tf.sin(thetas[param_i]/2)*coeffs],axis=0)             
                    param_i += 1
                else:
                    prods = np.vstack([prods,[gate_qubits]*prods.shape[1]])
            prods = prods[1:].T.tolist()
            
        return prods, coeffs, param_i
    
    def _evolve(self, prods, coeffs, ket0=None, U0=None):
        '''Evolve ket or unitary'''
        ket = U = 0
        for i in range(len(prods)):
            prod = [gate for gate in prods[i] if 'I' not in gate]
            prod.sort()
            if ket0 is not None:
                # ket += np.einsum('b,ij,bj->bi',coeffs[i],U,ket0)
                # ket += coeffs[i][:,None]*ket0.dot(U.T)
                ket += np.multiply(coeffs[i][:,None],ket0).dot(self.gate_combinations['_'.join(prod)].T)
            else:
                U += coeffs[i][:,None,None]*self.gate_combinations['_'.join(prod)]
        if ket0 is not None:
            self.ket = ket
            return self.ket
        else:
            self.U = np.einsum('ijk,ikl->ijl',U,U0) #U.dot(U0)
            return self.U
        
    def sample(self, shots=5000, override_thetas=[]):
        self.thetas = thetas = override_thetas if len(override_thetas) else np.random.uniform(0, 2*np.pi, size=[self.num_params, shots])
        thetas = self.thetas[np.hstack(self.param_indices).astype(int)] #the correct order of parameters
        
        ### evolution with a batch of sample ###
        if 'standard' in self.method:
            param_i = 0
            ket0 = np.eye(self.N)[0][None]
            for i in range(len(self.fixed_Us)):
                # ket0 = self.fixed_Us[i][None] @ ket0
                # ket0 = np.einsum('jk,bk->bj',self.fixed_Us[i],ket0)
                ket0 = ket0@self.fixed_Us[i].T
                if len(self.param_layers[i]) > 0:
                    param_Us = [[I]*self.num_qubits]

                    # Supports only one-qubit parameterized gates
                    for gate, qubits in self.param_layers[i]:
                        # param_Us[qubits[0]] = parameterized_gates(gate)(thetas[param_i]) 
                        for U in param_Us:
                            U[qubits[0]] = parameterized_gates(gate)(thetas[param_i]) 
                        param_i += 1

                    ### TODO: generalize for two-qubit paramterized gates ###
    #                 param_Us = [[I]*self.num_qubits]
    #                 for gate, qubits in self.param_layers[i]:
    #                     if 
    #                     param_Us[qubits[0]] = parameterized_gates(gate)(thetas[param_i]) 
    #                     param_i += 1

    #                 prod_id = [I]*num_qubits
    #                 prod_x = [I]*num_qubits
    #                 prod_id[qubits[0]] = P0
    #                 prod_x[qubits[0]] = P1
    #                 prod_x[qubits[1]] = common_gate(gate[-1])
                    # U = tensor(prod_id) + tensor(prod_x)

                    ket0 = np.einsum('ijk,ik->ij',sum([tensor(U) for U in param_Us]),ket0)
                    # temp = sum([tensor(U) for U in param_Us])@ket0.T
                    # ket0 = np.einsum('iji->ij',temp)
                
        elif 'precalc' in self.method:
            
            if self.special_type == 'tf':
                return self.get_tf_unitaries(self.thetas).numpy().dot(np.eye(self.N)[0])
            
            if 'unitary' in self.method:
                param_start, U0 = 0, np.eye(self.N)[None]
                for layer in self.layers:
                    prods, coeffs, param_count = self._get_tensorprod_coeff(layer, thetas[param_start:])
                    param_start += param_count
                    U0 = self._evolve(prods, coeffs, U0=U0)
                self.ket = self.U.dot(np.eye(self.N)[0])
                return self.ket               
            else:
                # ket0 = np.repeat(np.eye(self.N)[0][None],shots,0)
                param_start, ket0 = 0, np.eye(self.N)[0]
                for layer in self.layers:
                    prods, coeffs, param_count = self._get_tensorprod_coeff(layer, thetas[param_start:])
                    param_start += param_count
                    # print(np.array(prods))
                    # print(thetas.round(2))
                    # print(coeffs.round(2))
                    ket0 = self._evolve(prods, coeffs, ket0=ket0)
                return self.ket
                    
                # Us = np.array([self.gate_combinations['_'.join([gate for gate in prod if 'I' not in gate])] for prod in prods])
                # # ket0 = np.einsum('ab,aij,bj->bi',coeffs,Us,ket0)
                # # ket0 = np.einsum('ijk,jk->ji',coeffs.T@np.moveaxis(Us,0,1),ket0)
                # ket0 = np.einsum('jk,ijk->ki',coeffs,np.moveaxis(Us,0,1)@ket0.T)
    
        if ket0.shape[0] == 1:
            self.kets = np.repeat(ket0,shots,axis=0)
        else:
            self.kets = ket0
        return self.kets
        
    def get_tf_unitaries(self, thetas):
        if thetas.dtype is not tf.complex64:
            thetas = tf.cast(thetas, tf.complex64)
        if thetas.shape[0] != self.num_params:
            thetas = tf.transpose(thetas)
        
        if 'precalc' in self.method:
            U0 = tf.repeat(tf.eye(self.N,dtype=tf.complex64)[None],thetas.shape[1],0)
            U, param_start = 0, 0
            for layer in self.layers:
                prods, coeffs, param_count = self._get_tensorprod_coeff(layer, thetas[param_start:], kind='tf')
                param_start += param_count
                #loop over all terms with the appropriate coefficients
                for i in range(len(prods)):
                    prod = [gate for gate in prods[i] if 'I' not in gate]
                    prod.sort()
                    U += coeffs[i][:,None,None]*self.gate_combinations['_'.join(prod)]
                    
                U0 = tf.matmul(U,U0)
                U = 0
        return U0

    def expressibility(self, shots=5000, num_bins=75, fixed_circuit=True):
        if fixed_circuit and self.num_params == 0:
            return (2**self.num_qubits-1)*np.log(num_bins), None, None
        else:
            kets = self.sample(2*shots).reshape(2,shots,self.N) 
            bins = np.linspace(0,1,num_bins+1)
            counts, _ = np.histogram(fidelity(kets[0],kets[1]),bins=bins)
            midpoints = (bins[1:]+bins[:-1])/2
            p_PQC = counts/shots
            # p_Haar = pdf_Haar(midpoints,self.N)/num_bins # mid point approx
            # p_Haar = pdf_Haar_binned(bins[:-1],bins[1:],self.N)
            # return kl_divergence(p_PQC,p_Haar), p_PQC, p_Haar
            log_p_Haar = log_pdf_Haar_binned(bins[:-1],bins[1:],self.N)
            return kl_divergence_logq(p_PQC,log_p_Haar), p_PQC, log_p_Haar
        
    def expressibilities(self, shots, num_bins, trials=5):
        kls = []
        for _ in range(trials):
            kls.append(self.expressibility(shots, num_bins)[0])
        kls = np.array(kls)
        return kls
    
    def expressibility_minmeanmax(self, shots, num_bins, trials=5):
        kls = []
        for _ in range(trials):
            kls.append(self.expressibility(shots, num_bins)[0])
        kls = np.array(kls)
        return kls.min(), kls.mean(), kls.max()

    def expressibility_stat(self, shots, num_bins, trials=5):
        kls = []
        for _ in range(trials):
            kls.append(self.expressibility(shots, num_bins)[0])
        kls = np.array(kls)
        return kls.mean(), kls.std()
    
    def expressibility_minmeanmax(self, shots, num_bins, trials=5):
        kls = []
        for _ in range(trials):
            kls.append(self.expressibility(shots, num_bins)[0])
        kls = np.array(kls)
        return kls.min(), kls.mean(), kls.max()
    
    def entanglement_capability(self, shots=5000, measure='meyer_wallach', input_kets=[]):
        if len(input_kets):
            kets = input_kets
        else:
            # kets = self.kets.copy() if len(self.kets) else self.sample(shots)
            kets = self.sample(shots)
        if measure == 'meyer_wallach':
            avg_purity = np.array([purity(partial_trace(kets,[i])) for i in range(self.num_qubits)])
            assert abs(avg_purity.imag).max() < 1e-10
            return (2*(1 - avg_purity.real)).mean()
            
    def param_dim(self, thetas=None, method='shift_rule'):
        '''Estimated via effective quantum dimension at a random theta'''
        if self.num_params == 0:
            return 0
        
        if method == 'shift_rule':
            n = self.num_params
            n2 = self.num_params**2
            if thetas is None:
                thetas = np.random.uniform(0, 2*np.pi, n)

            ei = np.eye(n,dtype=int)
            ei_plus_ej = ei[None] + ei[:,None]
            ei_minus_ej = ei[None] - ei[:,None]

            thetas1 = (thetas[None,None] + ei_plus_ej*np.pi/2).reshape(-1,n)
            thetas2 = (thetas[None,None] + ei_minus_ej*np.pi/2).reshape(-1,n)
            thetas3 = (thetas[None,None] - ei_minus_ej*np.pi/2).reshape(-1,n)
            thetas4 = (thetas[None,None] - ei_plus_ej*np.pi/2).reshape(-1,n)
            all_thetas = np.vstack([thetas1,thetas2,thetas3,thetas4,thetas]).T


            kets = self.sample(override_thetas=all_thetas)
            fids = fidelity(kets[-2:-1],kets[:-1])

            self.qfi = -1/8*(  fids[:n2] \
                             - fids[n2:2*n2] \
                             - fids[2*n2:3*n2] \
                             + fids[3*n2:4*n2])
            self.qfi = self.qfi.reshape(n,n)
            return np.linalg.matrix_rank(self.qfi)
    
    @property
    def depth(self):
        return self.state.shape[0]
    @property
    def twoq_depth(self):
        return sum(['C' in ''.join(layer) for layer in self.layers])
    @property
    def twoq_count(self):
        return sum(['C' in gate for gate in np.hstack(self.layers)])
    
    ### Testing
    def test_sample_qiskit(self, shots=10):
        pqc_qiskit = PQC_Qiskit(self.num_qubits)
        pqc_qiskit.reset()
        for gate_qubits in self.gateseq:
            pqc_qiskit.append(*gate_qubits)
        kets = self.sample(shots)
        kets_qiskit = pqc_qiskit.sample(override_thetas=self.thetas)
        print(f'Sampled state difference: {abs(kets-kets_qiskit).max():.3e}')
    
def parameterized_gates(gatename):
    '''collection of parameterized gates'''
    if gatename[0] == 'R':
        return lambda theta: Rotation(theta, gatename[-1])
    elif gatename[:2] == 'CR':
        return lambda theta: Rotation(theta, gatename[-1])        
def Rotation(theta, axis):
    '''
    Single-qubit rotation
        theta: ndarray
        axis: X, Y, or Z
    '''
    extended_theta = theta[...,None,None]
    return I[None,...]*np.cos(extended_theta/2) - 1j*common_gate(axis)[None,...]*np.sin(extended_theta/2)

def fidelity(states1, states2):
    overlap = np.einsum('ij,ij->i',states1,states2.conj())
    return overlap.real**2 + overlap.imag**2

def frame_potential_haar(t,N):
    return np.math.factorial(t)*np.math.factorial(N-1)/np.math.factorial(t+N-1)

def pdf_Haar(F, N):
    return (N-1)*(1-F)**(N-2)

def pdf_Haar_binned(F1, F2, N):
    return (1-F1)**(N-1) - (1-F2)**(N-1)

def log_pdf_Haar_binned(F1, F2, N):
    return (N-1) * np.log(1-F1) + np.log( 1 - ((1-F2)/(1-F1))**(N-1) )

def kl_divergence(p, q):
    '''
    For PQC, with number of shots < 1e16, every p less than eps is identically 0.
    Hence, only p+eps
    '''
    eps = 1e-16
    # return np.sum(np.where(abs(p)>eps, p * np.log((p+eps)/q), 0))
    return np.sum(np.where(abs(p)>eps, p * (np.log(p+eps) - np.log(q)), 0))

def kl_divergence_logq(p, log_q):
    '''
    For PQC, with number of shots < 1e16, every p less than eps is identically 0.
    Hence, only p+eps
    '''
    eps = 1e-16
    return np.sum(np.where(abs(p)>eps, p * (np.log(p+eps) - log_q), 0))

def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2
    Q, R = np.linalg.qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)

def haar_expressibility(N,shots,num_bins):
    bins = np.linspace(0,1,num_bins)
    midpoints = (bins[1:]+bins[:-1])/2
    haar_unitaries = np.stack([[qr_haar(N) for i in range(shots)],
                            [qr_haar(N) for i in range(shots)]])
    init_ket = np.eye(N)[0]
    haar_kets = (haar_unitaries@init_ket)
    counts, _ = np.histogram(fidelity(haar_kets[0],haar_kets[1]),bins=bins)
    p_PQC = counts/shots
    # p_Haar = pdf_Haar(midpoints,N)/(num_bins-1)
    p_Haar = pdf_Haar_binned(bins[:-1],bins[1:],N)
    return kl_divergence(p_PQC,p_Haar)

def sample_generic(N, shots, dist, num_rm_params=0):
    num_params = N**2 - num_rm_params
    if dist[0] == 'normal':
        # mean,std = dist[1:]
        # thetas = np.random.normal(mean,std,[shots,num_params])
        thetas = np.random.normal(*dist[1:],[shots,num_params])
    elif dist[0] == 'uniform':
        thetas = np.random.uniform(*dist[1:],[shots,num_params])
        
    locs = np.hstack([np.arange(num_params),np.random.randint(num_params,size=num_rm_params)])
    np.random.shuffle(locs)
    thetas = thetas[:,locs]
    
    num_off_diags = int(0.5 * (N**2 - N))
    real_off_params = thetas[:,:num_off_diags]
    imag_off_params = thetas[:,num_off_diags:2 * num_off_diags]
    diag_params = thetas[:,2 * num_off_diags:]

    herm_mat = np.array([np.diag(diag_param) for diag_param in diag_params]).astype(np.complex128)
    count = 0
    for i in range(N):
        for j in range(i+1,N):
            herm_mat[:,i,j] = real_off_params[:,count] + 1j*imag_off_params[:,count]
            herm_mat[:,j,i] = real_off_params[:,count] - 1j*imag_off_params[:,count]
            count += 1
    unitary_mat = scipy.linalg.expm(1j*herm_mat)
    # print(abs(np.transpose(unitary_mat,[0,2,1]).conj()@unitary_mat - np.eye(N)).max())
    # print(abs(unitary_mat@np.transpose(unitary_mat,[0,2,1]).conj() - np.eye(N)).max())
    return unitary_mat

def generic_expressibility(N,shots,num_bins,test_log_Haar=False,dist=['normal',0,1],generic_unitaries=None,seed=0,ket0=None):
    bins = np.linspace(0,1,num_bins)
    midpoints = (bins[1:]+bins[:-1])/2
    if generic_unitaries is None:
        generic_unitaries = sample_generic(N,2*shots,dist)
    else:
        np.random.seed(seed)
        locs = np.arange(len(generic_unitaries))
        np.random.shuffle(locs)
        generic_unitaries = generic_unitaries[locs[:2*shots]]
    if ket0 is not None:
        kets = (generic_unitaries@ket0).reshape(2,shots,N)
    else:
        kets = (generic_unitaries@np.eye(N)[0]).reshape(2,shots,N)    
    counts, _ = np.histogram(fidelity(kets[0],kets[1]),bins=bins)
    p_PQC = counts/shots
    
    p_Haar = pdf_Haar_binned(bins[:-1],bins[1:],N)
    kl = kl_divergence(p_PQC,p_Haar)
    
    if test_log_Haar:
        log_p_Haar = log_pdf_Haar_binned(bins[:-1],bins[1:],N)
        print('Diff from log_Haar method:',abs(kl-kl_divergence_logq(p_PQC,log_p_Haar)).max())  
    
    return kl

### FOR ENTANGLEMENT CAPACITY ###
def partial_trace(state_vector, indices, num_level=2, test=False):
    '''
    Supports a single or a batch of state vectors [batch,...]
    '''
    if len(state_vector.shape) == 1:
        state_vector = state_vector[...,None]
    else:
        state_vector = np.moveaxis(state_vector,0,-1)
        
    batch = state_vector.shape[-1]
    n_qubits = int(np.emath.logn(num_level, state_vector.shape[0]))
    shape = (2,)*n_qubits

    indices = list(indices)
    state_vector = state_vector.reshape(*shape,batch)

    sum_inds = np.array(range(n_qubits))
    sum_inds[indices] += n_qubits+1
    
    rho = np.einsum(
        state_vector,
        list(range(n_qubits)) + [n_qubits],
        np.conj(state_vector),
        sum_inds.tolist() + [n_qubits],
        indices + list(sum_inds[indices]) + [n_qubits],
    )
    new_shape = np.prod([shape[i] for i in indices], dtype=np.int64)
    rho = np.moveaxis(rho.reshape((new_shape, new_shape, -1)),-1,0)
    
    if test:
        from qutip import Qobj, ket2dm
        rho_qutip = []
        for i in range(state_vector.shape[-1]):
            ket = Qobj(kets[i])
            ket.dims = [list(shape),[1]]
            rho_qutip.append(ket.ptrace(ind).data.toarray())
        rho_qutip = np.array(rho_qutip)
        print(f'Diff between qutip and this implementation {abs(rho-rho_qutip).max():.5e}')
        
    return rho[0] if rho.shape[0] == 1 else rho

def purity(rho):
    if len(rho.shape) == 2:
        rho = rho[None,...]
    pur = np.einsum('bij,bji->b',rho,rho)
    return pur[0] if pur.shape[0] == 1 else pur

#####################################################################################
###################################### UTILITY ######################################
P0 = np.array([[1,0],[0,0]])
P1 = np.array([[0,0],[0,1]])
I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])

ALPHABET = ['an', 'bo', 'cp', 'dq', 'er', 'fs', 'gt', 'hu', 'iv', 'jw', 'kx', 'ly', 'mz']

def tensor(arr, method='einsum_loop',args=None):
    '''
    Return the tensor product of all operators in arr
    arr: [P_1,P_2,P_3,...,P_N]
    method:
        kron_loop: 
            supports a single P_i with batch dimension
        einsum_loop: 
            replace kron with einsum
            best for changing final_dim + len(arr) and num_qubts>4
            supports multiple P_i's with the same batch dimension
            P_i can be multi-qubit operators: e.g. [H,Z,CRX,Y]
        einsum: 
            use einsum directly, requires args=(subscripts,final_dim), supports max(len(arr)) <= 13 (4.1x faster) 
            best for many shots with fixed final_dim + len(arr) and num_qubits<=4
    '''
    if len(arr) == 1:
        return arr[0]
    if method == 'kron_loop':
        prod = arr[0]
        for i in range(1,len(arr)):
            prod = np.kron(prod,arr[i])
    elif method == 'einsum_loop':
        prod = arr[0]
        for i in range(1,len(arr)):
            # skip scalar
            if type(arr[i]) == int:
                continue
            m,n = len(prod.shape),len(arr[i].shape)
            # support same batch_dimension for some operators in arr
            if m == n == 2:
                subscripts = 'im,jn'
                dim = prod.shape[-1]*arr[i].shape[-1]
                prod = np.einsum(subscripts,prod,arr[i]).reshape(dim,dim)
            else:
                if m == 2 and n == 3:
                    subscripts = 'im,ajn'
                elif m == 3 and n == 2:
                    subscripts = 'aim,jn'
                elif m == 3 and n == 3:
                    subscripts = 'aim,ajn->aijmn'  
                else:
                    print(f'm = {m}, n = {n}')
                dim = prod.shape[-1]*arr[i].shape[-1]
                prod = np.einsum(subscripts,prod,arr[i]).reshape(-1,dim,dim)
    elif method == 'einsum':
        final_dim, subscripts = args
        return np.einsum(subscripts,*arr).reshape(final_dim,final_dim)
    return prod

def common_gate(name):
    gate_dict = { 
        'I': I,
        'X': X,
        'Y': Y,
        'Z': Z,
        'H': 1/np.sqrt(2)*np.array([[1,1],[1,-1]]),
        'SX': 0.5*np.array([[1+1j,1-1j],[1-1j,1+1j]]),
        'CX': np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]]),
        'CY': np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, -1j],
                       [0, 0, 1j, 0]]),
        'CZ': np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, -1]]),
    }
    if name in gate_dict:
        return gate_dict[name]
    elif 'ctrl' in name or 'targ' in name:
        return gate_dict['I']
    else:
        print(f'The {name} gate is not implemented')
        raise NotImplementedError
        
def gate_to_unitary(gate, qubits, num_qubits):
    if gate in ['H','X','Y','Z','SX']:
        prod = [I]*num_qubits
        prod[qubits[0]] = common_gate(gate)
        U = tensor(prod)
        gatename = gate+str(qubits[0])
    elif gate in ['CX','CZ','CY']:
        prod_id = [I]*num_qubits
        prod_x = [I]*num_qubits
        prod_id[qubits[0]] = P0
        prod_x[qubits[0]] = P1
        prod_x[qubits[1]] = common_gate(gate[-1])
        U = tensor(prod_id) + tensor(prod_x)
        gatename = gate+''.join([str(q) for q in qubits])
    else:
        print(f'{gate} is not implemented')
        raise NotImplementedError

    return gatename, U

def get_gate_combinations(num_qubits, gateset_1q, gateset_2q, gatedict, special_type=None):
    '''
    Calculate unitaries for all gate combinations in one layer

    ---Pseudo code---

    Define all locations as [0,1,num_qubit-1] -> all_loc

    Loop over number of 2q gates (0,1,...,num_qubits//2) -> num_2q
        num_1q = num_qubits - num_2q
        Calculate products of 1q gates for num_1q qubits -> gate_1q_products
        Calculate products of 2q gates for num_2q qubits -> gate_2q_products

        Pick num_2q locations from all_loc as controls -> ctrl_loc
            remain_loc = all_loc - ctrl_loc

            Pick num_2q ordered-locations from remain_loc as targets -> targ_loc
                loc_1q = remain_loc - targ_loc

                Pick a product of 2q gates:
                    Create a layer of 2q gates -> layer_2q
                    
                   Pick a product of 1q gates:
                       layer = sort(layer_2q + layer_1q)
                       Compute and add unitary for that layer to dict
    '''
    dtype = list(gatedict.values())[0].dtype
    gate_combinations = {}
    all_loc = [str(q) for q in range(num_qubits)]
    for num_2q in tqdm(range(num_qubits//2+1)):
        num_1q = num_qubits - 2*num_2q
        gate_1q_products = list(itertools.product(['I']+gateset_1q, repeat=num_1q))
        gate_2q_products = list(itertools.product(gateset_2q, repeat=num_2q))

        # for ctrl_loc in tqdm(list(itertools.combinations(all_loc, num_2q))):
        for ctrl_loc in itertools.combinations(all_loc, num_2q):
            remain_loc = [loc for loc in all_loc if loc not in ctrl_loc]

            for targ_loc in itertools.permutations(remain_loc, num_2q):
                loc_1q = [loc for loc in remain_loc if loc not in targ_loc]

                for gate_2q_product in gate_2q_products:
                    layer_2q = [''.join(ind) for ind in zip(gate_2q_product,ctrl_loc,targ_loc)]

                    for gate_1q_product in gate_1q_products:
                        layer = layer_2q + [''.join(ind) for ind in zip(gate_1q_product,loc_1q) if ind[0] != 'I']
                        layer.sort() # to ensure a consistent order/unique key
                        
                        # Calculate tensor product for each layer
                        U = np.eye(2**num_qubits,dtype=dtype) # sp.eye(2**self.num_qubits)
                        for gatename in layer:                                    
                            # U = gatedict[gatename] @ U # U = pqc.gatedict[gatename] @ sp.csr_matrix(U)
                            U = gatedict[gatename].dot(U) # U = pqc.gatedict[gatename] @ sp.csr_matrix(U)
                        if special_type is None:
                            gate_combinations['_'.join(layer)] = U
                        elif special_type == 'sparse':
                            gate_combinations['_'.join(layer)] = scipy.sparse.csr_matrix(U)
                        elif special_type == 'tf':
                            gate_combinations['_'.join(layer)] = tf.convert_to_tensor(U,dtype=tf.complex64)
                        else:
                            print(f'Special type {special_type} option is not implemented!')
                            
    assert len(np.unique(list(gate_combinations.keys()))) == len(list(gate_combinations.keys()))
    return gate_combinations

def get_saved_pqc(label, ray_path=None):
    
    def get_gateset(gateseq):
        gateset_1q = list(set([key[0] for key in gateseq if 'C' not in key[0]]))
        gateset_2q = list(set([key[0] for key in gateseq if 'C' in key[0]]))
        return [gateset_1q,gateset_2q]
    
    if 'simetal' in label:
        gateseq = SimEtAl[label.split('_')[-1]]
        
    elif 'run' in label:
        run, str1, str2 = label.split('_')
        run = run.replace('run','')
        files = glob.glob(f'{ray_path}/*{run}*/all.pkl')
        assert len(files) == 1
        data = pickle.load(open(files[0], 'rb'))
        gateseq = data['gateseqs'][data['labels'].index('_'.join([str1,str2]))]
        
    else:
        raise NotImplementedError(f'`{label}` circuit not found!')
    return gateseq, get_gateset(gateseq)

SimEtAl = {
    '9': [
        ('H',[0]),('H',[1]),('H',[2]),('H',[3]),
        ('CZ',[2,3]),('CZ',[1,2]),('CZ',[0,1]),
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
    ],
    
    '2': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
        ('CX',[3,2]),('CX',[2,1]),('CX',[1,0]),
    ],
    '18': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]), 
        ('CRZ',[3,0]),('CRZ',[2,3]),('CRZ',[1,2]),('CRZ',[0,1]),
    ],
    '12': [
        ('RY',[0]),('RY',[1]),('RY',[2]),('RY',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]), 
        ('CZ',[0,1]),('CZ',[2,3]),
        ('RY',[1]),('RY',[2]),
        ('RZ',[1]),('RZ',[2]),
        ('CZ',[1,2]),
    ],
    '17': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),   
        ('CRX',[1,0]),('CRX',[3,2]),
        ('CRX',[2,1])
    ],
    '11': [
        ('RY',[0]),('RY',[1]),('RY',[2]),('RY',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]), 
        ('CX',[1,0]),('CX',[3,2]),
        ('RY',[1]),('RY',[2]),
        ('RZ',[1]),('RZ',[2]),
        ('CX',[2,1]),
    ],
    '7': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
        ('CRZ',[1,0]),('CRZ',[3,2]),
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
        ('CRZ',[2,1]),
    ],
    '8': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
        ('CRX',[1,0]),('CRX',[3,2]),
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
        ('CRX',[2,1]),
    ],
    '19': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]), 
        ('CRX',[3,0]),('CRX',[2,3]),('CRX',[1,2]),('CRX',[0,1]),
    ],
    '5': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
        ('CRZ',[3,2]),('CRZ',[3,1]),('CRZ',[3,0]),
        ('CRZ',[2,3]),('CRZ',[2,1]),('CRZ',[2,0]),
        ('CRZ',[1,3]),('CRZ',[1,2]),('CRZ',[1,0]),
        ('CRZ',[0,3]),('CRZ',[0,2]),('CRZ',[0,1]),
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
    ],
    '13': [
        ('RY',[0]),('RY',[1]),('RY',[2]),('RY',[3]),
        ('CRZ',[3,0]),('CRZ',[2,3]),('CRZ',[1,2]),('CRZ',[0,1]),
        ('RY',[0]),('RY',[1]),('RY',[2]),('RY',[3]),
        ('CRZ',[3,2]),('CRZ',[0,3]),('CRZ',[1,0]),('CRZ',[2,1]),
    ],
    '14': [
        ('RY',[0]),('RY',[1]),('RY',[2]),('RY',[3]),
        ('CRX',[3,0]),('CRX',[2,3]),('CRX',[1,2]),('CRX',[0,1]),
        ('RY',[0]),('RY',[1]),('RY',[2]),('RY',[3]),
        ('CRX',[3,2]),('CRX',[0,3]),('CRX',[1,0]),('CRX',[2,1]),
    ],
    '6': [
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),
        ('CRX',[3,2]),('CRX',[3,1]),('CRX',[3,0]),
        ('CRX',[2,3]),('CRX',[2,1]),('CRX',[2,0]),
        ('CRX',[1,3]),('CRX',[1,2]),('CRX',[1,0]),
        ('CRX',[0,3]),('CRX',[0,2]),('CRX',[0,1]),
        ('RX',[0]),('RX',[1]),('RX',[2]),('RX',[3]),
        ('RZ',[0]),('RZ',[1]),('RZ',[2]),('RZ',[3]),    
    ],
}

#####################################################################################
###################################### QISKIT SIM ######################################

try:
    from qiskit import QuantumRegister, QuantumCircuit, Aer, assemble
    from qiskit.circuit import Parameter

    class PQC_Qiskit():
        '''
        Parameterzed Quantum Circuit using Qiskit
        '''
        def __init__(self, num_qubits):
            self.num_qubits = num_qubits
            self.N = 2**num_qubits

            self.gateset = {
                'H': QuantumCircuit.h,
                'RX': QuantumCircuit.rx,
                'RY': QuantumCircuit.ry,
                'RZ': QuantumCircuit.rz,
                'CX': QuantumCircuit.cx,
                'CY': QuantumCircuit.cy,
                'CZ': QuantumCircuit.cz,
                'CRX': QuantumCircuit.crx,
                'CRY': QuantumCircuit.cry,
                'CRZ': QuantumCircuit.crz,
                'U': QuantumCircuit.u,
            }
            self.backend = Aer.get_backend('statevector_simulator')
            self.reset()

        def reset(self):
            self.circ = QuantumCircuit(self.num_qubits)
            self.params = []

        def append(self, gate, qubits):
            if 'R' in gate:
                self.params.append(Parameter(f'th{len(self.params)}'))
                self.gateset[gate](self.circ, self.params[-1], *qubits)   
            elif gate == 'U':
                for i in range(3):
                    self.params.append(Parameter(f'th{len(self.params)}'))
                self.gateset[gate](self.circ, *self.params[-3:], *qubits)   
            else:
                self.gateset[gate](self.circ, *qubits)

        def sample(self, shots=5000, override_thetas=[]):
            self.thetas = thetas_samples = override_thetas if len(override_thetas) else np.random.uniform(0, 2*np.pi, size=[len(self.params),shots])
            circuits = []
            for thetas in thetas_samples.T:
                bind_dict = {self.params[i]: theta for i,theta in enumerate(thetas)}
                circuits.append(self.circ.bind_parameters(bind_dict))

            pqc_kets = []
            for circuit in circuits:
                result = self.backend.run(assemble(circuit)).result()
                pqc_kets.append(reverse_qargs(result.get_statevector(),self.num_qubits))
            return np.array(pqc_kets)

        def expressibility(self, shots=5000, num_bins=75):        
            kets = self.sample(2*shots).reshape(2,shots,self.N)
            bins = np.linspace(0,1,num_bins)
            counts, _ = np.histogram(fidelity(kets[0],kets[1]),bins=bins)
            midpoints = (bins[1:]+bins[:-1])/2
            p_PQC = counts/shots
            p_Haar = pdf_Haar(midpoints,self.N)/(num_bins-1)
            return kl_divergence(p_PQC,p_Haar), p_PQC, p_Haar

    def reverse_qargs(state_vector, num_qubits):
        state_vector = state_vector.reshape((2,)*num_qubits).transpose(list(range(num_qubits))[::-1])
        return state_vector.reshape(2**num_qubits)
except:
    print('Qiskit not found. PQC_Qiskit class is not available.')

def test_PQC(pqc, circuits):
    for circuit in circuits:
        pqc.reset()
        for gate_qubits in circuit:
            pqc.append(*gate_qubits)
        pqc.test_sample_qiskit()
        
if __name__ == '__main__':
    
    num_qubits = 4
    num_bins = 75
    shots = 20000
    gateset = [['H','RX','RZ','RY'],['CZ','CX']]

    params = {
        'num_qubits': num_qubits,
        'gateset': gateset,
        'method': 'precalc',
        'dtype': 'complex64',
        'special_type': None,
    }
    pqc = PQC(params)
    circuits = [SimEtAl['9'],SimEtAl['9']*3,SimEtAl['2'],SimEtAl['2']*3]
    
    print('\n---Test standard numpy PQC---')
    pqc.method = 'standard'
    test_PQC(pqc, circuits)
    
    print('\n---Test precalc numpy PQC---')
    pqc.method = 'precalc'
    test_PQC(pqc, circuits)

    print('\n---Test precalc_unitary numpy PQC---')
    pqc.method = 'precalc_unitary'
    test_PQC(pqc, circuits)

    print('\n---Test precalc tensorflow PQC---')
    params = {
        'num_qubits': num_qubits,
        'gateset': gateset,
        'method': 'precalc',
        'dtype': 'complex64',
        'special_type': 'tf',
    }
    pqc = PQC(params)
    test_PQC(pqc, circuits)
    # print(pqc.get_tf_unitaries(tf.random.uniform([4, shots],0, 2*np.pi)))