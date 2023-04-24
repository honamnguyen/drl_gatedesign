#####################################################################################
###################################### CORE SIM ######################################

import numpy as np, itertools

class PQC():
    '''
    Parameterzed Quantum Circuit
    '''
    def __init__(self, params):
        self.num_qubits = params['num_qubits']
        self.N = 2**self.num_qubits
        self.sequence = []
        self.gateset_1q, self.gateset_2q = params['gateset']
        self.method = params['method'] if 'method' in params else 'standard'
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
                
        if self.method == 'precalc':
            self.precalc_init()

        self.reset()
    
    def precalc_init(self):
        print('Precalculating gate combinations')
        # add base_gateset to gatedict
        self.base_gateset_1q = [g.replace('R','') for g in self.gateset_1q] 
        self.base_gateset_2q = self.gateset_2q
        assert len(self.base_gateset_2q) == 1
        for gate in self.base_gateset_1q: 
            if gate not in self.gateset_1q:
                for qubit in range(self.num_qubits):
                    gatename, unitary = gate_to_unitary(gate, [qubit], self.num_qubits)
                    self.gatedict[gatename] = unitary
                    
        # precalculate unitaries for all gate combinations in one layer
        self.gate_combinations = {}
        # loop through the allowed number of 2q gates
        for num_2q in range(self.num_qubits//2+1):
            num_1q = self.num_qubits - 2*num_2q
            gatecombs_1q = list(itertools.product(['I']+self.base_gateset_1q,repeat=num_1q)) 
            all_locs = [str(q) for q in range(self.num_qubits)]

            # loop over ctrl_loc in all_locs
            for ctrl_loc in itertools.combinations(all_locs, num_2q):
                remain_locs = [loc for loc in all_locs if loc not in ctrl_loc]
                # loop over targ_loc in remain_locs
                for targ_loc in itertools.combinations(remain_locs, num_2q):
                    locs_1q = [loc for loc in remain_locs if loc not in targ_loc]
                    for targ_ind in itertools.permutations(targ_loc):
                        layer_2q = [self.base_gateset_2q[0]+''.join(ind) for ind in zip(ctrl_loc,targ_ind)]
                        for gatecomb in gatecombs_1q:
                            layer = layer_2q + [''.join(ind) for ind in zip(gatecomb,locs_1q) if ind[0] != 'I']
                            # layer = layer_2q + [''.join(ind) for ind in zip(gatecomb,locs_1q)]
                            # Calculate tensor product for each layer
                            U = np.eye(2**self.num_qubits)
                            # U = sp.eye(2**self.num_qubits)
                            for gatename in layer:
                                # if 'I' not in gatename:
                                    # U = pqc.gatedict[gatename] @ sp.csr_matrix(U)
                                U = self.gatedict[gatename] @ U
                            layer.sort()
                            self.gate_combinations['_'.join(layer)] = U
                            # Us.append(U.astype(np.float32))

            
    def update_gateset(self, gate, qubits):
        if 'R' not in gate:
            gatename, unitary = gate_to_unitary(gate, qubits, self.num_qubits)
            self.gatedict[gatename] = unitary
        self.gatelist.append((gate,qubits))

    def reset(self):
        # state dim: [layer, qubit, gate class]
        self.state = np.zeros([1,self.num_qubits,len(self.gateclass)],dtype=int)
        self.used_qubits = np.zeros(self.num_qubits,dtype=int) #if qubit already used in current layer
        if 'standard' in self.method:
            self.fixed_Us = [np.eye(self.N)]
            self.param_layers = [[]] # only has nontrivial gates
        elif 'precalc' in self.method:
            self.layers = [[]]
        self.num_params = 0
        self.kets = []
        self.gateseq = []
           
    def append(self, gate, qubits):
        '''
        Parallelize new gate, otherwise add new layer 
        
        Currently only support single-qubit parameterized gates
        '''
        self.gateseq.append([gate,qubits])
        # if qubit already acted on, move to next layer
        if (self.used_qubits[qubits]==1).any():
            self.state = np.vstack([self.state,np.zeros([1,self.num_qubits,len(self.gateclass)],dtype=int)])
            self.used_qubits *= 0
            if 'standard' in self.method:
                self.fixed_Us.append(np.eye(self.N))
                self.param_layers.append([])
            elif 'precalc' in self.method:
                self.layers.append([])

            
        # update state
        self.used_qubits[qubits] = 1
        if len(qubits) == 1:
            self.state[-1,qubits[0],self.gateclass.index(gate)] = 1
        elif len(qubits) == 2:
            self.state[-1,qubits[0],self.gateclass.index(gate+'_ctrl')] = 1
            self.state[-1,qubits[1],self.gateclass.index(gate+'_targ')] = 1
        
        # update layer
        if 'standard' in self.method:
            if 'R' in gate:
                self.param_layers[-1].append([gate,qubits])
                self.num_params += 1
            else:
                U = self.gatedict[gate+''.join([str(q) for q in qubits])]
                self.fixed_Us[-1] = U @ self.fixed_Us[-1] 
        elif 'precalc' in self.method:
            self.layers[-1].append(''.join([str(x) for x in [gate,*qubits]]))
            if 'R' in gate: self.num_params += 1
            
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
        
    
    def sample(self, shots=5000, override_thetas=[]):
        self.thetas = thetas = override_thetas if len(override_thetas) else np.random.uniform(0, 2*np.pi, size=[self.num_params, shots])

        ### evolution with a batch of sample ###
        param_i = 0
        
        if 'standard' in self.method:
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
            ket0 = np.repeat(np.eye(self.N)[0][None],shots,0)
            ket = 0
            for layer in self.layers:
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

                #loop over all terms with the appropriate coefficients
                for i in range(len(prods)):
                    prod = [gate for gate in prods[i] if 'I' not in gate]
                    prod.sort()
                    U = self.gate_combinations['_'.join(prod)]
                    # ket += np.einsum('b,ij,bj->bi',coeffs[i],U,ket0)
                    # ket += coeffs[i][:,None]*ket0.dot(U.T)
                    ket += np.multiply(coeffs[i][:,None],ket0).dot(U.T)
                ket0 = ket.copy()
                ket = 0
                    
                # Us = np.array([self.gate_combinations['_'.join([gate for gate in prod if 'I' not in gate])] for prod in prods])
                # # ket0 = np.einsum('ab,aij,bj->bi',coeffs,Us,ket0)
                # # ket0 = np.einsum('ijk,jk->ji',coeffs.T@np.moveaxis(Us,0,1),ket0)
                # ket0 = np.einsum('jk,ijk->ki',coeffs,np.moveaxis(Us,0,1)@ket0.T)
    
        if ket0.shape[0] == 1:
            self.kets = np.repeat(ket0,shots,axis=0)
            return np.repeat(ket0,shots,axis=0)
        else:
            self.kets = ket0
            return ket0
    
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

    def expressibility_stat(self, shots, num_bins, trials=5):
        kls = []
        for _ in range(trials):
            kls.append(self.expressibility(shots, num_bins)[0])
        kls = np.array(kls)
        return kls.mean(), kls.std()
    
    def entanglement_capability(self, shots=5000, measure='meyer_wallach', input_kets=[]):
        if len(input_kets):
            kets = input_kets
        else:
            kets = self.kets.copy() if len(self.kets) else self.sample(shots)
        if measure == 'meyer_wallach':
            avg_purity = np.array([purity(partial_trace(kets,[i])) for i in range(self.num_qubits)]).mean()
            Q = 2*(1 - avg_purity)
            assert abs(Q.imag) < 1e-10
            return Q.real
    
    @property
    def depth(self):
        return self.state.shape[0]
    
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