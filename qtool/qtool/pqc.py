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
        self.reset()
            
    def update_gateset(self, gate, qubits):
        if 'R' not in gate:
            gatename, unitary = gate_to_unitary(gate, qubits, self.num_qubits)
            self.gatedict[gatename] = unitary
        self.gatelist.append((gate,qubits))

    def reset(self):
        # state dim: [layer, qubit, gate class]
        self.state = np.zeros([1,self.num_qubits,len(self.gateclass)],dtype=int)
        self.used_qubits = np.zeros(self.num_qubits,dtype=int) #if qubit already used in current layer
        self.fixed_Us = [np.eye(self.N)]
        self.param_layers = [[]] # only has nontrivial gates
        self.num_params = 0
           
    def append(self, gate, qubits):
        '''
        Parallelize new gate, otherwise add new layer 
        
        Currently only support single-qubit parameterized gates
        '''
        # if qubit already acted on, move to next layer
        if (self.used_qubits[qubits]==1).any():
            self.state = np.vstack([self.state,np.zeros([1,self.num_qubits,len(self.gateclass)],dtype=int)])
            self.used_qubits *= 0
            self.fixed_Us.append(np.eye(self.N))
            self.param_layers.append([])
            
        # update state
        self.used_qubits[qubits] = 1
        if len(qubits) == 1:
            self.state[-1,qubits[0],self.gateclass.index(gate)] = 1
        elif len(qubits) == 2:
            self.state[-1,qubits[0],self.gateclass.index(gate+'_ctrl')] = 1
            self.state[-1,qubits[1],self.gateclass.index(gate+'_targ')] = 1
        
        # update layer
        if 'R' in gate:
            self.param_layers[-1].append([gate,qubits])
            self.num_params += 1
        else:
            U = self.gatedict[gate+''.join([str(q) for q in qubits])]
            self.fixed_Us[-1] = U @ self.fixed_Us[-1]   
            
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
        
    
    def sample(self, shots=5000):
        thetas = np.random.uniform(0, 2*np.pi, size=[self.num_params, shots])
        
        ### evolution with a batch of sample ###
        ket0 = np.eye(self.N)[0][None]
        param_i = 0
        for i in range(len(self.fixed_Us)):
            # ket0 = self.fixed_Us[i][None] @ ket0
            ket0 = np.einsum('jk,ik->ij',self.fixed_Us[i],ket0)
            if len(self.param_layers[i]) > 0:
                param_Us = [I]*self.num_qubits
                for gate, qubits in self.param_layers[i]:
                    ### TODO: generalize for two-qubit param-ed gates ###
                    param_Us[qubits[0]] = parameterized_gates(gate)(thetas[param_i]) 
                    param_i += 1
                # ket0 = tensor(param_Us) @ ket0
                ket0 = np.einsum('ijk,ik->ij',tensor(param_Us),ket0)
        if ket0.shape[0] == 1:
            return np.repeat(ket0,shots,axis=0)
        else:
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
    
    @property
    def depth(self):
        return self.state.shape[0]
        
def parameterized_gates(gatename):
    '''collection of parameterized gates'''
    if 'R' in gatename:
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
            supports a single P_i with battch dimension
        einsum_loop: 
            replace kron with einsum
            best for changing final_dim + len(arr) and num_qubts>4
            supports multiple P_is with the same batch dimension
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
                dim = prod.shape[-1]*arr[i].shape[-1]
                prod = np.einsum(subscripts,prod,arr[i]).reshape(-1,dim,dim)
    elif method == 'einsum':
        final_dim, subscripts = args
        return np.einsum(subscripts,*arr).reshape(final_dim,final_dim)
    return prod

def common_gate(name):
    gate_dict = { 
        'X': X,
        'Y': Y,
        'Z': Z,
        'H': 1/np.sqrt(2)*np.array([[1,1],[1,-1]]),
    }
    if name in gate_dict:
        return gate_dict[name]
    else:
        print(f'The {name} gate is not implemented')
        raise NotImplementedError
        
def gate_to_unitary(gate, qubits, num_qubits):
    if gate in ['H']:
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

        def sample(self, shots=5000):
            thetas_samples = np.random.uniform(0, 2*np.pi, size=[shots,len(self.params)])
            circuits = []
            for thetas in thetas_samples:
                bind_dict = {self.params[i]: theta for i,theta in enumerate(thetas)}
                circuits.append(self.circ.bind_parameters(bind_dict))

            pqc_kets = []
            for circuit in circuits:
                result = backend.run(assemble(circuit)).result()
                pqc_kets.append(result.get_statevector())
            return np.array(pqc_kets)

        def expressibility(self, shots=5000, num_bins=75):        
            kets = self.sample(2*shots).reshape(2,shots,self.N)
            bins = np.linspace(0,1,num_bins)
            counts, _ = np.histogram(fidelity(kets[0],kets[1]),bins=bins)
            midpoints = (bins[1:]+bins[:-1])/2
            p_PQC = counts/shots
            p_Haar = pdf_Haar(midpoints,self.N)/(num_bins-1)
            return kl_divergence(p_PQC,p_Haar), p_PQC, p_Haar
except:
    print('Qiskit not found. PQC_Qiskit class is not available.')