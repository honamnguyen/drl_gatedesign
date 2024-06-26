import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import minimize
from qtool.scqp import solve_SCQP


#############################################################################
#################################### MISC ###################################
#############################################################################

# def tensor(arr):
#     '''
#     Return the tensor product of all operators in arr
#     '''
#     prod = arr[0]
#     for i in range(1,len(arr)):
#         prod = np.kron(prod,arr[i])
#     return prod

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

def qubit_subspace(num_level,num_transmon):
    '''
    Return qubit_indices_vec, qubit_indices_mat and qubit_projector for multi-level transmon
    '''
    all_indices = np.array(list(product(range(num_level),repeat=num_transmon))) 
    qubit_indices = np.where(all_indices.max(1)<=1)[0]
    qubit_proj = np.diag((all_indices.max(1)<=1).astype(int))
    return qubit_indices, np.ix_(qubit_indices,qubit_indices), qubit_proj

def kets2vecs_optimized(kets,basis):
    vecs = kets.conj()@basis@kets
    return vecs.real

def ket_coeff_to_bloch_vector(z,basis):
    norm = np.sqrt(z[:4]@z[:4])
    psi = z[:4].astype(np.complex128)
    psi[1:4] *= np.exp(1j*z[4:])
    return kets2vecs_optimized(psi/norm,basis)

def block_diag_transf_mat(hamiltonian,num_level):
    '''
    Block diagonalize two-transmon hamiltonian
    
    Input: Hamiltonian with stardard indexing [00,01,02,10,11,12,20,21,22]
    '''
    n = np.diag(np.arange(num_level))
    I = np.eye(num_level)
    n1 = tensor([n,I])
    n2 = tensor([I,n])
    
    # turn into blocks via indexing by number of excitations
    perm = np.argsort(np.diag(n1 + n2))
    idx = np.ix_(perm,perm)
    block_n = [np.where(np.diag(n1 + n2).astype(int) == n)[0].tolist() for n in range(5)]
    
    S = np.eye(num_level**2,dtype=np.complex128)
    i = 0
    for block in block_n:
        if len(block)>1:
            _,evecs = np.linalg.eigh(hamiltonian[idx][i:i+len(block),i:i+len(block)])
            for evec in evecs.T:
                ind = abs(abs(evec)-1).argmin()
                S[i:i+len(block),i+ind] = evec * np.sign(evec[ind])
        i += len(block)
        
    # revert back to standard indexing
    inv_perm = np.argsort(perm)
    inv_idx = np.ix_(inv_perm,inv_perm)
    return S[inv_idx]

def pca(state,num_level,num_transmon,order=(2,1),test=False):
    '''
    Reduce dimension via eigendecomp
    
    Input: state from transmon env [d^2,basis_size]
    Output: reduced input state for RL agent
    '''
    d = num_level**num_transmon
    # Combine dms to reduce # of nonzero evals
    dms = state.T.reshape(-1,d,d).copy()
    dms[1:] += dms[0]
    dms += np.eye(d)/d

    order_0,order_i = order
    evals,evecs = np.linalg.eigh(dms)
    evals[abs(evals)<1e-14] = 0
    trunc_evals = np.hstack([evals[0,-order_0:],evals[1:,-order_i:].flatten()]) # evals[abs(evals)>1e-10]
    trunc_evecs = np.vstack([evecs[0,:,-order_0:].T,np.swapaxes(evecs[1:,:,-order_i:],1,2).reshape(-1,d)])
    scaled_evecs = np.sqrt(trunc_evals)[:,None]*trunc_evecs

    if test:
        scaled_dms = np.einsum('ij,ik->ijk',scaled_evecs,scaled_evecs.conj())
        recon_dms = [scaled_dms[:order_0].sum(0)]
        i = 0
        while i<len(scaled_dms)-order_0:
            recon_dms.append(scaled_dms[order_0+i:order_0+i+order_i].sum(0))
            i += order_i
        recon_dms = np.array(recon_dms)
        diff = np.linalg.norm(dms-recon_dms)/np.linalg.norm(dms)
        print(f'Reconstruction relative error at order {order}: {diff:.3e}')
        
    return scaled_evecs

#############################################################################
################################## MATRIX ###################################
#############################################################################

def gellmann(j,k,d):
    '''
    Returns a generalized Gell-Mann matrix (GGM) of dimension d.
    
    Follow *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008)
    . d(d-1)/2 symmetric GGMs
    . d(d-1)/2 antisymmetric GGMs
    . d-1 diagonal GGMs
    . 1 identity
    '''
    g = np.zeros((d,d),dtype=np.complex128)
    if j > k:
        g[j,k] = 1
        g[k,j] = 1
    elif j < k:
        g[j,k] = -1j
        g[k,j] = 1j
    elif j == k and j < d-1:
        j+=1
        g[:j,:j] = np.eye(j)
        g[j,j] = -j
        g *= np.sqrt(2/j/(j+1))
    else:
        g = np.eye(d)
    return g
        
    
def get_basis(d,jax=False):
    '''
    Return a basis of orthogonal Hermitian operators on a Hilbert space
    of dimension d, with the identity element in the first place.
    
    Example:
    --------
    get_basis(2) returns the Pauli matrices (I,X,Y,Z)
    get_basis(3) returns the Gellmann matrices
    '''
    basis = [gellmann(j, k, d) for j, k in product(range(d), repeat=2)][::-1]
    return basis


def get_reduced_basis(d,L):
    '''
    Return the set of operators to be tracked
    '''
    basis = np.array(get_basis(d))
    size = len(basis)
    
    # Sort and pick basis elements
    if d==2:
        ind = [1,2,3]
        factor = np.ones(3)*0.5
    elif d==3:
        order = np.argsort([0,6,4,7,8,1,5,2,3])
        basis = basis[order]
        ind = [8,1,2,3] #[1,2,3,8] #
        factor = np.ones(4)*0.5
        factor[0] /= np.sqrt(3)
    else:
        print(f'Entire basis for SU({d})')
    
    if L==1:
        return np.einsum('i...,i->i...',basis[ind],factor)
    elif L==2:
        if d==2:
            # Products of Paulis and Identity
            pbasis = []
            for i in range(1,4):
                pbasis.append(tensor([basis[0],basis[i]]))
            for i in range(1,4):
                pbasis.append(tensor([basis[i],basis[0]]))
            for i in range(1,4):
                for j in range(1,4):
                    pbasis.append(tensor([basis[i],basis[j]]))
            return np.array(pbasis)
        elif d==3:
            # Linear combinations of products of Gell-Manns and Identity
            pbasis = []
            pbasis.append(4/9*( np.eye(9) + 3/4 * tensor([basis[8],basis[8]]) + 
                                np.sqrt(3)/2 * (tensor([basis[8],basis[0]]) + tensor([basis[0],basis[8]])) ))
            for i in range(1,4):
                pbasis.append(2/3          * tensor([basis[0],basis[i]]) + 
                              1/np.sqrt(3) * tensor([basis[8],basis[i]]) )
            for i in range(1,4):
                pbasis.append(2/3          * tensor([basis[i],basis[0]]) + 
                              1/np.sqrt(3) * tensor([basis[i],basis[8]]) )
            for i in range(1,4):
                for j in range(1,4):
                    pbasis.append(tensor([basis[i],basis[j]]))
            return np.array(pbasis)
    else:
        print(f'Not implemented for {L} qudits!')
        raise NotImplementedError
        
def get_ket_basis(num_level,num_transmon):
    all_indices = np.array(list(product(range(num_level),repeat=num_transmon)))
    qubit_indices = np.where(all_indices.max(1)<=1)[0]
    return np.eye(num_level**num_transmon,dtype=np.complex128)[qubit_indices]


###################################################################################
################################## QUANTUM GATES ##################################
###################################################################################

P0 = np.array([[1,0],[0,0]])
P1 = np.array([[0,0],[0,1]])
I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
X90 = (I-1j*X)/np.sqrt(2)
S = np.diag([1,1j])

def common_gate(name):
    gate_dict = {'sqrtZX': 1/np.sqrt(2)*np.array([[ 1, 1j,   0,   0],
                                                    [1j,  1,   0,   0],
                                                    [ 0,  0,   1, -1j],
                                                    [ 0,  0, -1j,   1]]),
                 
                 'ZX90': 1/np.sqrt(2)*np.array([[  1, 0, -1j,  0],
                                                [  0, 1,   0, 1j],
                                                [-1j, 0,   1,  0],
                                                [  0, 1j,  0,  1]]),
                 
                 'ZXp90' : (np.eye(4)-1j*tensor([Z,X]))/np.sqrt(2),
                 'ZXm90' : (np.eye(4)+1j*tensor([Z,X]))/np.sqrt(2),
                 'XZp90' : (np.eye(4)-1j*tensor([X,Z]))/np.sqrt(2),
                 'XZm90' : (np.eye(4)+1j*tensor([X,Z]))/np.sqrt(2),
                 
                 'ZX': np.array([[ 0, 1,  0,  0],
                                 [ 1, 0,  0,  0],
                                 [ 0, 0,  0, -1],
                                 [ 0, 0, -1,  0]]),
                 
                 'CNOT': np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]),
                 
                 'NOTC': tensor([I,P0])+tensor([X,P1]),
                 
                 'NOTCCNOT': np.array([[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1],
                                       [0, 1, 0, 0]]),
                 
                  'SWAP': np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]]), 
                 'X': X,
                 'Y': Y,
                 'Z': Z,
                 'XI': tensor([X,I]),
                 'IX': tensor([I,X]),
                 
                 'X90': X90, 
                
                 
                 'IX90': tensor([I,X90]),
                 'X90I': tensor([X90,I]),
                 'X90X90': tensor([X90,X90]),
                 
                 'H': H,
                 'II': tensor([I,I]),
                 'IH': tensor([I,H]),
                 'HI': tensor([H,I]),
                 'HH': tensor([H,H]),
                 
                }
    
    if name in gate_dict:
        return gate_dict[name]
    else:
        print(f'The {name} gate is not implemented')
        raise NotImplementedError

def Zgate(theta):
    return np.diag([1,np.exp(1j*theta)])

def Zgate_on_all(thetas,num_level=2):
    Z_list = []
    for i,theta in enumerate(thetas):
        Z_list.append(Zgate(theta))
        
    Zs = np.eye(num_level**len(thetas),dtype=np.complex128)
    _,qubit_indices,_ = qubit_subspace(num_level,len(thetas))
    Zs[qubit_indices] = tensor(Z_list)
    return Zs

def get_trace(operator, pauli):
    rate = np.trace(operator@pauli)
    assert rate.imag < 1e-10
    return rate.real

#------------------------------------- Gate fidelity -------------------------------------#

def worst_fidelity(env,method='SCQP-dm-0',bounds=None,init_guess=None,overlap_args=(None,[None])):
    '''
    Input: ndarray of shape [num_level^2^N,basis_size]
    Allowed method:
    - 1-transmon: SCQP-dm-0, SLSQP-ket-2,SLSQP-ket-3
    - 2-transmon: SLSQP-ket-7, SLSQP-dm-7 
    '''
    def quadprog(x,A,b):
        return 0.5*x@A@x + x@b
    
    def quadprog_ketcoeff(z,A,b,basis):
        x = ket_coeff_to_bloch_vector(z,basis)
        return 0.5*x@A@x + x@b 
        
    eps = 1e-7
    num_param = [int(s) for s in method.split('-') if s.isdigit()][0]
    M_qubit,theta = overlap_args
    if 'dm' in method:
        if theta[0] is not None:
            # raise NotImplementedError('Fix virtualZ_on_control, now can perform Z on all transmons')
            virtualZI = Zgate_on_all(theta,num_level=3) #virtualZ_on_control(theta,num_level=3)
            correction = tensor([virtualZI,virtualZI.conj()])
        else:
            correction = np.eye(env.state.shape[0])
        A,b,constraints = bloch_args(env,correction)
        
        # reduce to eigenvalue problem
        if (abs(b)<eps).all() and constraints is None:
            evals,evecs = np.linalg.eigh(A)
            return evecs[:,0],0.5*evals[0] #+ c
        
        elif 'SCQP' in method:
            x = solve_SCQP(A,b)
            return x,quadprog(x,A,b) #+c
        
        elif 'SLSQP' in method:
            # only implemented for 2-transmon
            return SLSQP_minimize(num_param,c,quadprog_ketcoeff,(A,b,get_reduced_basis(d=2,L=2)))
        
    elif 'ket' in method:
        return SLSQP_minimize(num_param,0,fidelity_from_ket,(M_qubit),bounds=bounds)
    
def SLSQP_minimize(num_param,const,obj_func,args,jac=None,init_guess=None,bounds=None,cons=()):
    '''
    Use SLSQP method from scipy
    Add two additional runs from random initial guess when obj_fun excess the threshold
    '''
    z0 = np.random.random(num_param) if init_guess is None else init_guess
    options = {'maxiter': 500, 'ftol': 1e-10, 'disp': False, 'eps': 1e-10}
    res = minimize(obj_func,z0,args=args,jac=jac,method='SLSQP',bounds=bounds,options=options,constraints=cons)
    if res.fun+const < 0.99:
        return res.x,res.fun+const
    else:
        xs,fs = [res.x],[res.fun]
        for i in range(2):
            res = minimize(obj_func,np.random.random(num_param),args=args,jac=jac,method='SLSQP',bounds=bounds,options=options,constraints=cons)
            xs.append(res.x)
            fs.append(res.fun)
        return xs[np.argmin(fs)],min(fs)+const

### Both fidelity functions are valid ###
def fidelity_from_basis_kets(z,evolved,target,num_qubit):
    '''
    Input
        z: real parameters
        evolved: evolved basis kets psi_i(t)
        target:  target basis kets psi^T_i
    Output
        |<psi^T|psi(t)>|^2 where psi = z_i@psi_i
    '''
    psi = param2ket(z,num_qubit)
    overlap = psi.conj()@target.T.conj()@evolved@psi
    res = overlap.conj()*overlap
    assert abs(res.imag) < 1e-10
    return res.real

# tiny tiny bit faster
def fidelity_from_ket(z,M_qubit):
    '''
    Input
        z: real parameters
        M_qubit[k,n,n]: overlap between target and evolution
    Output
        |<psi|M|psi>|^2
    '''
    psi = param2ket(z,M_qubit.shape[-1]//2)
    overlap = psi.conj()@M_qubit@psi
    res = overlap.conj()@overlap
    # print(abs(res.imag))
    assert abs(res.imag) < 1e-10
    return res.real

def param2ket(z,num_qubit):
    ''' From real parameters to a complex ket'''
    if num_qubit == 1:
        # 2 DOFs: (z,theta) -> psi = sqrt(1-z**2)|0> + z*exp(1j*theta)|1>
        if len(z) == 2:
            psi = np.array([np.sqrt(1-z[0]**2),z[0]*np.exp(1j*z[1])])
        # 3 DOFs: (z0,z1,theta) -> psi = z0|0> + z1*exp(1j*theta)|1>
        elif len(z) == 3:
            psi = z[:2].astype(np.complex128)/np.sqrt(z[:2]@z[:2])
            psi[1] *= np.exp(1j*z[2])
    elif num_qubit == 2:
        # 7 DOFs: (z0,z1,z2,z3,theta1,theta2,theta3)
        if len(z) == 7:
            psi = z[:4].astype(np.complex128)/np.sqrt(z[:4]@z[:4])
            psi[1:4] *= np.exp(1j*z[4:])
    else:
        raise NotImplementedError
    
    return psi

def bloch_args(env,correction):
    '''
    Input: ndarray of shape [num_level^2^N,basis_size]
    Return:
        Bloch representation arguments, i.e. A_ij and b_i
    '''
    num_transmon,num_level = env.sim.L,env.sim.num_level
    evolved_basis,target_basis = correction@env.state,env.target_state

    eps = 5e-8
    if num_level == 2:
        
        if num_transmon == 1:
            A = (evolved_basis.T.conj() @ target_basis+
                 target_basis.T.conj() @ evolved_basis)
            b = np.zeros(len(A))
            c = 0.5
            constraints = None
        else:
            print(f'Worst fidelity is not implemented!')
            raise NotImplementedError

    elif num_level == 3:
        
        if num_transmon == 1:
            # first element is lambda_8
            evolved_pauli,evolved_lam8 = evolved_basis[:,1:],evolved_basis[:,0]
            target_pauli,target_lam8 = target_basis[:,1:],target_basis[:,0]
            c = (1/3) + (evolved_lam8@target_lam8)
            b = (evolved_lam8.conj() @ target_pauli + 
                 target_lam8.conj() @ evolved_pauli)
            A = (evolved_pauli.T.conj() @ target_pauli +
                 target_pauli.T.conj() @ evolved_pauli) + 2*c*np.eye(len(b))
            constraints = None
            
        elif num_transmon == 2:
            # first element is Lambda_0
            evolved_pauli,evolved_lam0 = evolved_basis[:,1:],evolved_basis[:,0]
            target_pauli,target_lam0 = target_basis[:,1:],target_basis[:,0]
            b = (evolved_lam0.conj() @ target_pauli + 
                 target_lam0.conj()  @ evolved_pauli)/16
            A = evolved_lam0.conj()@target_lam0/24*np.eye(len(b)) +\
                (evolved_pauli.T.conj() @ target_pauli +
                 target_pauli.T.conj()  @ evolved_pauli)/16
                 
            constraints = 'useless string'

        else:
            print(f'Worst fidelity is not implemented!')
            raise NotImplementedError
        assert (abs(b.imag)<eps).all()
        assert (abs(A.imag)<eps).all()
        assert (abs(A.real.T-A.real)<eps).all()
        return A.real,b.real,constraints    

def NLI(fidelity):
    '''Negative logarithmic infidelity'''
    return -np.log10(1-fidelity)

def fromNLI(nli):
    '''From negative logarithmic infidelity to fidelity'''
    return 1 - 10**(-nli)

def average_over_pure_states(M):
    '''
    Calculate average fidelity for a known quantum operation
    Input: M[k,n,n] with k is the number of Kraus operators. k=1 recovers the unitary case
    '''
    eps = 1e-13
    n = M.shape[-1]
    
    term1 = np.einsum('ijk,ijk->',M,M.conj())
    
    traceM = M.trace(axis1=1,axis2=2)
    term2 = traceM.conj() @ traceM
    
    assert abs(term1.imag)<eps and abs(term2.imag)<eps
    return (term1+term2).real/n/(n+1)

def projected_overlap(operator,target,qubit_indices,correct_Z_after=False):
    '''
    Implement average fidelity with projection onto the qubit subspace
    
    M_k = Z @ P(E_k U^dag)P
    - Z is the correctional virtual Z-gate(s)
    - the order means the correction is applied by modifying the phase of the next pulse
    
    Compute correction angle via
        tan theta = - Im(sum_M)/Re(sum_M)
    where for 1q: sum_M = ( (M[0,0]+M[1,1]).conj() * (M[2,2]+M[3,3]) ).sum()
          for 2q: sum_M = ( (M[0,0]+M[1,1]).conj() * (M[2,2]+M[3,3]) ).sum()
    '''
    def _correction_angle(sum_M):
        if abs(sum_M.real)<1e-16:
            theta = np.pi/2*np.sign(-sum_M.imag*sum_M.real)
        else:
            theta = np.nan_to_num(np.arctan(-sum_M.imag/sum_M.real))
        # Pick the other solution using the sign of 2nd deriv
        if np.sin(theta)*sum_M.imag > np.cos(theta)*sum_M.real: 
            theta = theta-np.sign(theta)*np.pi
        return theta
    
    full_map = operator
    if len(operator)==len(target):
        kraus_ops = full_map[None,...]
    else:
        kraus_ops = super2kraus(full_map)
    # M = (target.T.conj()@kraus_ops)
    M = kraus_ops@target.T.conj() # this means virtual-Z is applied for the next pulse
    M_qubit = np.array([m[qubit_indices] for m in M])
    
    if correct_Z_after:
        num_qubit = M_qubit.shape[-1]//2
        indices = np.array(list(product(range(2),repeat=num_qubit)))
        thetas = []
        for i in range(num_qubit):
            # 1 on 0, e^(i theta) on 1
            zero_part = sum([M_qubit[:,j,j] for j in np.where(indices[:,i] == 0)[0]])
            one_part = sum([M_qubit[:,j,j] for j in np.where(indices[:,i] == 1)[0]])
            sum_M = (zero_part.conj()@one_part).sum()
            theta = _correction_angle(sum_M)

            # virtual Z on the corresponding qubit
            Z_correction = [I]*num_qubit
            Z_correction[i] = Zgate(theta)
            M_qubit = tensor(Z_correction)@M_qubit
            thetas.append(theta)
        return M_qubit, thetas
    else:
        return M_qubit, None

def super2choi(super_op):
    dA = int(np.sqrt(super_op.shape[0])) 
    dB = int(np.sqrt(super_op.shape[1]))
    Q = super_op.reshape(dA,dA,dB,dB)
    # row reshuffle Q_mi;nj -> Q_mn;ij
    Q = Q.swapaxes(1,2) #Q = Q.swapaxes(0,3)
    return Q.reshape(dA*dB,dA*dB)

def choi2kraus(choi):
    eps = 1e-10
    d = int(np.sqrt(choi.shape[0]))
    evals,evecs = np.linalg.eig(choi)
    
    # picking non-zero eigenvalues
    assert abs(evals.imag).max() < eps
    ind = np.where(abs(evals.real)>eps)
    evecs = evecs[:,ind]
    evals = evals.real[ind]
    return np.einsum('i,ijk->ijk',np.sqrt(evals),evecs.T.reshape(-1,d,d))

def super2kraus(super_op,test=False):
    kraus = choi2kraus(super2choi(super_op))
    if test:
        eps = 1e-10
        recon_super_op = 0
        for k in kraus:
            recon_super_op += tensor([k,k.conj()])
        assert(abs(recon_super_op-super_op).max()<eps)
        print('Correctly reconstructed super operator!')
    return kraus

#---------------------------- Leakage ------------------------------#

def compute_leakage(env):
    '''
    Compute leakage from and seepage to the computational basis
    
    Ref: Wood & Gambetta 2017, Quantification and Characterization of Leakage Errors 
    '''
    I1 = env.qubit_proj.flatten()
    I2 = (np.eye(env.sim.dim) - env.qubit_proj).flatten()
    L1 = env.map_super@(I1/I1.sum()) @ I2.conj()
    L2 = 1 - env.map_super@(I2/I2.sum()) @ I2.conj()
    assert (abs(L1.imag) + abs(L2.imag))<1e-10 
    L1, L2 = L1.real, L2.real
    return L1, L2

#---------------------------- Entanglement ------------------------------#
def partial_trace(state_vector, indices, num_level=2, test=False):
    '''
    Supports a single or a batch of state vectors [batch,...]
    output reduced density matrix for the qubits specified by the indices
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
        kets = state_vector
        from qutip import Qobj, ket2dm
        rho_qutip = []
        for i in range(state_vector.shape[-1]):
            ket = Qobj(kets[i])
            ket.dims = [list(shape),[1]]
            rho_qutip.append(ket.ptrace(i).data.toarray())
        rho_qutip = np.array(rho_qutip)
        print(f'Diff between qutip and this implementation {abs(rho-rho_qutip).max():.5e}')
        
    return rho[0] if rho.shape[0] == 1 else rho

def purity(rho):
    if len(rho.shape) == 2:
        rho = rho[None,...]
    pur = np.einsum('bij,bji->b',rho,rho)
    return pur[0] if pur.shape[0] == 1 else pur

single_qubit_cliffords = [
    I,
    H, S,
    H@S, S@H, S@S,
    H@S@H, H@S@S, S@H@S, S@S@H, S@S@S,
    H@S@H@S, H@S@S@H, H@S@S@S, S@H@S@S, S@S@H@S,
    H@S@H@S@S, H@S@S@H@S, S@H@S@S@H, S@H@S@S@S, S@S@H@S@S,
    H@S@H@S@S@H, H@S@H@S@S@S, H@S@S@H@S@S,
]
clifford_states = np.array([U[:,0] for U in single_qubit_cliffords])
clifford_states_labels = [
    'z+', 
    'x+', 'z+',
    'x+', 'y+', 'z+',
    '~iy-', 'x+', 'y+', 'x-', 'z+',
    '~iy-', 'z-', 'x+', 'y+', 'x-',
    '~iy-', 'z-', 'iz-', 'y+', 'x-',
    'ix-', '~iy-', 'z-',
]

pauli_states = np.vstack([
    np.array([1,0]),
    np.array([0,1]),
    np.array([1,1])/np.sqrt(2),
    np.array([1,-1])/np.sqrt(2),
    np.array([1,1j])/np.sqrt(2),
    np.array([1,-1j])/np.sqrt(2),
])

def von_Neumann_entropy(rhos):
    e = np.linalg.eigh(rhos)[0]
    entropy = (-e*np.nan_to_num(np.log(e))).sum(1)
    return entropy

def get_linear_entropy_3design(U,normalized=True,states='pauli'):
    if states == 'pauli':
        kets = np.einsum('mi,nj->mnij',pauli_states,pauli_states).reshape(pauli_states.shape[0]**2,4)
    elif states == 'clifford':
        kets = np.einsum('mi,nj->mnij',clifford_states,clifford_states).reshape(clifford_states.shape[0]**2,4)
        
    kets = np.einsum('ij,bj->bi',U,kets)
    if normalized:
        overlap = np.einsum('ni,ni->n',kets.conj(),kets)
        kets /= np.sqrt(overlap)[:,None]
    return 1-purity(partial_trace(kets,[0])).real

def get_von_Neumann_entropy_3design(U,normalized=True):
    kets = np.einsum('mi,nj->mnij',pauli_states,pauli_states).reshape(pauli_states.shape[0]**2,4)
    kets = np.einsum('ij,nj->ni',U,kets)
    if normalized:
        overlap = np.einsum('ni,ni->n',kets.conj(),kets)
        kets /= np.sqrt(overlap)[:,None]
    # doesn't matter which qubit to trace out
    return von_Neumann_entropy(partial_trace(kets,[0])) 


###########################################################################
################################## PULSE ##################################
###########################################################################

#---------------------------- Common pulses ------------------------------#

try:
    from qiskit.pulse import library as qiskitpulse

    def qiskit_drag_pulse(duration, amp, sigma, beta, phi):
        return np.exp(-1j*phi)*qiskitpulse.Drag(duration, amp, sigma, beta).get_waveform().samples.reshape(-1,1)

    def _base_cr_pulse(duration, amp, sigma, width, phi, XI_params=None):
        if XI_params is not None:
            pi_pulse = qiskit_drag_pulse(**XI_params)
            cr_p = qiskitpulse.GaussianSquare(duration, amp, sigma, width).get_waveform().samples
            pi_pulse = np.hstack([pi_pulse,np.zeros_like(pi_pulse)])
            cr_p = np.vstack([np.zeros_like(cr_p),cr_p]).T
            cr_m = -cr_p
            return np.exp(-1j*phi)*np.vstack([cr_p,pi_pulse,cr_m,pi_pulse])
        else:
            return np.exp(-1j*phi)*qiskitpulse.GaussianSquare(duration, amp, sigma, width).get_waveform().samples.reshape(-1,1)

    # def qiskit_cr_pulse(duration, amp, sigma, width, phi, XI_params=None, 
    #                     cancel_amp=None, cancel_phi=None):
    #     cr_pulse = _base_cr_pulse(duration, amp, sigma, width, phi, XI_params)
    #     if cancel_amp is not None:
    #         cancel_pulse = _base_cr_pulse(duration, cancel_amp, sigma, width, cancel_phi, XI_params)
    #         if XI_params is not None: cancel_pulse = cancel_pulse[:,[1]]
    #         cr_pulse = np.hstack([cr_pulse, cancel_pulse])
    #     return cr_pulse
#     def qiskit_cr_pulse(duration, amp, sigma, width, phi, XI_params=None, 
#                         cancel_amp=None, cancel_phi=None, 
#                         rotary_amp=None, rotary_phi=None):
#         cr_pulse = _base_cr_pulse(duration, amp, sigma, width, phi, XI_params)

#         # if echo
#         if XI_params is not None: 
#             if cancel_amp is not None:
#                 cancel_pulse = _base_cr_pulse(duration, cancel_amp, sigma, width, cancel_phi, XI_params)[:,[1]]
#                 cr_pulse = np.hstack([cr_pulse, cancel_pulse])
#         # if direct
#         else:
#             if (cancel_amp is not None) or (rotary_amp is not None):
#                 target_pulse = np.zeros_like(cr_pulse)
#                 if cancel_amp is not None:
#                     target_pulse += _base_cr_pulse(duration, cancel_amp, sigma, width, cancel_phi)
#                 if rotary_amp is not None:
#                     risefall = duration-width
#                     rotary_pulse = _base_cr_pulse(duration//2, rotary_amp, sigma, width-duration//2, rotary_phi)
#                     target_pulse += np.vstack([rotary_pulse,-rotary_pulse])
#                 cr_pulse = np.hstack([cr_pulse, target_pulse])
#         return cr_pulse
    def qiskit_cr_pulse(duration, amp, sigma, width, phi, XI_params=None, 
                        cancel_amp=None, cancel_phi=None, 
                        rotary_amp=None, rotary_phi=None, output_all=False):
        cr_pulse = _base_cr_pulse(duration, amp, sigma, width, phi, XI_params)

        # if echo
        if XI_params is not None: 
            if cancel_amp is not None:
                cancel_pulse = _base_cr_pulse(duration, cancel_amp, sigma, width, cancel_phi, XI_params)[:,[1]]
                cr_pulse = np.hstack([cr_pulse, cancel_pulse])
        # if direct
        else:
            if (cancel_amp is not None) or (rotary_amp is not None):
                target_pulse = np.zeros_like(cr_pulse)
                if cancel_amp is not None:
                    cancel_pulse = _base_cr_pulse(duration, cancel_amp, sigma, width, cancel_phi)
                    target_pulse += cancel_pulse
                if rotary_amp is not None:
                    risefall = duration-width
                    rotary = _base_cr_pulse(duration//2, rotary_amp, sigma, width-duration//2, rotary_phi)
                    rotary_pulse = np.vstack([rotary,-rotary])
                    target_pulse += rotary_pulse
                cr_pulse = np.hstack([cr_pulse, target_pulse])
        if output_all:
            return cr_pulse, cancel_pulse, rotary_pulse
        else:
            return cr_pulse
except:
    print('Qiskit not found. Some pulse shapes are not available.')

def DRAG(num_seg, amp, sig, beta):
    '''
    DRAG pulse for implementing 1 qubit rotation
    '''
    
    t_gauss = np.linspace(0,num_seg-1,num_seg)
    gauss = amp*np.exp( -(t_gauss-t_gauss.max()/2)**2 /2 /(sig*len(t_gauss))**2 )
    return np.array([gauss,beta*( -(t_gauss-t_gauss.max()/2) / (sig*len(t_gauss))**2 )*gauss]).T

    
def GaussianSquare(num_seg, gauss_ratio, amp, sig):
    '''
    Guassian rise and fall with a flat top
    '''
    num_seg_gauss = int(gauss_ratio*num_seg)
    num_seg_const = num_seg - num_seg_gauss
    
    t_gauss = np.linspace(0,num_seg_gauss-1,num_seg_gauss)
    t_const = np.ones(num_seg_const)
    
    gauss = amp*np.exp( -(t_gauss-t_gauss.max()/2)**2 /2 /(sig*len(t_gauss))**2 )
    const = gauss.max()*t_const
    pulse = np.hstack([gauss[:len(t_gauss)//2],const,gauss[len(t_gauss)//2:]])
    return np.vstack([pulse,np.zeros_like(pulse)]).T

def Z_shift(pulse,theta):
    '''
    Effectively perform a virtual Z gate before a pulse
    '''
    complex_pulse = (pulse[:,0]+1j*pulse[:,1])*np.exp(1j*theta)
    shifted_pulse = np.zeros_like(pulse)
    shifted_pulse[:,0] = complex_pulse.real
    shifted_pulse[:,1] = complex_pulse.imag
    return shifted_pulse

def plot_pulse(pulse,channel_labels,axs=None,xlim=None,ylim='adjusted',ysize=1,xsize=10):
    '''
    For discretized complex pulse
    '''
    if pulse.dtype == np.float64: 
        pulse *= (1+0j)
    
    pulse = np.vstack([pulse,np.zeros_like(pulse[0])])
    steps = range(len(pulse))
    
    num_channel = len(channel_labels)
    if axs is None:
        fig, axs = plt.subplots(num_channel,1,sharex=True,figsize=(xsize,ysize*num_channel))
    if num_channel == 1: axs = [axs]
    for i in range(num_channel):
        axs[i].fill_between(steps,pulse[:,i].real  ,color='C3',step='post',alpha=0.4)
        axs[i].fill_between(steps,pulse[:,i].imag,color='C0',step='post',alpha=0.4)
        axs[i].step(steps,pulse[:,i].real  ,'C3',where='post',label='Re')
        axs[i].step(steps,pulse[:,i].imag,'C0',where='post',label='Im')
        axs[i].set_ylabel(channel_labels[i])
        if ylim=='standard':
            axs[i].set_ylim([-1.1,1.1])
        elif ylim=='adjusted':
            ymin = min(pulse[:,i].real.min(),pulse[:,i].imag.min())
            ymax = max(pulse[:,i].real.max(),pulse[:,i].imag.max())
            axs[i].set_ylim([ymin*1.3,ymax*1.3])  
        else:
            raise NotImplementedError
        if xlim is None:
            axs[i].hlines(0,-int(0.1*len(pulse)),len(pulse)+int(0.1*len(pulse)),'grey',alpha=0.4)
            axs[i].set_xlim([-int(0.1*len(pulse)),len(pulse)+int(0.1*len(pulse))])
        else:
            axs[i].hlines(0,*xlim,'grey',alpha=0.4)
            axs[i].set_xlim(xlim)
        if i == 0: axs[i].legend(loc='upper right')
        if i == num_channel-1: axs[i].set_xlabel('Time step')
    # plt.show()

    
###########################################################################
################################# QISKIT ##################################
###########################################################################

#---------------------------- Bloch plot ------------------------------#

def bloch_vecs(kets, q_idx):
    '''
    Args:
        kets[:,dim]
    '''
    num_qubit = int(np.log2(kets.shape[1]))
    vecs = []
    for P in [X,Y,Z]:
        op = [I]*num_qubit
        op[q_idx] = P
        op = tensor(op)
        vecs.append(np.einsum('ij,jk,ik->i',kets.conj(),op,kets))
    vecs = np.array(vecs)
    assert abs(vecs.imag).max()<1e-9
    return vecs.real

