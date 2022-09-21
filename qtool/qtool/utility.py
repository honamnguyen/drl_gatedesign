import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import minimize
from qtool.scqp import solve_SCQP


#############################################################################
#################################### MISC ###################################
#############################################################################

def tensor(arr):
    '''
    Return the tensor product of all operators in arr
    '''
    prod = arr[0]
    for i in range(1,len(arr)):
        prod = np.kron(prod,arr[i])
    return prod

def qubit_subspace(num_level,num_transmon):
    '''
    Return qubit_indices and qubit_projector for multi-level transmon
    '''
    all_indices = np.array(list(product(range(num_level),repeat=num_transmon))) 
    qubit_indices = np.where(all_indices.max(1)<=1)[0]
    qubit_proj = np.diag((all_indices.max(1)<=1).astype(int))
    return np.ix_(qubit_indices,qubit_indices),qubit_proj

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

zero = np.array([[1,0],[0,0]])
one = np.array([[0,0],[0,1]])
I = np.eye(2)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
X90 = (I-1j*X)/np.sqrt(2)

def common_gate(name):
    gate_dict = {'sqrtZX': 1/np.sqrt(2)*np.array([[ 1, 1j,   0,   0],
                                                    [1j,  1,   0,   0],
                                                    [ 0,  0,   1, -1j],
                                                    [ 0,  0, -1j,   1]]),
                 
                 'ZX90': 1/np.sqrt(2)*np.array([[  1, 0, -1j,  0],
                                                [  0, 1,   0, 1j],
                                                [-1j, 0,   1,  0],
                                                [  0, 1j,  0,  1]]),
                 
                 'ZX': np.array([[ 0, 1,  0,  0],
                                 [ 1, 0,  0,  0],
                                 [ 0, 0,  0, -1],
                                 [ 0, 0, -1,  0]]),
                 
                 'CNOT': np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]),
                 
                 'NOTC': tensor([I,zero])+tensor([X,one]),
                 
                 'SWAP': np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]]), 
                 'X': X, 
                 
                 'X90': X90, 
                 
                 'IX': np.array([[0, 1, 0, 0],
                                 [1, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]]),
                 
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
    qubit_indices,_ = qubit_subspace(num_level,len(thetas))
    Zs[qubit_indices] = tensor(Z_list)
    return Zs

#------------------------------------- Gate fidelity -------------------------------------#

def worst_fidelity(env,method='SCQP-dm-0',bounds=None,init_guess=None,overlap_args=(None,None)):
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
        if theta is not None:
            raise NotImplementedError('Fix virtualZ_on_control, now can perform Z on all transmons')
            virtualZI = virtualZ_on_control(theta,num_level=3)
            correction = tensor([virtualZI,virtualZI.conj()])
        else:
            correction = np.eye(env.get_state().shape[0])
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
    evolved_basis,target_basis = correction@env.get_state(),env.target_state

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
                 target_pauli.T.conj() @ evolved_pauli) + c*np.eye(len(b))
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

###########################################################################
################################## PULSE ##################################
###########################################################################

#---------------------------- Common pulses ------------------------------#

def drag(x,amp,beta,sig):
    gauss = amp*np.exp( -(x-x.max()/2)**2 / (2*sig**2) )
    return np.array([gauss,beta*( -(x-x.max()/2) / (sig**2) )*gauss]).T

def cr1(x_gauss,x_const,amp,sig,phase):
    gauss = amp*np.exp( -(x_gauss-x_gauss.max()/2)**2 / (2*sig**2) )
    const = gauss[len(x_gauss)//2]*x_const
    pulse = np.hstack([gauss[:len(x_gauss)//2],const,gauss[len(x_gauss)//2:]])
    return pulse[:,None]*np.array([np.cos(phase),np.sin(phase)])

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

def plot_pulse(pulse,channel_labels,axs=None,xlim=None):
    '''
    For discretized pulse, input pulse = np.array(actions)[seq]
    '''
    pulse = np.vstack([pulse,np.zeros_like(pulse[0])])
    steps = range(len(pulse))
    
    num_channel = len(channel_labels)
    if axs is None:
        fig, axs = plt.subplots(num_channel,1,sharex=True,figsize=(10,1.5*num_channel))
    if num_channel == 1: axs = [axs]
    for i in range(num_channel):
        axs[i].fill_between(steps,pulse[:,2*i]  ,color='C3',step='post',alpha=0.4)
        axs[i].fill_between(steps,pulse[:,2*i+1],color='C0',step='post',alpha=0.4)
        axs[i].step(steps,pulse[:,2*i]  ,'C3',where='post',label='Re')
        axs[i].step(steps,pulse[:,2*i+1],'C0',where='post',label='Im')
        axs[i].set_ylabel(channel_labels[i])
        axs[i].set_ylim([-1.1,1.1])
        if xlim is None:
            axs[i].hlines(0,-int(0.1*len(pulse)),len(pulse)+int(0.1*len(pulse)),'grey',alpha=0.4)
            axs[i].set_xlim([-int(0.1*len(pulse)),len(pulse)+int(0.1*len(pulse))])
        else:
            axs[i].hlines(0,*xlim,'grey',alpha=0.4)
            axs[i].set_xlim(xlim)
        if i == 0: axs[i].legend(loc='upper right')
        if i == num_channel-1: axs[i].set_xlabel('Time step')
    plt.show()