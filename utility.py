import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#############################################################################
################################## GENERAL ##################################
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