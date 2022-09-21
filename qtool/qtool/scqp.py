'''scqp.py'''
import numpy as np
from scipy.optimize import brentq

def func(lam,s,c):
    return 1 - (c**2/(lam-s)**2).sum()
def poly(t,d,c):
    return t**4 + 2*d*t**3 + (d**2-1)*t**2 - 2*c**2*d*t - c**2*d**2

def qps_nnz(s,c,eps):
    # print('s:',s)
    # print('c,c@c:',c,c@c)
    if len(s) == 1:
        return -c/s
    elif len(s) == 2:
        d1 = s[1]-s[0]
        t1 = brentq(poly, abs(c[0]), 1, args = (d1,c[0]))
        return c/(1-t1-s)
    else:    
        d1 = s[1]-s[0]
        d2 = s[-1]-s[0]
        t1 = brentq(poly, abs(c[0]), 1, args = (d1,c[0]))
        t2 = brentq(poly, abs(c[0]), 1, args = (d2,c[0]))
        # print('d1,d2:',d1,d2)
        # print('t1,t2:',t1,t2)
        # print(abs(t1-t2))
        # 2nd condition is when solution is at the boundary 1-t1
        if abs(t1-t2)<eps or abs(func(1-t1,s,c))<eps:
            return c/(1-t1-s)
        else:
            # print('1-t1:',1-t1)
            # for lam in np.linspace(1-t1,1-t2,10):
            #     print('func:',func(lam,s,c))                
            lam = brentq(func, 1-t1, 1-t2, args = (s,c))
            return c/(lam-s)
    
def solve_SCQP(A,b):
    '''
    Spherically Constrained Quadratic Programming
    Input: A and b, A must be positive semidefinite
    Output x minimizes 0.5*x@A@x + b@x s.t. x@x = 1
    '''
    eps = 1e-10
    
    x = np.zeros(len(A))
    eigval,U = np.linalg.eigh(A)
    norm_b = np.sqrt(b@b)

    # print(eigval)
    s = (eigval-eigval[0])/norm_b + 1
    c = U.T@b/norm_b
    set_I = abs(c[1:])>eps

    c_tilde = c[1:][set_I]
    s_tilde = s[1:][set_I]
    
    # print('small #:',abs(c[0]))
    if abs(c[0])<eps:
        d = (c_tilde**2/(s_tilde-1)**2).sum()
        if d<1 and s_tilde[0]>1:
            x[0] = np.sqrt(1-d) #this could be plus or minus
            x[1:][set_I]= c_tilde/(1-s_tilde)
        else:
            x[1:][set_I] = qps_nnz(s_tilde-s_tilde[0]+1,c_tilde,eps)
    else:
        tmp = qps_nnz(np.hstack([s[0],s_tilde]),np.hstack([c[0],c_tilde]),eps)
        x[0] = tmp[0]
        x[1:][set_I] = tmp[1:]
    return U@x