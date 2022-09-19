from scipy.optimize import Bounds, minimize

def calibration_loss_1channel(x, pulse_shape, num_seg, channel, U_target, qubit_indices, sim):
    '''Calibrate pulse parameters and initial Z angles'''
    pulse = pulse_shape(num_seg,*x[:-1])    
    total_pulse = np.zeros([len(pulse),8])   
    total_pulse[:,2*channel:2*channel+2] = Z_shift(pulse,x[-1])
    return -sim.pulse_average_fidelity(total_pulse, U_target, qubit_indices)

def calibration(x0, loss_func, bounds, args=()):
    '''Simple calibration'''
    res = minimize(loss_func, x0, method='nelder-mead',args=args, options={'disp': True}, bounds=Bounds(*bounds))
    return res.x,res.fun