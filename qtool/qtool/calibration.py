import numpy as np
from scipy.optimize import Bounds, minimize
from qtool.utility import Z_shift, qiskit_cr_pulse, qiskit_drag_pulse
import time
import pickle, os

nanosec = 1e-9
############# THEORY #################
def calibration_loss_1channel(x, pulse_shape, num_seg, channel, U_target, qubit_indices, sim):
    '''Calibrate pulse parameters and initial Z angles'''
    pulse = pulse_shape(num_seg,*x[:-1])    
    total_pulse = np.zeros([len(pulse),8])   
    total_pulse[:,2*channel:2*channel+2] = Z_shift(pulse,x[-1])
    return -sim.pulse_average_fidelity(total_pulse, U_target, qubit_indices)

# def calibration(x0, loss_func, bounds, args=()):
#     '''Simple calibration'''
#     res = minimize(loss_func, x0, method='nelder-mead',args=args, options={'disp': True}, bounds=Bounds(*bounds))
#     return res.x,res.fun

def calibration_loss(x, pulse_func, channel, U_target, qubit_indices, sim, method):
    pulse = pulse_func(x)
    total_pulse = np.zeros([len(pulse),4],dtype=np.complex128)   
    total_pulse[:,channel] = pulse
    return -sim.pulse_average_fidelity(total_pulse, U_target, qubit_indices, method=method)  

def calibration(loss_func, bounds, args=(), x0=None, display=False):
    '''Simple calibration'''
    if x0 is None:
        x0 = [np.random.uniform(*bounds[:,i]) for i in range(bounds.shape[1])]
    start = time.time()
    res = minimize(loss_func, x0, method='nelder-mead',args=args, options={'disp': display}, bounds=Bounds(*bounds))
    print(f'Time taken: {time.time()-start:.1f}s, f = {res.fun:.6f}, x = {res.x}')
    return res.x,res.fun

################### EXPERIMENT ######################
def rabi_experiments(dt, channel, var_range, name, use_cache, exp_cr_params, cr_times,
                     exp_cancel_params=None, cancel_channel=None, plot=False):
    file_path = 'data/'+name+'.pkl'
    if use_cache:
        print(f'\nLoad data from {file_path}\n')
        with open(file_path, 'rb') as fp:
            data = pickle.load(fp)
        return data['var_range'], data['cr_durations'], data['Us']                     
    else:
        Us = []
        if type(exp_cr_params) is list:
            for cr_params in exp_cr_params:
                print('\n cr:',cr_params)
                U, cr_durations = rabi_experiment(dt, channel, cr_params, cr_times, 
                                                  exp_cancel_params, cancel_channel, plot)
                Us.append(U)
        elif type(exp_cancel_params) is list:
            for cancel_params in exp_cancel_params:
                print('\n cancel:',cancel_params)
                U, cr_durations = rabi_experiment(dt, channel, exp_cr_params, cr_times, 
                                                  cancel_params, cancel_channel, plot)
                Us.append(U)
        Us = np.array(Us)
        # save data
        print(f'\nSave data to {file_path}\n')
        data = {'Us': Us, 'cr_durations': cr_durations, 'var_range': var_range}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as fp:
            pickle.dump(data, fp)
        return var_range, cr_durations, Us
    
def rabi_experiment(dt, channel, cr_params, cr_times, 
                    cancel_params=None, cancel_channel=None, plot=False):
    '''
    Get evolution map for multiple CR interation's durations
    '''
    cr_rf_sigma_ratio = (cr_params['duration'] - cr_params['width']) / (2 * cr_params['sigma'])
    Us, cr_durations = [], []

    for width in cr_times:
        tomo_cr_params = cr_params.copy()
        tomo_cr_params['duration'] = int(width + 2 * cr_rf_sigma_ratio * cr_params['sigma'])
        tomo_cr_params['width'] = width
        cr_durations.append(tomo_cr_params['duration'])
        print(f'duration: {cr_durations[-1]*dt/nanosec:.0f}ns')

        cr_pulse = qiskit_cr_pulse(**tomo_cr_params)
        total_pulse = np.zeros([len(cr_pulse),4], dtype=np.complex128)
        total_pulse[:,channel] = cr_pulse

        # active cancellation
        if cancel_params:
            tomo_cancel_params = cancel_params.copy()
            tomo_cancel_params['duration'] = tomo_cr_params['duration']
            tomo_cancel_params['width'] = tomo_cr_params['width']
            cancel_pulse = qiskit_cr_pulse(**tomo_cancel_params)[:,[1]]
            total_pulse[:,cancel_channel] = cancel_pulse

        if plot:
            plot_pulse(total_pulse,['d0','u01','d1','u10'],ylim='adjusted')

        U = duffing_sim.evolve(total_pulse.view(np.float64))[-1]
        Us.append(U)
    return np.array(Us), np.array(cr_durations)