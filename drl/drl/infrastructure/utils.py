import argparse
from gym_utils import *

def write_dict_to_file(name,d):
    f = open(name+'.txt', 'w')
    for k in d.keys():        
        f.write(f'{k}: {d[k]}\n')  # add comma at end of line
    f.close()

def parser_init(parser):

    ### ----- PARSING ARGUMENTS ----- ###
    parser.add_argument('-logtraindata',action=argparse.BooleanOptionalAction,help='Enable training data logging. Default: None')
    parser.add_argument('-study',default='NoStudy',help='Study name for easy data analysis. Default: NoStudy.')
    
    # rllib
    parser.add_argument('-numworkers',type=int,default=0,help='Number of workers for data collection. Default: 0')    
    
    # environment
    parser.add_argument('-numtransmon',type=int,default=2,help='Number of transmons. Default: 2')
    parser.add_argument('-numlevel',type=int,default=3,help='Number of level in Duffing oscillator. Default: 3')
    parser.add_argument('-numseg',type=int,default=20,help='Number of PWC segments. Default: 20')
    parser.add_argument('-duration',type=float,default=250,help='Pulse max duration in nanoseconds. Default: 250')
    parser.add_argument('-targetgate',default='sqrtZX',help='Target gate to be learned. Default: sqrtZX')
    parser.add_argument('-drivestrength',default='30,300,30,300',help='Drive strengths on two transmons (MHz). Default: 30,300,30,300')
    parser.add_argument('-coupling',type=float,default=1.87,help='Coupling strength between two transmons (MHz). Default: 1.87')
    parser.add_argument('-ctrlnoise',type=float,default=0,help='Noisy control variance in % or in Hz. Default: 0')
    parser.add_argument('-rewardtype',default='worst',help='Reward type: worst or average. Default: worst')
    parser.add_argument('-rewardscheme',default='fnli',help='Reward scheme. Default: fnli')
    parser.add_argument('-fidthreshold',type=float,default=0.999,help='Fidelity threshold to terminate pulse. Default: 0.999')
    parser.add_argument('-worstfidmethod',default='SLSQP-ket-7',help='Method to calculate worst fidelity. Default: SLSQP-ket-7')
    parser.add_argument('-channels',default='2,3,4,5',help='Pulse channels that agent controls. Default: 2,3,4,5')
    parser.add_argument('-subactionscale',type=float,default=0.05,help='Scale of relative change in pulse. Default: 0.05')
    parser.add_argument('-endampwindow',type=float,default=None,help='End amplitude window, reward=0 if outside. Default: None')
    parser.add_argument('-rlstate',default='ket',help='Quantum state representation as input for RL agent. Default: ket')
    
    # networks
    parser.add_argument('-hidsizes',default='800,400,200',help='Hidden layer sizes. Default: 800,400,200')
    parser.add_argument('-layernum',type=int,default=3,help='Number of hidden layers. Default: 3')
    parser.add_argument('-activation',default='relu',help='Activation function. Default: relu')
    parser.add_argument('-qnetscale',type=int,default=3,help='Q-net output scale after tanh. Default: 3')
    
    # training
    parser.add_argument('-gamma',type=float,default=0.99,help='Discount rate gamma. Default: 0.99')
    parser.add_argument('-lr',type=float,default=1e-4,help='Learning rate. Default: 1e-4')
    parser.add_argument('-actorlr',type=float,default=1e-4,help='Actor learning rate. Default: 1e-4')
    parser.add_argument('-criticlr',type=float,default=1e-4,help='Critic learning rate. Default: 1e-4')
    parser.add_argument('-batchsize',type=int,default=64,help='Batch size. Default: 64')
    parser.add_argument('-replaysize',type=int,default=int(1e5),help='Replay buffer size. Default: 100000')
    parser.add_argument('-replayinitial',type=int,default=int(1e4),help='Number of transitions in replay buffer before training starts. Default: 10000')
    parser.add_argument('-testiters',type=int,default=int(1e3),help='Number of iterations between testing. Default: 1000')
    parser.add_argument('-testcount',type=int,default=1,help='Number of tests to average over. Default: 1')
    parser.add_argument('-checkpoints',default=None,help='Checkpoints in terms of average of last 10 test rewards. Default: None')
    parser.add_argument('-initmodel',default=None,help='Initial model to train from. Default: None')
    
    return parser.parse_args()

def transmon_kw(args):
    
    rsdict = {'f':'only-final-step-','l':'local-fidelity-difference-'}    
    # physical setting
    num_transmon = args.numtransmon
    num_level = args.numlevel
    sim_frame_rotation = False
    anharm   = 2*np.pi * np.array([-319.7,-320.2])*MHz
    drive    = 2*np.pi * np.array([float(x) for x in args.drivestrength.split(',')])*MHz
    detune   = 2*np.pi * np.array([115,0])*MHz
    coupling = 2*np.pi * np.array([args.coupling])*MHz

    ctrl_noise = args.ctrlnoise

    # objective
    num_seg = args.numseg
    duration = args.duration
    dt = duration/num_seg*nanosec
    target_gate = args.targetgate

    # state
    rl_state = args.rlstate
    pca_order = (4,2)

    # reward
    reward_type = args.rewardtype
    reward_scheme = rsdict[args.rewardscheme[0]]+args.rewardscheme[1:]
    fid_threshold = args.fidthreshold
    worstfid_method = args.worstfidmethod #has nothing to do with the state

    # action
    channels = [int(x) for x in args.channels.split(',')]
    sub_action_scale = args.subactionscale
    end_amp_window = args.endampwindow
    evolve_method = 'exact'

    kw = initialize_transmon_env('TransmonDuffingSimulator', num_transmon, num_level, sim_frame_rotation,
                                 drive, detune, anharm, coupling, ctrl_noise,
                                 num_seg, dt, target_gate,
                                 rl_state, pca_order,
                                 reward_type, reward_scheme, fid_threshold, worstfid_method,
                                 channels, sub_action_scale, end_amp_window, evolve_method)
    
    return kw
    