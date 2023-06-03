import argparse, json

def write_dict_to_file(name,d):
    f = open(name+'.txt', 'w')
    for k in d.keys():        
        f.write(f'{k}: {d[k]}\n')  # add comma at end of line
    f.close()

def parser_init(parser):

    ### ----- PARSING ARGUMENTS ----- ###
    parser.add_argument('-study',default='NoStudy',help='Study name for easy data analysis. Default: NoStudy.')
    parser.add_argument('-restart',action='store_true',help='Restart run from latest checkpoint. Default: None')
    
    # for inference
    parser.add_argument('-run',default=None,help='Name fragments of run. Default: None.')
    parser.add_argument('-chpt',default=None,help='Checkpoint. Default: None.')

    # rllib
    parser.add_argument('-numworkers',type=int,default=0,help='Number of workers for data collection. Default: 0')    
    parser.add_argument('-numgpus',type=int,default=0,help='Number of GPUs. Default: 0')    
    parser.add_argument('-numiter',type=int,default=300,help='Number of iteration. Default: 300')    
    # parser.add_argument('-stepsperiter',type=int,default=100,help='Timesteps per iteration. Default: 100')    
    parser.add_argument('-evaluationinterval',type=int,default=30,help='Number of train() steps between evaluation/save. Default: 30')
    parser.add_argument('-testcount',type=int,default=1,help='Number of tests to average over. Default: 1')
    # parser.add_argument('-numbesteps',type=int,default=10,help='Number of best episodes to save. Default: 10')
        
    # environment
    parser.add_argument('-numqubits',type=int,default=1,help='Number of qubits. Default: 1')
    parser.add_argument('-maxdepth',type=int,default=5,help='Maximum gate depth. Default: 5')
    parser.add_argument('-rewardscheme',default='dense',help='Reward scheme. Default: dense')
    parser.add_argument('-depthpenalty',type=float,default=0,help='Depth penalty. Default: 0')
    parser.add_argument('-kltarget',type=float,default=0,help='Target KL divergence. Default: 0')
    parser.add_argument('-klshots',type=int,default=20000,help='Shots used to estimate KL divergence. Default: 20000')
    parser.add_argument('-kltrials',type=int,default=5,help='Trials used to estimate KL divergence. Default: 5')
    parser.add_argument('-klbins',type=int,default=75,help='Number of bins used to estimate KL divergence. Default: 75')
    parser.add_argument('-rlstate',default='2D',help='State type for RL agent. Default: 2D')
    parser.add_argument('-q2gates',default='CX',help='Two qubit entangeling gates. Default: CX')
    parser.add_argument('-q1gates',default='H_RX_RZ',help='Single qubit gates. Default: H_RX_RZ')
    parser.add_argument('-method',default='precalc',help='Single qubit gates. Default: precalc')
    
    # networks
    parser.add_argument('-hiddens',default='16,16',help='Hidden layer sizes. Default: 16,16')
    # parser.add_argument('-layernum',type=int,default=3,help='Number of hidden layers. Default: 3')
    parser.add_argument('-activation',default='tanh',help='Activation function. Default: tanh')
    
    # training
    parser.add_argument('-seed',type=int,default=1,help='Seed. Default: None')
    parser.add_argument('-gamma',type=float,default=0.99,help='Discount rate gamma. Default: 0.99')
    parser.add_argument('-lr',type=float,default=5e-5,help='Learning rate. Default: 5e-5')
    parser.add_argument('-trainbatch',type=int,default=4000,help='Train batch size. Default: 4000')
    parser.add_argument('-sgdminibatch',type=int,default=128,help='Train batch size. Default: 128')
    

    
    parser.add_argument('-actorlr',type=float,default=1e-4,help='Actor learning rate. Default: 1e-4')
    parser.add_argument('-criticlr',type=float,default=1e-4,help='Critic learning rate. Default: 1e-4')
    parser.add_argument('-replaysize',type=int,default=int(1e5),help='Replay buffer size. Default: 100000')
    parser.add_argument('-replayinitial',type=int,default=int(1e4),help='Number of transitions in replay buffer before training starts. Default: 10000')
    parser.add_argument('-checkpoints',default=None,help='Checkpoints in terms of average of last 10 test rewards. Default: None')
    parser.add_argument('-initmodel',default=None,help='Initial model to train from. Default: None')
    parser.add_argument('-td3twinq',default=False, action='store_true',help='TD3 modification, twin Q networks. Default: False')
    parser.add_argument('-td3policydelay',type=int,default=1,help='TD3 modification, delay policy update. Default: 1')
    parser.add_argument('-td3smoothtarget',default=False,action='store_true',help='TD3 modification, smooth target policy. Default: False')
    
    # ray tune
    # parser.add_argument('-tunecpu',type=int,default=1,help='Tune CPUs. Default: 1')
    parser.add_argument('-tunegpus',type=int,default=0,help='Tune GPUs. Default: 0')

    return parser

def get_kw(args):
    
    # physical setting
    num_qubits = args.numqubits
#     if args.entgate in ['CX','CY','CZ']:
#         gateset = [['H','RX','RZ'],[args.entgate]]
    gateset = [args.q1gates.split('_'),args.q2gates.split('_')]
    print(gateset)
    method = args.method
    
    # state
    rl_state = args.rlstate

    # reward
    max_depth = args.maxdepth
    reward_scheme = args.rewardscheme
    depth_penalty = args.depthpenalty
    # reward_scheme = rsdict[args.rewardscheme[0]]+args.rewardscheme[1:]
    kl_target = args.kltarget
    kl_shots = args.klshots
    kl_trials = args.kltrials
    kl_bins = args.klbins

    kw = {
        'rl_state': rl_state, # 2D
        'step_params': {
            'max_depth': max_depth,
            'reward_scheme': reward_scheme, # dense
            'depth_penalty': depth_penalty,
            'kl_shots': kl_shots,
            'kl_trials': kl_trials,
            'num_bins': kl_bins,
            'kl_target': kl_target,
        },
        'sim_params': {
            'name': 'PQC',
            'num_qubits': num_qubits,
            'gateset': gateset,
            'method': method,
        },
    }

    return kw
    