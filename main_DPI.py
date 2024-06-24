import torch
import numpy as np
import sys
from utils.training_loop import start_train
import argparse
import pprint as pp
from envs import env_dict


# ====================================
#           Initializations
# ====================================

torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
np.random.seed(np_seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

# ======================================
#           Set Hyperparameters
# ======================================
parser = argparse.ArgumentParser(description='Argument parser')

# The environment. Current options
parser.add_argument('--env_name',                   default='AnalyticEnv', help='The environment.')

parser.add_argument('--torch_seed',            default=torch_seed)
parser.add_argument('--np_seed',               default=np_seed)
parser.add_argument('--max_epochs',            default=int(50000))
parser.add_argument('--device',                default=device)
parser.add_argument('--print_rate',            default=100, help='How often to print to console and log')

parser.add_argument('--batch_size',            default=100)
parser.add_argument('--ns',                    default=100, help='Network size')
parser.add_argument('--1_lr',                  default=1e-4)
parser.add_argument('--2_lr',                  default=1e-4)
parser.add_argument('--3_lr',                  default=1e-4)
parser.add_argument('--betas',                 default=(0.5, 0.9), help='Adam only')
parser.add_argument('--weight_decay',          default=1e-3)
parser.add_argument('--act_func_1',            default='softplus', help='Activation function for network_1')
parser.add_argument('--act_func_2',            default='tanh', help='Activation function for network_2')
parser.add_argument('--act_func_3',            default='tanh', help='Activation function for control')
parser.add_argument('--hh',                    default=1, help='ResNet step-size')


bool_logging = True
parser.add_argument('--do_logging',            default=bool_logging)


# Parse args, get the environment, and set some arguments automatically
args = vars(parser.parse_args())
env = env_dict[args['env_name']](device=device)
args['env'] = env
args['experiment_name'] = '_' + str(env.name)

# ==================================
#           Start training
# ==================================
pp.pprint(env.info_dict)
pp.pprint(args)
the_logger = start_train(args)
