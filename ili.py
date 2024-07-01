import time
import argparse
import sys,os
import numpy as np
import torch
import sbi.utils as utils
from sbi.inference.base import infer
import matplotlib.pyplot as plt
from sbi import analysis as analysis
#from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import SNPE, SNRE, SNLE, SNRE_B
from sbi.utils.get_nn_models import posterior_nn
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
#####################################################################################
# get GPU if possible
#if torch.cuda.is_available():
#    print("CUDA Available")
#    device = torch.device('cuda')
#else:
#    print("CUDA Not Available")
device = torch.device('cpu')

#####################################################################################
parser = argparse.ArgumentParser(description="Wrapper")
parser.add_argument("-b", "--observable", required=True,
                    choices=["smf", "smz", "smgf"],
                    nargs="+", help="Choose observables")
parser.add_argument("-n", "--name", required=True, type=str, 
                    help="Name of output folder")
parser.add_argument("-N", "--num_sim", default=50, type=int, 
                    help="Number of simulations per iteration")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose output")
args = parser.parse_args()
#####################################################################################
num_dim    = 7
batch_size = 10
num_sim    = args.num_sim
mcmc_method = 'hmc'
sample_with = 'mcmc'
nde_type    = 'SNLE'
fpath   = '/mnt/home/yjo10/ceph/ILI/SAM/result/' + args.name
sam_path   = '/mnt/home/yjo10/ceph/ILI/SAM/one_click_sam'

try:
    f = open("{}/n_round.txt".format(fpath), 'r')
    n_round = int(f.read())
except FileNotFoundError:
    print("Something went wrong.")
    raise

prior   = utils.BoxUniform(
    low =torch.tensor((0.25, 0.425, 1., 27.5, 0.6, 5e-4, 0.025)),
    high=torch.tensor((4., 6.8, 5., 440., 2.4, 0.008, 0.4)))
"""
prior   = utils.BoxUniform(
    low =torch.tensor((1e-2, 1e-3, 1e-2, 1, 1e-1, 1e-8, 1e-4)),
    high=torch.tensor((1e2, 1e2, 1e2, 1e3, 10, 0.1, 10)))
"""
"""
tau_star_0     [0.25, 4]
epsilon_SN_0   [6.8, 0.425]
alpha_rh       [5, 1]
f_eject_thresh [440, 27.5]
YIELD          [2.4, 0.6]
f_Edd_radio    [0.008, 5e-4]
f_return       [0.4, 0.025]
"""

observation = []
if 'smf' in args.observable:
    smf_obs =\
            np.load('/mnt/home/yjo10/ceph/ILI/SAM/observation/bernardi13_intp.npy')[:,1]
    smf_obs = np.log10(smf_obs)
    observation.extend(smf_obs)

if 'smz' in args.observable:
    smz_obs =\
            np.load('/mnt/home/yjo10/ceph/ILI/SAM/observation/smz.npy')[:,1]
    observation.extend(smz_obs)

if 'smgf' in args.observable:
    smgf_obs =\
            np.load('/mnt/home/yjo10/ceph/ILI/SAM/observation/smfgas.npy')[:,1]
    observation.extend(smgf_obs)

observation = np.array(observation)
observation = torch.tensor(observation, dtype=torch.float).to(device)


theta = np.load('{}/params/params_{}.npy'.format(fpath,n_round))


x = np.zeros((num_sim, len(observation)))
start = 0
end   = 0
if 'smf' in args.observable:
    end += len(smf_obs)
    # should I log it?
    x[:,start:end] = np.load('{}/sam/smf_{}.npy'.format(fpath,n_round))
    start += len(smf_obs)

if 'smz' in args.observable:
    end += len(smz_obs)
    x[:,start:end] = np.load('{}/sam/smz_{}.npy'.format(fpath,n_round))
    start += len(smz_obs)

if 'smgf' in args.observable:
    end += len(smgf_obs)
    x[:,start:end] = np.load('{}/sam/smgf_{}.npy'.format(fpath,n_round))
    start += len(smgf_obs)


print("Shape of X and observation=", x.shape, observation.shape)

theta = torch.tensor(theta, dtype=torch.float).to(device)
x     = torch.tensor(x, dtype=torch.float).to(device)

print ("Data loaded!")


try:
    os.system("mkdir {}/posterior".format(fpath))
    os.system("mkdir {}/ili".format(fpath))
except:
    pass
#####################################################################################



try:
    f = open("{}/ili/ILI.pkl".format(fpath), 'rb')
    ILI = pickle.load(f)
    print("ILI loaded.")
    f.close()
    #if ILI.get_round()+1 != n_round:
        #print("Something's wrong!!")
        #raise
except IOError:
    n_round = 0
    if nde_type == 'SNPE':
        ILI = SNPE(prior=prior)
    elif nde_type == 'SNLE':
        ILI = SNLE(prior=prior)
    print("ILI initialized.")




## Load proposal which is the posterior
if nde_type == 'SNPE':
    try:
        f = open("{}/posterior/{}.pkl".format(fpath,n_round-1),
                 'rb')
        proposal = pickle.load(f)
        proposal   = proposal.set_default_x(observation)
        f.close()
        print("Posterior loaded.")
    except IOError:
        proposal = prior
        print("Posterior is set to the prior.")
    nde        = ILI.append_simulations(theta, x, proposal=proposal)\
            .train(training_batch_size=batch_size)
else:
    nde        = ILI.append_simulations(theta, x).train(training_batch_size=batch_size)

posterior      = ILI.build_posterior(nde, sample_with=sample_with, mcmc_method=mcmc_method)
_              = posterior.set_default_x(observation)
samples        = posterior.sample((num_sim,))
try:
    samples  = samples.cpu().detach().numpy()
except AttributeError:
    samples  = samples.numpy()

## Save ILI and posterior
save_object(ILI, '{}/ili/ILI.pkl'.format(fpath))
save_object(posterior, '{}/posterior/{}.pkl'.format(fpath,n_round))
print ("\nILI module and posterior saved.")

## Save parameters
print("\n{}".format(ILI.get_round()))
print("\n{}".format(ILI._data_round_index))

n_round = n_round + 1
#n_round = ILI.get_round() + 1
np.save("{}/params/params_{}".format(fpath,n_round), samples)
with open("{}/n_round.txt".format(fpath), 'w') as f:
    f.write(str(int(n_round)))
print ("Proposal saved (params{})!".format(n_round))


