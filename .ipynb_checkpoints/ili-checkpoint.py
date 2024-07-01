import time
import sys,os
import numpy as np
import torch
import sbi.utils as utils
from sbi.inference.base import infer
import matplotlib.pyplot as plt
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
import pickle

#####################################################################################
# get GPU if possible
#if torch.cuda.is_available():
#    print("CUDA Available")
#    device = torch.device('cuda')
#else:
#    print("CUDA Not Available")
device = torch.device('cpu')

#####################################################################################
def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
num_dim = 3
num_sim = 50
fpath   = '/mnt/home/yjo10/ceph/ILI/SAM/data'
with open("{}/n_round.txt".format(fpath), 'r') as f:
    n_round = int(f.read())
prior   = utils.BoxUniform(low=-2*torch.ones(num_dim),
                           high=2*torch.ones(num_dim))
observation = torch.zeros(3)
#####################################################################################



try:
    f = open("{}/ILI.pkl".format(fpath), 'rb')
    ILI = pickle.load(f)
    print("ILI loaded.")
    f.close()
    if ILI.get_round()+1 != n_round:
        print("Something's wrong!!")
        raise
except IOError:
    ILI = SNPE(prior=prior)
    print("ILI initialized.")



print ("New data exists!")
theta = np.load('{}/params_{}.npy'.format(fpath,n_round))
x     = np.load('{}/obs_{}.npy'.format(fpath,n_round))
theta = torch.tensor(theta).to(device)
x     = torch.tensor(x).to(device)
print ("Data loaded!")

## Load proposal which is the posterior
try: 
    f = open("{}/posterior/{}.pkl".format(fpath,ILI.get_round()),
             'rb')
    proposal = pickle.load(f)
    f.close()
    proposal = proposal.set_default_x(observation)
    print("Posterior loaded.")
except IOError:
    proposal = prior
    print("Posterior is set to the prior.")


_        = ILI.append_simulations(theta, x, proposal=proposal).train()
posterior  =  ILI.build_posterior()
_          = posterior.set_default_x(observation)
samples    = posterior.sample((num_sim,))
try:
    samples  = samples.cpu().detach().numpy()
except AttributeError:
    samples  = samples.numpy()

## Save ILI and posterior
save_object(ILI, '{}/ILI.pkl'.format(fpath))
save_object(posterior, '{}/posterior/{}.pkl'.format(fpath,n_round))
print ("\nILI module and posterior saved.")

## Save parameters
print("\n{}".format(ILI.get_round()))
print("\n{}".format(ILI._data_round_index))
#n_round = n_round + 1
n_round = ILI.get_round() + 1
np.save("{}/params_{}".format(fpath,n_round), samples)
np.save("{}/n_round".format(fpath),np.array(n_round))
print ("Proposal saved (params{})!".format(n_round))


