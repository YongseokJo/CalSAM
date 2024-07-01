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
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print("CUDA Not Available")
    device = torch.device('cpu')


#####################################################################################
def linear_gaussian(theta):
    return theta + 1.0 + torch.randn_like(theta) * 0.1

def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

num_dim = 3
n_round = 0
num_sim = 50
fpath   = '/mnt/home/yjo10/ceph/IFI/SAM/data'

prior   = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 *
                         torch.ones(num_dim))
observation = torch.zeros(3)
proposal = prior
#####################################################################################

try:
    f = open("{}/IFI.pkl", 'rb')
    inference = pickle.load(f)
    print("IFI loaded.")
    f.close()
    n_round = inference.get_round()
except IOError:
    inference = SNPE(prior=prior)
    samples = prior.sample((num_sim,))
    np.save("{}/params0".format(fpath),samples)
    print("IFI initialized.")

while (True):
    time.sleep(5)
    if os.path.isfile('{}/obs{}.npy'.format(fpath,n_round)):
        print ("New data exists!")
        theta = np.load('{}/params{}.npy'.format(fpath,n_round))
        x     = np.load('{}/obs{}.npy'.format(fpath,n_round))
        theta = torch.tensor(theta).to(device)
        x     = torch.tensor(x).to(device)
        print ("Data loaded!")

        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train()
        posterior =\
                inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(observation)
        samples  = posterior.sample((num_sim,))
        try: 
            samples  = samples.cpu().detach().numpy()
        except AttributeError:
            samples  = samples.numpy()
        n_round = inference.get_round() + 1
        np.save("{}/params{}".format(fpath,n_round), samples)
        print ("Proposal saved!")
        save_object(inference, '{}/IFI.pkl'.format(fpath))
        print ("Inference module saved.")
    else:
        pass



