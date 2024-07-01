import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import pickle
from utils_plot import corner_plot


#####################################################################################
parser = argparse.ArgumentParser(description="Wrapper")
parser.add_argument("-d", "--dir", required=True, type=str,
                    help="output dir")
parser.add_argument("-ob1", "--observable1", required=True,
                    choices=["smf", "smz", "smgf"],
                    nargs="+", help="Choose observables")
parser.add_argument("-ob2", "--observable2", required=True, 
                    choices=["smf", "smz", "smgf"],
                    nargs="+", help="Choose observables")
parser.add_argument("-n1", "--num_sim1", default=10, type=int, 
                    help="Number of simulations per iteration")
parser.add_argument("-n2", "--num_sim2", default=10, type=int, 
                    help="Number of simulations per iteration")
parser.add_argument("-N", "--num_sample", default=1000, type=int, 
                    help="Number of simulations per iteration")
parser.add_argument("-s", "--sampling", default=False, action="store_true", 
                    help="Number of simulations per iteration")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose output")
args = parser.parse_args()
#####################################################################################


observations = []
ifname = []
for obs_tmp in [args.observable1, args.observable2]:
    name_tmp = "" 
    observation = []
    if 'smf' in obs_tmp:
        smf_obs =\
                np.load('/mnt/home/yjo10/ceph/ILI/SAM/observation/bernardi13_intp.npy')[:,1]
        smf_obs = np.log10(smf_obs)
        observation.extend(smf_obs)
        name_tmp += 'smf_'

    if 'smz' in obs_tmp:
        smz_obs =\
                np.load('/mnt/home/yjo10/ceph/ILI/SAM/observation/smz.npy')[:,1]
        observation.extend(smz_obs)
        name_tmp += 'smz_'

    if 'smgf' in obs_tmp:
        smgf_obs =\
                np.load('/mnt/home/yjo10/ceph/ILI/SAM/observation/smfgas.npy')[:,1]
        observation.extend(smgf_obs)
        name_tmp += 'smgf_'
    observations.append(observation)
    ifname.append(name_tmp[:-1])



print(ifname)
n_sample = args.num_sample
n_round  = [args.num_sim1   , args.num_sim2]; 
bandwidth = [[0.2, 0.3, 0.3, 10, 0.1, 1e-3, 0.02], 
             [0.1, 0.3, 0.1, 40, 0.1, 1e-3, 0.01]]
new_samples = []

### SAMPLING
if args.sampling:
    sample_with = 'mcmc'
    mcmc_method = 'hmc'
    for i in range(len(ifname)):
        fname    = ifname[i]
        num      = n_round[i]
        fpath    = f"./result/{fname}" 
        ## load
        params      = np.load(f"{fpath}/params/params_{num}.npy")
        f           = open("{}/ili/ILI.pkl".format(fpath), 'rb')
        ILI         = pickle.load(f)
        theta       = np.load(f'{fpath}/params/params_{num}.npy')
        x           = np.load(f'{fpath}/sam/smgf_{num}.npy')
        ## sampling
        theta       = torch.tensor(theta, dtype=torch.float)#.to(device)
        x           = torch.tensor(x, dtype=torch.float)#.to(device)
        nde         = ILI.append_simulations(theta, x)
        posterior   = nde.get_posterior()
        posterior.set_default_x(observations[i])
        new_sample  = posterior.sample((n_sample,))
        np.save(f"{fpath}/params/params_{num}_N{n_sample}",new_sample)
        new_samples.append(new_sample)
else:
    new_samples = None


fig, axes = corner_plot(ifname, n_round, bandwidth, new_samples)

plt.savefig(f"./paperplot/{args.dir}.png", dpi=200, bbox_inches='tight')
plt.close()
