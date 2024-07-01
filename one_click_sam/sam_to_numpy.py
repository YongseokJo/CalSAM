import numpy as np
import argparse
from scipy.interpolate import splrep, BSpline
from uncertainty_utils import add_uncertainty
import torch
from scipy.stats import ortho_group
import os

#####################################################################################
parser = argparse.ArgumentParser(description="Wrapper")
parser.add_argument("-n", "--name", required=True, type=str, 
                    help="Name of output folder")
parser.add_argument("-s", "--simulation", required=True, type=str, 
                    help="Simulation Type")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose output")
args = parser.parse_args()
#####################################################################################

def check_file_exists(file_name):
    return os.path.exists(file_name)

### Load what round it is
fpath = '/mnt/home/yjo10/ceph/ILI/SAM/result/' + args.name

try:
    f = open("{}/n_round.txt".format(fpath),"r")
    n_round = int(f.read())
except:
    n_round = 0


### Load params proposed by ILI
params = np.load("{}/params/params_{}.npy".format(fpath, n_round))

newsmf   = []
newsmz   = []
newsmgas = []

for i in range(params.shape[0]):
    print(i)
    smf         = np.loadtxt("./result/{}/{}/smf.dat".format(args.name,i))
    smf[smf==0] = 1e-10
    x, y        = smf[:,0], np.log10(smf[:,1])

    tck    = splrep(x, y, s=0)
    xnew   = np.linspace(9,12,8) # Bernadi13
    newsmf.append(BSpline(*tck)(xnew))

    smz   = np.loadtxt("./result/{}/{}/mstar_Zstar.dat".format(args.name,i))
    x, y  = smz[:,0], smz[:,2]
    tck   = splrep(x, y, s=0)
    xnew  = np.array([9.11, 9.31, 9.51, 9.72, 9.91, 10.11, 10.31, 10.51,
                      10.72, 10.91, 11.11]) # Gallazzi
    newsmz.append(BSpline(*tck)(xnew))

    smgas = np.loadtxt("./result/{}/{}/mstar_fcold.dat".format(args.name,i))
    x, y  = smgas[:,0], smgas[:,2]
    try:
        tck   = splrep(x, y, s=0)
        xnew  = np.array([9.15, 9.49, 10., 10.52]) # Boselli14
        newsmgas.append(BSpline(*tck)(xnew))
    except TypeError:
        newsmgas.append(np.array([-10,-10,-10,-10]))


newsmf   = np.vstack(newsmf)
newsmz   = np.vstack(newsmz)
newsmgas = np.vstack(newsmgas)
print(newsmf.shape)

#np.save("./result/smf_{}_org".format(n_round), newsmf)
#np.save("./result/smz_{}_org".format(n_round), newsmz)
#np.save("./result/smgas_{}_org".format(n_round), newsmgas)

np.save("{}/sam/smf_{}".format(fpath, n_round), newsmf)
np.save("{}/sam/smz_{}".format(fpath, n_round), newsmz)
np.save("{}/sam/smgf_{}".format(fpath,n_round), newsmgas)


subprocess.run(["rm -rf","./result/{}".format(args.name)], capture_output=True)


"""
# Example usage:
file_name = "covariance/random1.npy"  # Change this to the name of the file you want to check
if check_file_exists(file_name):
    print(f"File '{file_name}' exists.")
    cov = np.load(file_name)
else:
    dial = 5e-2
    dim = newsmf.shape[1]
    O = ortho_group.rvs(dim=dim)
    D = np.eye(dim)*np.random.random(size=dim)*dial*2
    cov = np.matmul(O,np.matmul(D,O.T))
    for i in range(dim):
        pass
    #detA = np.linalg.det(cov)
    np.save("covariance/random1", cov)


"""
"""
cov = torch.eye(newsmf.shape[1])*0.2
for i in range(5,8):
    cov[i,i] *= 1e-2
newsmf = add_uncertainty(newsmf, covariance_matrix=cov,
                batch_size=newsmf.shape[0])
#newsmf = np.log10(newsmf)
"""
"""
cov    = torch.tensor(cov)
newsmf = add_uncertainty(newsmf, covariance_matrix=cov,
                batch_size=newsmf.shape[0])

cov    = torch.eye(newsmz.shape[1])*0.8
newsmz = add_uncertainty(newsmz, covariance_matrix=cov,
                batch_size=newsmz.shape[0])

newsmgas = add_uncertainty(newsmgas, covariance_matrix=torch.eye(newsmgas.shape[1]),
                batch_size=newsmgas.shape[0])

np.save("./result/smf_{}".format(n_round), newsmf)
np.save("./result/smz_{}".format(n_round), newsmz)
np.save("./result/smgas_{}".format(n_round), newsmgas)
"""
