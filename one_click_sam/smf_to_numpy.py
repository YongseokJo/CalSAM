import numpy as np
from scipy.interpolate import splrep, BSpline


### Load what round it is
fpath = '/mnt/home/yjo10/ceph/ILI/SAM/data'
try:
    f = open("{}/n_round.txt".format(fpath),"r")
    n_round = int(f.read())
except:
    n_round = 0

### Load params proposed by ILI
params = np.load("/mnt/home/yjo10/ceph/ILI/SAM/data/params_{}.npy".format(n_round))

ynew = []

for i in range(params.shape[0]):
    smf         = np.loadtxt("./result/parallel/{}/smf.dat".format(i))
    smf[smf==0] = 1e-10
    x, y        = smf[:,0], np.log10(smf[:,1])

    tck    = splrep(x, y, s=0)
    #tck_s  = splrep(x, y, s=len(x))
    #xnew   = np.linspace(6,12,15)
    xnew   = np.linspace(9,12,8) # Bernadi13
    ynew.append(BSpline(*tck)(xnew))

ynew = np.vstack(ynew)
print(ynew.shape)

np.save("./result/obs_{}".format(n_round), ynew)



"""
smf         = np.loadtxt("./result/smf_1.dat")
smf[smf==0] = 1e-10
x, y        = smf[:,0], np.log10(smf[:,1])
tck    = splrep(x, y, s=0)
#tck_s  = splrep(x, y, s=len(x))
xnew   = np.linspace(6,12,15)
#xnew   = np.linspace(8,12,8)
ynew   = BSpline(*tck)(xnew)

print(ynew.shape)
np.save("./result/obs_target", ynew)
"""
