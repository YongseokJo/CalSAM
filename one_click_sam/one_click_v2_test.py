#!/mnt/sw/nix/store/6a11b1qfnfwxjzs8q4xnlid1z3mvkz9l-python-3.9.12-view/bin/python
from scipy.interpolate import splrep, BSpline
import numpy as np
import os
import one_click_SAM as ocs


sim       = 'TNG300'
params    = (1.0,1.7,3.0,110.0,1.2,2.0E-03,0.1)
para_path ='/mnt/home/yjo10/ceph/ILI/SAM/one_click_sam/test3_1/'
res_path  =\
        '/mnt/home/yjo10/ceph/ILI/SAM/one_click_sam/test3_1/'

try:
    os.system("mkdir {}".format(res_path))
except:
    pass

ocSAM = ocs.one_click_SAM(para_path, res_path, 0, sim)
ocSAM.run(params)


smf         = np.loadtxt("{}/smf.dat".format(res_path))
smf[smf==0] = 1e-10
x, y        = smf[:,0], np.log10(smf[:,1])
tck    = splrep(x, y, s=0)
xnew   = np.linspace(9,12,8) # Bernadi13
newsmf = BSpline(*tck)(xnew)

smz   = np.loadtxt("{}/mstar_Zstar.dat".format(res_path))
x, y  = smz[:,0], smz[:,2]
tck   = splrep(x, y, s=0)
xnew  = np.array([9.11, 9.31, 9.51, 9.72, 9.91, 10.11, 10.31, 10.51,
                      10.72, 10.91, 11.11]) # Gallazzi
newsmz = BSpline(*tck)(xnew)

smgas = np.loadtxt("{}/mstar_fcold.dat".format(res_path))
x, y  = smgas[:,0], smgas[:,2]
tck   = splrep(x, y, s=0)
xnew  = np.array([9.15, 9.49, 10., 10.52]) # Boselli14
newsmgas = BSpline(*tck)(xnew)

np.save("{}/smf_{}.npy".format(res_path,sim), newsmf)
np.save("{}/smz_{}.npy".format(res_path,sim), newsmz)
np.save("{}/smgas_{}.npy".format(res_path, sim), newsmgas)
