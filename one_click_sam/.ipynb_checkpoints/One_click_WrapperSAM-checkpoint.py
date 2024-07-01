#!/mnt/sw/nix/store/6a11b1qfnfwxjzs8q4xnlid1z3mvkz9l-python-3.9.12-view/bin/python

### rusty
#SBATCH --job-name=OneClick
#SBATCH -p cca --qos=cca
#SBATCH -n 1
#SBATCH --time=168:00:00
#SBATCH --export=ALL


# In[1]:


import numpy as np
import pandas as pd
import os

import astropy.units as u
from   astropy.cosmology import FlatLambdaCDM
from   scipy.optimize    import fsolve
cosmo = FlatLambdaCDM(Om0=0.307, Ob0=0.048, H0=67.8)  ## Matching BolshoiP


# In[2]:


### parameters to vary
sam_path  ='/mnt/ceph/users/lyung/sam_code/sc-sam.gureft/'
lib_path  ='/mnt/ceph/users/lyung/sam_code/gflib/'
out_path  ='/mnt/ceph/users/lyung/One_click_WrapperSAM/Testrun/'
tree_path ='/mnt/ceph/users/lyung/Nbody/VSMDPL_TreesPrep/25_25/gridtree.dat'
cat_path  ='/mnt/ceph/users/lyung/Nbody/VSMDPL_TreesPrep/25_25/halocat.dat'


# In[3]:


def run_SAM(alpha_rh, tau_star_0,
            sam_path = sam_path, 
            lib_path = lib_path, 
            out_path = out_path, 
            tree_path= tree_path,
            cat_path = cat_path):
    
    ### name of the parameter file
    para_name = 'test.param'
    
    ### create a parameter file
    os.system('python paragen.py %s %s %s %s %s %s %s'%(para_name, out_path, 
                                                        lib_path, tree_path, 
                                                        'VSMDPL', 
                                                        alpha_rh, tau_star_0))
    ### run the SAM with that parameter file
    os.system('cd %s && ./gf %s%s &> %s%s'%(sam_path,out_path,para_name,out_path,"output.txt"))

    galprop = weight_galprop(cat_path, out_path, 
                   'galprop_150-150.dat', 'haloprop_150-150.dat', grid = 25., realization = 25.)
    
    SMF = make_smf(galprop, Mmin=4.0, Mmax=14.0, step=0.25)
    write_results(out_path,'smf.dat',SMF,xhead="log M*",yhead="phi")
    

    

    
def read_data(x):
    with open(x, 'r') as dat:
        lines   = dat.readlines()
        header  = [ line.strip().split(" ")[2] for line in lines if line[0] == "#"]
    return pd.read_table(x, comment="#", delim_whitespace=True, names=header)


def sigma(M_vir):
    y = 1e12/M_vir
    term1 = 17.967209*y**0.409964
    term2 = 1 + 1.179455*y**0.210397 + 6.192103*y**0.320073
    return term1/term2

def g(z):                           #eq.28 in RP+2016
    a  = cosmo.scale_factor(z)
    om = cosmo.Om(z)
    ol = cosmo.Ode(z)
    return (0.4 * om * a)/(np.power(om,0.571428571)-ol+(1+om*0.5)*(1+ol*0.014285714))

def D(z):
    return g(z)/g(0)

def f(M_vir, z):  # following RP+2016
    A = lambda x: 0.143873 - 0.0112026*x  + 0.00253025*(x**2)
    a = lambda x: 1.35053  + 0.0681399*x  - 0.00593122*(x**2)
    b = lambda x: 3.11271  - 0.0770555*x  - 0.0134007*(x**2)
    c = lambda x: 1.1869   + 0.00875954*x - 0.000739608*(x**2)
    
    sig = sigma(M_vir)*D(z)
    return A(z) * (((sig/b(z))**(-a(z))) +1 ) * np.exp(-c(z)/(sig**2))

def diff_sigma(M_vir):
    return (np.log(1.0/sigma(1.002*M_vir))-np.log(1.0/sigma(0.998*M_vir)))/(np.log(1.002*M_vir)-np.log(0.998*M_vir))
    

def hmf(M_vir,z):
    rho_m = (cosmo.Om(0)*cosmo.critical_density(0).to(u.solMass/(u.Mpc)**3)).value/cosmo.h**2
    return f(M_vir,z) * (rho_m / (M_vir**2)) * abs(diff_sigma(M_vir)) * (M_vir / np.log10(np.exp(1.)))

def new_weight(M_vir,z,DLOGM,realizations = 100.):
    return hmf((10**M_vir)*cosmo.h,z)*DLOGM*(cosmo.h**3)/realizations

def weight_galprop(cat_path, out_path, galprop, haloprop, 
                   grid = 100., realization = 100.):
    halocat = pd.read_table(cat_path, 
                            comment="#", delim_whitespace=True, 
                            names=["tree_id", "root halo mass", "mass bin", "bin number", "realization"])

    galprop = read_data(out_path+galprop)
    haloprop = read_data(out_path+haloprop)
    
    DLOGM = (np.log10(max(halocat['mass bin'].values))-np.log10(min(halocat['mass bin'].values)))/(len(halocat['mass bin'].values)/grid)
    halocat['weight'] = new_weight(np.log10(halocat['mass bin']), 0, DLOGM, realization)
    haloprop['weight'] = halocat['weight']

    halo_weight = pd.Series(haloprop['weight'].values,index=haloprop['halo_index']).to_dict()
    galprop['weight']   = galprop['halo_index'].map(halo_weight)

    return galprop
    
def make_smf(galprop, Mmin=4, Mmax=13.5, step=0.25):
    smf = np.histogram(np.log10(galprop.mstar*1e9), 
                       weights = [_/step for _ in galprop['weight']], 
                       bins=np.arange(Mmin,Mmax,step))
    return smf[1][:-1]+step/2., smf[0]

def write_results(outpath,filename,data,xhead="none",yhead="none"):
    data_name   = filename
    with open("%s/%s"%(outpath,data_name), 'w') as out:
        out.write('# 0 %s \n'%xhead)
        out.write('# 1 %s \n'%yhead)
        for _ in range(len(data[0])):
            out.write(('%0.4f %0.4e \n')%(data[0][_], data[1][_]))


# In[4]:


SMF = run_SAM(3.0, 1.0)
