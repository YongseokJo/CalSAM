#!/mnt/sw/nix/store/6a11b1qfnfwxjzs8q4xnlid1z3mvkz9l-python-3.9.12-view/bin/python

import argparse
import numpy as np
import pandas as pd
import os
import time
import astropy.units as u
from   astropy.cosmology import FlatLambdaCDM
from   scipy.optimize    import fsolve
import multiprocessing


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



#### new in v3 -- Feb 9, 2024
### -- gas fraction is only computed for galaxies with (galprop['mbulge']/galprop['mstar'] < 0.4)


### User setup

class one_click_SAM():
    ## define cosmology, choose one -- 
    #cosmo = FlatLambdaCDM(Om0=0.307, Ob0=0.048, H0=67.8)    ## Matching BolshoiP
    cosmo = FlatLambdaCDM(Om0=0.3069, Ob0=0.0486, H0=67.74)  ## Matching IllustrisTNG

    ### parameters to vary
    sam_path  ='/mnt/ceph/users/lyung/sam_code/sc-sam.gureft/'
    lib_path  ='/mnt/ceph/users/lyung/sam_code/gflib/'


    def __init__(self, para_path, res_path, proc_num, sim='TNG100'):
        self.para_path = para_path
        self.res_path  = res_path
        self.proc_num  = proc_num
        self.sim       = sim

        if sim == 'TNG50':
            self.tree_path = '/mnt/ceph/users/lyung/Nbody/TNG_TreePrep/TNG50_50_50/gridtree.dat'
            self.cat_path  = '/mnt/ceph/users/lyung/Nbody/TNG_TreePrep/TNG50_50_50/halocat.dat'
            self.z0snap    = 99
        elif sim == 'TNG100':
            self.tree_path ='/mnt/ceph/users/lyung/Nbody/TNG_TreePrep/TNG100_50_50/gridtree.dat'
            self.cat_path  ='/mnt/ceph/users/lyung/Nbody/TNG_TreePrep/TNG100_50_50/halocat.dat'
            self.z0snap    = 99
        elif sim == 'TNG300':
            self.tree_path ='/mnt/ceph/users/lyung/Nbody/TNG_TreePrep/TNG300_50_50/gridtree.dat'
            self.cat_path  ='/mnt/ceph/users/lyung/Nbody/TNG_TreePrep/TNG300_50_50/halocat.dat'
            self.z0snap    = 98


    def run(self, params, keep_catalogs=True):
        ### Assign params
        tau_star_0, epsilon_SN_0, alpha_rh,\
                f_eject_thresh, YIELD, f_Edd_radio, f_return = params

        ## Ranges
        """
        tau_star_0     [0.25, 4]
        epsilon_SN_0   [6.8, 0.425]
        alpha_rh       [5, 1]
        f_eject_thresh [440, 27.5]
        YIELD          [2.4, 0.6]
        f_Edd_radio    [0.008, 5e-4]
        f_return       [0.4, 0.025]
        """

        ### name of the parameter file
        para_name = 'params_{}.param'.format(self.proc_num)

        ### create a parameter file
        os.system('python paragen.py %s %s %s %s %s %s %s %s %s %s %s %s %s %s'\
                  %(para_name,
                    self.para_path,
                    self.lib_path,
                    self.tree_path,
                    self.sim,
                    self.z0snap,
                    tau_star_0,
                    epsilon_SN_0,
                    alpha_rh,
                    f_eject_thresh,
                    YIELD,
                    f_Edd_radio,
                    f_return,
                    self.res_path))

        ### run the SAM with that parameter file
        os.system('cd %s && ./gf %s/%s &> %s/%s'\
                  %(self.sam_path,self.para_path,para_name,self.res_path,"output.txt"))
        galprop = \
                self.weight_galprop('galprop_%d-%d.dat'%(self.z0snap,self.z0snap),
                                    'haloprop_%d-%d.dat'%(self.z0snap,self.z0snap))


        ## make and write SMF
        SMF = self.make_smf(galprop, Mmin=4.0, Mmax=14.0, step=0.25)
        self.write_results('smf.dat',SMF,xhead="log M*",yhead="phi")


        ## make and write scaling relations
        galprop['f_cold'] = (galprop['mHI'] + galprop['mH2'])  / galprop['mstar']
        galprop['Zstar']  = galprop['Metal_star']/galprop['mstar']

        galprop2            = galprop[(galprop['mbulge']/galprop['mstar'] < 0.4)]
        x, y_84, y_50, y_16 = self.scaling_relation(np.log10(galprop2['mstar']*1e9), np.log10(galprop2['f_cold']),
                                                    galprop2['weight'], lolim=8, uplim=12, steps=20, xlog=False)
        self.write_results_scale('mstar_fcold.dat',[x, y_16, y_50, y_84],xhead="log mstar",yhead="log f_cold")

        x, y_84, y_50, y_16 = self.scaling_relation(np.log10(galprop['mstar']*1e9), np.log10(galprop['Zstar']),
                                                    galprop['weight'], lolim=8, uplim=12, steps=20, xlog=False)
        self.write_results_scale('mstar_Zstar.dat',[x, y_16, y_50, y_84],xhead="log mstar",yhead="log Zstar")

        galprop3            = galprop[(galprop['mbulge'] > 0) & (galprop['mBH'] > 0)]
        x, y_84, y_50, y_16 = self.scaling_relation(np.log10(galprop3['mbulge']*1e9), np.log10(galprop3['mBH']*1e9),
                                                    galprop3['weight'], lolim=8, uplim=12, steps=20, xlog=False)
        self.write_results_scale('mbulge_mBH.dat',[x, y_16, y_50, y_84],xhead="log mbulge",yhead="log mBH")

        ## delete catalogues at the end
        if keep_catalogs==False:
            os.system("rm %s/galprop*.dat"%out_path)
            os.system("rm %s/haloprop*.dat"%out_path)

    def read_data(self, x):
        with open(x, 'r') as dat:
            lines  = dat.readlines()
            header = [line.strip().split(" ")[2] for line in lines if line[0] == "#"]
        return pd.read_table(x, comment="#", delim_whitespace=True, names=header)



    ############################# 
    #### Halo mass functions ####
    #############################
    def sigma(self, M_vir):
        y     = 1e12/M_vir
        term1 = 17.967209*y**0.409964
        term2 = 1 + 1.179455*y**0.210397 + 6.192103*y**0.320073
        return term1/term2

    def g(self, z):                           #eq.28 in RP+2016
        a  = self.cosmo.scale_factor(z)
        om = self.cosmo.Om(z)
        ol = self.cosmo.Ode(z)
        return (0.4 * om * a)/(np.power(om,0.571428571)-ol+(1+om*0.5)*(1+ol*0.014285714))

    def D(self, z):
        return self.g(z)/self.g(0)

    def f(self, M_vir, z):  # following RP+2016
        A = lambda x: 0.143873 - 0.0112026*x  + 0.00253025*(x**2)
        a = lambda x: 1.35053  + 0.0681399*x  - 0.00593122*(x**2)
        b = lambda x: 3.11271  - 0.0770555*x  - 0.0134007*(x**2)
        c = lambda x: 1.1869   + 0.00875954*x - 0.000739608*(x**2)
        sig = self.sigma(M_vir)*self.D(z)
        return A(z) * (((sig/b(z))**(-a(z))) +1 ) * np.exp(-c(z)/(sig**2))

    def diff_sigma(self, M_vir):
        return (np.log(1.0/self.sigma(1.002*M_vir))-np.log(1.0/self.sigma(0.998*M_vir)))/(np.log(1.002*M_vir)-np.log(0.998*M_vir))

    def hmf(self,M_vir,z):
        rho_m = \
                (self.cosmo.Om(0)*self.cosmo.critical_density(0).to(u.solMass/(u.Mpc)**3)).value/self.cosmo.h**2
        return self.f(M_vir,z) * (rho_m / (M_vir**2)) * abs(self.diff_sigma(M_vir)) * (M_vir / np.log10(np.exp(1.)))

    def new_weight(self,M_vir,z,DLOGM,realizations = 100.):
        return self.hmf((10**M_vir)*self.cosmo.h,z)*DLOGM*(self.cosmo.h**3)/realizations
    #################################### 
    #### end of Halo mass functions ####
    ####################################


    ### this function assigns weight based on halo mass to the galprop table
    def weight_galprop(self, galprop, haloprop):#, 
                       #grid = 100., realization = 100.):
        halocat = pd.read_table(self.cat_path, comment="#", delim_whitespace=True,
                                names=["tree_id", "root halo mass", "mass bin", "bin number", "realization"])
        grid        = max(halocat['bin number'])+1
        realization = max(halocat['realization'])+1
        galprop     = self.read_data(self.res_path+galprop)
        haloprop    = self.read_data(self.res_path+haloprop)
        DLOGM = (np.log10(max(halocat['mass bin'].values))-np.log10(min(halocat['mass bin'].values)))/(len(halocat['mass bin'].values)/grid)
        halocat['weight']  = self.new_weight(np.log10(halocat['mass bin']), 0, DLOGM, realization)
        haloprop['weight'] = halocat['weight']
        halo_weight        = pd.Series(haloprop['weight'].values,index=haloprop['halo_index']).to_dict()
        galprop['weight']  = galprop['halo_index'].map(halo_weight)
        return galprop


    def quantile_1D(self, data, weights, quantile):
        # Check the data
        if not isinstance(data, np.matrix):
            data = np.asarray(data)
        if not isinstance(weights, np.matrix):
            weights = np.asarray(weights)
        nd = data.ndim
        if nd != 1:
            raise TypeError("data must be a one dimensional array")
        ndw = weights.ndim
        if ndw != 1:
            raise TypeError("weights must be a one dimensional array")
        if data.shape != weights.shape:
            raise TypeError("the length of data and weights must be the same")
        if ((quantile > 1.) or (quantile < 0.)):
            raise ValueError("quantile must have a value between 0. and 1.")
        # Sort the data
        ind_sorted = np.argsort(data)
        sorted_data = data[ind_sorted]
        sorted_weights = weights[ind_sorted]
        # Compute the auxiliary arrays
        Sn = np.cumsum(sorted_weights)
        # TODO: Check that the weights do not sum zero
        #assert Sn != 0, "The sum of the weights must not be zero"
        Pn = (Sn-0.5*sorted_weights)/Sn[-1]
        # Get the value of the weighted median
        return np.interp(quantile, Pn, sorted_data)


    ### this function make distribution functions that can be printed out with the next function
    def make_smf(self, galprop, Mmin=4, Mmax=13.5, step=0.25):
        galprop = galprop[galprop.mstar>0]
        smf = np.histogram(np.log10(galprop.mstar*1e9), 
                           weights = [_/step for _ in galprop['weight']], 
                           bins=np.arange(Mmin,Mmax,step))
        return smf[1][:-1]+step/2., smf[0]


    def write_results(self,filename,data,xhead="none",yhead="none"):
        with open("%s/%s"%(self.res_path,filename), 'w') as out:
            out.write('# 0 %s \n'%xhead)
            out.write('# 1 %s \n'%yhead)
            for _ in range(len(data[0])):
                out.write(('%0.4f %0.4e \n')%(data[0][_], data[1][_]))


    ### this function make scaling relations that can be printed out with the next function
    def scaling_relation(self, xlist, ylist, weight, lolim, uplim, steps, xlog=True):
        table = pd.DataFrame({'x':xlist, 'y':ylist, 'weight':weight})
        table = table[table['y'] > -100]

        if xlog == True:
            binedges = np.logspace(np.log10(lolim), np.log10(uplim), steps)
        elif xlog == False:
            binedges = np.linspace(lolim, uplim, steps)

        x, y_84, y_50, y_16 = [],[],[],[]

        for _ in range(len(binedges)-1):
            table = table[table['x'] > 0]
            hold = table[(table['x']>binedges[_]) & (table['x']<binedges[_+1]) ]

            if len(hold)>10:
                x.append( (binedges[_+1]+binedges[_])/2. )
                y_84.append(self.quantile_1D(hold['y'], hold['weight'], 0.84))
                y_50.append(self.quantile_1D(hold['y'], hold['weight'], 0.50))
                y_16.append(self.quantile_1D(hold['y'], hold['weight'], 0.16))

        return x, y_84, y_50, y_16


    def write_results_scale(self,filename,data,xhead="none",yhead="none"):
        data_name   = filename
        with open("%s/%s"%(self.res_path,filename), 'w') as out:
            out.write('# 0 %s \n'%xhead)
            out.write('# 1 %s 16 \n'%yhead)
            out.write('# 2 %s 50 \n'%yhead)
            out.write('# 3 %s 84 \n'%yhead)
            for _ in range(len(data[0])):
                out.write(('%0.4f %0.4f %0.4f %0.4f \n')%(data[0][_], data[1][_], data[2][_], data[3][_]))


def parallel_SAM(data):
    proc_num  = int(data[0])
    params    = data[2:]
    sim_num   = int(data[1])

    if sim_num == 50:
        sim = "TNG50"
    elif sim_num == 100:
        sim = "TNG100"
    elif sim_num == 300:
        sim = "TNG300"

    para_path = f'/mnt/home/yjo10/ceph/ILI/SAM/one_click_sam/params/{args.name}'
    res_path  =\
            '/mnt/home/yjo10/ceph/ILI/SAM/one_click_sam/result/{}'.format(args.name)

    try:
        os.system("mkdir {}".format(res_path))
        res_path += '/' + str(proc_num)
        os.system("mkdir {}".format(res_path))
        res_path += '/'
        os.system("mkdir {}".format(para_path))
        para_path += '/'
    except:
        pass

    ocSAM = one_click_SAM(para_path,res_path, proc_num, sim)
    ocSAM.run(params)



if __name__=="__main__":
    ### Load what round it is
    fpath = '/mnt/home/yjo10/ceph/ILI/SAM/result/' + args.name
    try:
        f = open(f"{fpath}/n_round.txt","r")
        n_round = int(f.read())
    except:
        n_round = 0

    sim = args.simulation


    ### Load params proposed by ILI
    params = np.load(f"{fpath}/params/params_{n_round}.npy")
    new_data = np.zeros((params.shape[0], params.shape[1]+2))
    new_data[:,2:] = params
    new_data[:,0]  = np.arange(new_data.shape[0])

    if sim == "TNG50":
        new_data[:,1]  = 50
    elif sim == "TNG100":
        new_data[:,1]  = 100
    elif sim == "TNG300":
        new_data[:,1]  = 300


    ### Run SAM
    num_processes = multiprocessing.cpu_count()
    print(params.shape[0], num_processes)
    with multiprocessing.Pool(processes=params.shape[0]) as pool:
        pool.map(parallel_SAM, (new_data))
    #SMF = run_SAM(params)




