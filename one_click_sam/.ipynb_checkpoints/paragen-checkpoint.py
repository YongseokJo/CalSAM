import sys

if len(sys.argv)==1:
    para_name    = "param.scsam"
    out_path     = "."
    lib_path     = "lib_path"
    tree_path    = "\place\holder\path\isotree.dat"
    sim_name     = "VSMDPL"
    
################################## Read System Argv ##################################


else:
    para_name    = sys.argv[1]      ### name of the parameter file
    out_path     = sys.argv[2]      ### path to the actual run
    lib_path     = sys.argv[3]      ### path to the library folder
    tree_path    = sys.argv[4]      ### path to the tree file
    sim_name     = sys.argv[5]      ### output snapshots
    alpha_rh     = sys.argv[6]
    tau_star_0   = sys.argv[7]

box_lst    = ['VSMDPL']
#snaps = [12, 15, 17, 20, 22, 25, 28, 31, 34, 38, 42, 46, 51, 56, 62]
snaps = [150]

snap_lst = [snaps]
text = dict(zip(box_lst, snap_lst))

res_lst = ['300000000.0']
res_dict = dict(zip(box_lst, res_lst))

txt = ''
for _ in text[sim_name]:
    txt+= str(_)+str(' ')+str(_)+'\n'


snaplen = len(text[sim_name])


################################## parameter file content ##################################

content = '\
#gf parameter file (gfn.v2)\n\
#december 2017\n\
#pathname of input and output\n\
"%s"\n\
#pathname of library files\n\
"%s"\n\
#verbosity\n\
1\n\
#seed\n\
-12345\n\
#main free parameters\n\
#chi_gas sigma_crit\n\
1.7 6.0\n\
#tau_star_0\n\
%s\n\
#recipe for quiescent star formation 0=kenn density 1=bigiel1 2=bigiel2\n\
2\n\
#H2 recipe (0 = GK 1 = KMT 2= BR)\n\
0\n\
#ionized gas\n\
#sigma_HII fion_internal\n\
0.4 0.2\n\
#epsilon_SN_0 alpha_rh f_eject_thresh (km/s)\n\
#gas is ejected if V_halo<f_eject_thresh\n\
1.7 %s 110.0\n\
#drive_AGN_winds epsilon_wind_QSO\n\
1 0.5\n\
#YIELD f_enrich_cold\n\
1.2 1.0\n\
#cooling enrichment (0=self-consistent 1=fixed) Zh_fixed\n\
0 0.0\n\
#radio heating parameters: f_Edd_radio\n\
2.0E-03\n\
#f_recycle (recycled gas from mass loss/SN)\n\
0.43\n\
#BH parameters\n\
#fbhcrit tau_Q mseedbh\n\
0.5 0.4 0.04 1.0E04\n\
#bhmassbulgemassev\n\
#0=NOEV 1= ZEV 2=FGAS 3=BHFP\n\
0\n\
#sigmaBH_create\n\
0.3\n\
#tidal stripping (0/1)\n\
#if on, expects arguments f_strip tau_strip f_dis\n\
#to make stripping MORE effective make f_strip<1\n\
#to make stripping less effective make f_strip>1\n\
1 1.0 0.22 0.8\n\
#f_scatter\n\
0.2\n\
#f_return (re-infall of ejected gas)\n\
0.1\n\
#SQUELCH z1 z1 z_squelch\n\
#z1 and z2 should be regarded as just parameters in the okamoto fitting fnc\n\
#z_squelch is the redshift that squelching is turned on\n\
1 9.0 3.5 8.0\n\
#disk model 0=SP 1=isothermal 2=MMW\n\
2\n\
#alpha_burst\n\
3.0\n\
#disk instability\n\
#0=off 1=on (stars+gas)\n\
#epsilon_m\n\
1 0.3\n\
#cosmological parameters\n\
#geometry: 0=EDS, 1=FLAT, 2=OPEN\n\
1\n\
#Omega_0 (matter) Omega_Lambda_0 h_100\n\
0.3089 0.6911 0.6774\n\
#f_baryon: negative number means use default value\n\
0.1573\n\
#variance file\n\
"variance/var.planck15.dat"\n\
#save_history\n\
1\n\
#metallicity binning information\n\
#NMET_SFHIST minmet_sfhist maxmet_sfhist\n\
12 -2.5 0.6\n\
#NT_SFHIST dt_sfhist\n\
1381 0.01\n\
#zmin_outputsfhist zmax_outputsfhist mstarmin_outputsfhist [code units]\n\
0.0 20.0 1.0E-12\n\
#minimum root mass (Msun)\n\
%s\n\
#tree file format 0=bolshoi planck 1=illustrisTNG 2=GUREFT 3=ForestMgnt\n\
3\n\
#filename of file containing list of tree filenames\n\
"%sfiles.list"\n\
#output\n\
#NSNAP (number of snapshots in the merger tree file)\n\
151\n\
#NZOUT\n\
%d\n\
#minsnap maxsnap must be NZOUT entries (set to 1 NSNAP for NZOUT=1)\n\
%s\
#quantities to output:\n\
#galprop halos history trace mergers\n\
1 1 0 0 0\n\
#GAL_SELECT 0=mstar 1=mhalo\n\
1\n\
1.0E05\n\
#C_rad (major): sp-sp, sp-e, e-e\n\
2.5 0 0\n\
#C_rad (minor): sp-sp, sp-e, e-e\n\
1.35 0 0\n\
#C_int (major): sp-sp, sp-e, e-e\n\
0.5 0.5 0.5\n\
#C_int (minor): sp-sp, sp-e, e-e\n\
0.5 0.5 0.5\n\
#usemainbranchonly (0/1)\n\
0\n\
'%(out_path, lib_path, tau_star_0, alpha_rh, res_dict[sim_name], out_path, snaplen, txt)

################################## parameter file content ##################################


with open('%s%s'%(out_path,para_name), "w") as f:
    f.write(content)





################################## parameter file content ##################################

content = '\
1\n\
"%s"\n\
'%tree_path

################################## tree file content ##################################

with open('%s/files.list'%(out_path), "w") as f:
    f.write(content)



