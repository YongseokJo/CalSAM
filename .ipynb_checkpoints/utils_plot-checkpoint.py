from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import *
import matplotlib.colors as mcolors
from sbi import analysis as analysis
import pickle
import torch
from matplotlib.cm import ScalarMappable
import sys,os
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/CAMELS/MIEST/utils/"))
sys.path.append(os.path.abspath("/mnt/home/yjo10/ceph/myutils/"))
from plt_utils import generateAxesForMultiplePlots, remove_inner_axes

smf_obs    = np.load('observation/bernardi13_intp.npy')
smf_obs[:,1] = np.log10(smf_obs[:,1])
smz_obs    = np.load('observation/smz.npy')
smgf_obs = np.load('observation/smfgas.npy')


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])#,axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def probs2contours(probs, levels):
    """Takes an array of probabilities and produces an array of contours at specified percentile levels
    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]
    Return
    ------
    Array of same shape as probs with percentile labels
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def corner(axes, data,limits,fig_size, labels=None, bandwidth = [0.02,0.02,0.05,0.2,0.05,0.2],
           overlay=None, cmap_custom="Blues", alpha=1.0, fill=True, color='blue', num_level=20, ticks=(0,1)):
    n_params=data.shape[1]
    X=n_params; Y=n_params;
    tick_min, tick_max = ticks
    if isinstance(axes,bool) == True:
        if overlay:
            fig, axes = overlay
        else:
            axes = np.empty((n_params,n_params),dtype=object)
            fig = plt.figure(figsize=(14,14))
            fig.patch.set_visible(False)
            plt.subplots_adjust(hspace=0.0, wspace=0.0)

            ## construct axes list
            for i in range(n_params):
                for j in range(n_params):
                    n = n_params*j+i+1
                    axes[i,j] = fig.add_subplot(Y,X,n)
                    if i != 0:
                        axes[i,j].set_yticks([])
                    if j != n_params-1:   
                        axes[i,j].set_xticks([])
    else:
        fig = None

                
    ##################################
    ### Diagonal histogram!
    ##################################
    for i in range(n_params):
        X = data[:,i].reshape(-1,1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth[i]).fit(X)
        bins=200
        xs = np.linspace(
            limits[i][0], limits[i][1], bins+1,
        )
        ys = np.power(np.e,kde.score_samples(xs.reshape(-1,1)))
        axes[i,i].plot(
            xs,
            ys,
            color=color,
        )
        axes[i,i].set_xlim(limits[i])

    ##################################
    ### Contour plots!
    ##################################

    for i in range(n_params):
        for j in range(n_params):
            
            if overlay is None:
                ticks_y =\
                [[0.2,0.4],[0.6,1],[1.5,3],[0.25,4],[1,1.5],[0.5,2]]
                ticks_x =\
                [[0.1,0.5],[0.7,0.9],[0.25,4],[1,3],[0.5,2],[1,1.5]]
                ticks=[[0.2,0.3,0.4],[0.7,0.8,0.9],[1,2.125,3],
                       [1,2.125,3],[0.8,1.25,1.8],[0.8,1.25,1.8]]
                #############################
                ### Set labels and ticks
                ###############################
                if i == 0 and j > 0:
                    axes[i,j].set_ylabel(labels[j],fontsize=15)
                    #axes[i,j].set_yticks(ticks[j])
                    #axes[i,j].set_yticklabels(
                    #    ["{:.2f}".format(xtick) for xtick in ticks[j]])
                    #axes[i,j].tick_params(labelsize=12)
                if (j== n_params-1 and i < n_params-1) or (i==n_params-1 and j==n_params-1): ## (5,5) for 1d hist
                    axes[i,j].set_xlabel(labels[i],fontsize=15)
                    #axes[i,j].set_xticks(ticks[i])
                    #axes[i,j].set_xticklabels(
                    #    ["{:.2f}".format(xtick) for xtick in ticks[i]])
                    #axes[i,j].tick_params(labelsize=12)
                    
                
            if i>j:
                pass
                




            ##############################
            ### Plot
            ##############################
            if  i>=j:
                if i == n_params-1 and j==n_params-1:
                    axes[i,j].spines['right'].set_visible(False)
                    #axes[i,j].spines['left'].set_visible(False)
                    axes[i,j].spines['top'].set_visible(False)
                else:
                    axes[i,j].axis("off")
                continue
            else:
                axes[i,j].set_ylim(limits[j])
                axes[i,j].set_xlim(limits[i])

            if True:
                bins=20
                X = data[:,[i, j]]#.reshape(-1,2)
                bandwidth_here = np.mean([bandwidth[i],bandwidth[j]])
                #kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth_here).fit(X) #'epanechnikov'
                kde = KDEMultivariate(X, bw=[bandwidth[i],bandwidth[j]], var_type='cc')
                
                X, Y = np.meshgrid(
                    np.linspace(
                        limits[i][0],
                        limits[i][1],
                        bins,#opts["kde_offdiag"]["bins"],
                    ),
                    np.linspace(
                        limits[j][0],
                        limits[j][1],
                        bins, #opts["kde_offdiag"]["bins"],
                    ),
                )
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(
                    #np.power(np.e,kde.score_samples(positions.T)),
                    #np.power(np.e, kde.pdf(positions)),
                    kde.pdf(positions),
                    X.shape)

                Z = (Z - Z.min()) / (Z.max() - Z.min())

                levels = LinearLocator(num_level).tick_values(tick_min,tick_max)
                if fill:
                    cbar = axes[i,j].contourf(
                        X,
                        Y,
                        Z,
                        origin="lower",
                        extent=[
                            limits[i][0],
                            limits[i][1],
                            limits[j][0],
                            limits[j][1],
                        ],
                        levels=levels,
                        cmap=mpl.cm.get_cmap(cmap_custom).copy(), #"Blues"
                        extend="both",
                        #alpha=alpha,
                    )

                else:
                    cbar = axes[i,j].contour(
                        X,
                        Y,
                        Z,
                        origin="lower",
                        extent=[
                            limits[i][0],
                            limits[i][1],
                            limits[j][0],
                            limits[j][1],
                        ],
                        levels=levels,
                        #colors=color,
                        cmap=mpl.cm.get_cmap(cmap_custom).copy(),
                        linestyles="-"
                        #extend="both",
                    )
                cbar.cmap.set_under([1, 1, 1, 0])
                cbar.changed()

    return fig, axes



def corner_plot(ifname, n_round, bandwidth, new_sample=None, limits=None, old=False, colors=["b", "r"], cmaps=['Blues', 'Reds']):
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='Times'
    plt.rcParams["font.family"] = "serif"
    #plt.rcParams['text.usetex']=False
    alpha_posterior = 0.2
    num_level=6
    ticks=(1e-1,1)    
    true     = (1.0,1.7,3.0,110.0,1.2,2.0E-03,0.1)
    if limits is None:
        limits = [(0.25,4), (0.425,6.8), (1,5), (27.5,440), (0.6,2.4), (5e-4,0.008), (0.025,0.4)]
        
        

    tng_path = "./one_click_sam/test3_1/"
    ##############################################################################
    ## Construnction of subplots: SMFs, Residuals, and Legend
    center_x=-1;center_y=0.6;width_x=1.8;width_y=1.6;residual_height=0.2;slack=0.05;
    subpos = [[center_x-width_x, center_y                              , width_x, width_y],
              [center_x        , center_y                              , width_x, width_y],
              [center_x        , center_y-width_y-residual_height-slack, width_x, width_y]]

    respos = [[center_x-width_x, center_y-residual_height                , width_x, residual_height],
              [center_x        , center_y-residual_height                , width_x, residual_height],
              [center_x        , center_y-width_y-2*residual_height-slack, width_x, residual_height]]
    #legendpos = [center_x+width_x,
    #            center_y-2*width_y-2*residual_height-slack, width_x, width_y]
    subax = []
    resax = []
    ## Finished the constructions
    ##############################################################################
    
    #colors    = ['b', 'r', 'g', 'y']
    #cmaps     = ['Blues', 'Reds', 'Greens', 'Yellows']
    obs       = [smf_obs, smz_obs, smgf_obs]
    obs_label = ["Bernardi", "Gallazi", "Boselli"]
    obs_name  = ['smf', 'smz', 'smgf']
    for k in range(len(ifname)):
        fname    = ifname[k]; num = n_round[k]
        if old:
            fpath    = f"./result/{fname}"         
        else:
            fpath    = f"../CalSAM/result/{fname}"         
        if new_sample is None:
            params   = np.load(f"{fpath}/params/params_{num}.npy")
        else:
            params = new_sample[k]
        

        if k == 0:
            overlay = None
            fill    = True
        else:
            overlay = (fig,axes)
            fill    = False
        
        fig, axes = corner(False, params,
                           limits=limits, fig_size=(14,14),
                           labels=[r"$\tau_{\star,0}$",
                                   r"$\epsilon_\mathrm{SN,0}$",
                                   r"$\alpha_\mathrm{rh}$",
                                   "f_eject_thresh",
                                   "YIELD",
                                   "f_Edd_radio",
                                   "f_return"],
                          bandwidth=bandwidth[k],
                          cmap_custom=cmaps[k],#new_cmap,
                          overlay=overlay,
                          alpha=1, fill=fill, color=colors[k], num_level=num_level, ticks=ticks)

        if k == 0:
            for pos in subpos:
                subax.append(add_subplot_axes(axes[6,1],pos))

        for j in range(3):
            x = obs[j][:,0]
            observable = np.load(f'{fpath}/sam/{obs_name[j]}_{num}.npy')
            subax[j].plot(x,observable.T, c=colors[k], alpha=alpha_posterior,lw=1)
            subax[j].plot(x,np.mean(observable,axis=0), c=colors[k], alpha=1,lw=2)
            fid = np.load(f'{tng_path}/{obs_name[j]}_TNG300.npy')
            subax[j].plot(x,fid.T, c='k',ls="--", alpha=1.,lw=2, label="Fiducial")
            subax[j].plot(x,obs[j][:,1], c='k',label=obs_label[j],lw=2)
            if k == 0:
                subax[j].legend()
        subax[0].set_ylim(-6,-1.5)
        subax[1].set_ylim(-0.9,0.4)
        subax[2].set_ylim(-0.7,0.5)
        subax[1].yaxis.tick_right()

    n_params = 7
    for i in range(n_params):
        axes[i,i].axvline(x=true[i],c='k', ls=':')
        for j in range(n_params-1):
            if i<n_params-1 and j>=i:
                axes[i,j+1].scatter(true[i],true[j+1], marker='x',c='k',zorder=100,linewidths=0.5)

        # Create the legend and position it outside the plot
        #axes[4,3].legend(custom_lines, custom_labels, loc='center',
        #          bbox_to_anchor=(0.5, 0.5),fontsize=12)
    return fig, axes
    #plt.savefig(f"paper_plot/{fname}.png", dpi=200, bbox_inches="tight")
        #plt.show()
