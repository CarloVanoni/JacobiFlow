import numpy as np
import csv
import networkx as nx
import sys, os
import random
from matplotlib import pyplot as plt
#from matplotlib import colormaps as cm
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter
import sys
import matplotlib as mpl
#from histo_maker import *

def powerLawDistribution(n, theta, x_min, x_max):
    number = np.array([])
    alpha = 2-theta
    for i in range(n):
        norm = x_max**(1-alpha) - x_min**(1-alpha)
        u_val = random.uniform(0, 1)
        number = np.append(number,random.choice((-1, 1)) * (u_val*norm + x_min**(1-alpha))**(1/(1-alpha)))

    return number

plt.rcParams["figure.figsize"] = [5,5]
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})
#cols = plt.get_cmap('cool', 11)
fig, ax = plt.subplots()

dis = 7999
gamma = 1
mu_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2]
#mu_vec = [ 1.2]
W = 1

w_start=10
w_end = 0.001

factor = 1.1

cmap = plt.cm.viridis_r
norm = mpl.colors.Normalize(vmin=np.min(mu_vec), vmax=np.max(mu_vec))

for i,n in enumerate(range(10,11)):
    N = 2**n
    #for ia, a in enumerate([0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 1.7, 2]):
    for iw, mu in enumerate(mu_vec):

        num_steps = int(np.log(w_end / w_start) / np.log(1/1.1))

        w_plot = np.array([w_start/(factor**i) for i in range(num_steps)])
        w_plot = w_plot[:-1]

        filename = "Results_summary_bin/theta_Levy_Jac_N%d_gamma%.2f_mu%.2f_W%.2f_dis%d_bin.txt"%(N,gamma,mu,W,dis)
        theta = np.loadtxt(filename)

        filename = "Results_summary_bin/theta_err_Levy_Jac_N%d_gamma%.2f_mu%.2f_W%.2f_dis%d_bin.txt"%(N,gamma,mu,W,dis)
        theta_err = np.loadtxt(filename)

        #filename = "Results_summary/rho_res_N%d_W%.2f_z%.2f_dis%d_2.txt"%(N,W,z,dis)
        #mean_rN = np.loadtxt(filename)
            
        #mean_nN_new = gaussian_filter(mean_nN,sigma=1)

        #theta = 1 + np.divide(np.diff(np.log(mean_nN_new)),np.diff(np.log(mean_w)))

        #plt.plot(w_plot[1:-1],theta[1:-1],"--",linewidth=0.5,color=cols(iw), label=r'$\mu=%.2f$'%mu)

        ind_plot = np.where(w_plot>0.0001)[0]
        ind_plot = np.where(w_plot[ind_plot]<0.1)[0]
        #print(ind_plot)

        min_theta = np.min(theta[ind_plot])
        ind_min_theta = np.where(theta==min_theta)
        #print(ind_min_theta[0])
        ind_min_theta = ind_min_theta[0][0]

        """mean_nN = mean_nN[1:-1]

        gauss = gaussian_filter(mean_nN[::-1]*N,sigma = 1)

        theta_smooth = 1 + np.divide(np.diff(np.log(gauss[::-1])),np.diff(np.log(w_plot)))"""

        theta_smooth = gaussian_filter(theta,sigma = 1)[:-1]

        if mu == 0.1 or mu == 0.2:
            ax.plot(w_plot[:-1],theta_smooth,"-",color=cmap(norm(mu)))
            ax.plot(w_plot,theta,"-",alpha=0.5,linewidth=0.5,color=cmap(norm(mu)))
            ax.fill_between(w_plot, theta - theta_err,theta + theta_err,color=cmap(norm(mu)), alpha=0.3)

        else:
            ax.plot(w_plot[:ind_min_theta],theta_smooth[:ind_min_theta],"-",color=cmap(norm(mu)))
            ax.plot(w_plot[ind_min_theta-1:-1],theta[ind_min_theta-1:-1],"-",alpha=0.5,linewidth=0.5,color=cmap(norm(mu)))
            ax.fill_between(w_plot[:ind_min_theta], theta[:ind_min_theta] - theta_err[:ind_min_theta],theta[:ind_min_theta] + theta_err[:ind_min_theta],color=cmap(norm(mu)), alpha=0.3)
        

# Create scalar mappable
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add colorbar above the plot
cbar_ax = fig.add_axes([0.2, 0.82, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"$\mu$", labelpad=2)

# === Choose which values to label ===
mu_labels = [0.1, 0.5, 0.8, 1.2]

# Set major ticks and labels (only for selected W)
cbar.set_ticks(mu_labels)
cbar.set_ticklabels([str(w) for w in mu_labels])

# Add minor ticks for all W_vec values (unlabeled)
cbar.ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(mu_vec))
cbar.ax.tick_params(axis='x', which='minor', length=3, width=0.6)

# === Move ticks and labels ABOVE the bar ===
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(axis='x', direction='out', top=True, bottom=False, labeltop=True, labelbottom=False)

plt.subplots_adjust(top=0.8)



x = np.linspace(0.1,1,1000)
#plt.plot(x,0.1*x**(exp),"-",c="orange",label = r"$\theta_0 = 1-(0.5 + 1/(z \log(2)))^{-1}$")

#plt.hlines(1- 1/(0.5 + 1/(z*np.log(2))),0.001,1,linestyles="--",colors="orange",label = r"$\theta_0$")
ax.hlines(0,0.001,10,colors="black",linewidth=0.5)
ax.set_ylim(-3,1)


#plt.yscale("log")
ax.set_xscale("log")
ax.set_xlabel("$w$")
#plt.ylabel(r"$\rho(\log w)$")
ax.set_ylabel(r"$\theta$")

#plt.title("$W=%d$"%W)
ax.set_xlim(1e-3,1)
ax.set_ylim(-3,1)
#plt.savefig("Plots/rho_dec_Bethe_W%.2f.pdf"%W)

ax.text(0.04, 0.04, "Levy RP",
        transform=ax.transAxes,   # use axes fraction coordinates
        ha='left', va='bottom')

plt.savefig("Plots/theta_Levy_W%d_bin.pdf"%W, bbox_inches="tight")

plt.show()
