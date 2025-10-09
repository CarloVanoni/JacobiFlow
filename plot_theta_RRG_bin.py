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

dis = 9999
W_vec = np.arange(1,26)

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=np.min(W_vec), vmax=np.max(W_vec))

for L in range(10,11):

    N=2**L

    k=0
    for iw, W in enumerate(W_vec[::-1]):

        w_start = 1.0   # Starting value
        w_end = 0.001    # Ending value
        factor = 1.1    # Geometric progression factor

        # Compute the number of steps needed
        num_steps = int(np.log(w_end / w_start) / np.log(1/1.1))

        # Generate the sequence using geomspace
        w_values = [w_start/(factor**i) for i in range(num_steps)]

        #print(w_values)

        filename = "Results_summary_bin/theta_RRG_Jac_L%d_W%.2f_dis%d_bin.txt"%(L,W,dis)
        theta = np.loadtxt(filename)

        filename = "Results_summary_bin/theta_err_RRG_Jac_L%d_W%.2f_dis%d_bin.txt"%(L,W,dis)
        theta_err = np.loadtxt(filename)

        filename = "Results_summary_bin/rho_dec_RRG_Jac_L%d_W%.2f_dis%d_bin.txt"%(L,W,dis)
        mean_nN = np.loadtxt(filename)

        w_values = np.array(w_values)
        w_plot = (w_values[2:-1]+w_values[3:])/2

        ind_plot = np.where(w_plot>0.001)[0]
        ind_plot = np.where(w_plot[ind_plot]<0.2)[0]

        theta_ind = theta[ind_plot]
        theta_err_ind = theta_err[ind_plot]

        w_plot_ind = w_plot[ind_plot]

        ind_max = np.where(w_plot_ind>0.035)[0]

        max_theta = np.max(theta_ind[ind_max])
        ind_max_theta = np.where(theta_ind==max_theta)[0]

        ind_max_theta = ind_max_theta[0]


        min_theta = np.min(theta_ind[ind_max_theta:])
        ind_min_theta = np.where(theta_ind==min_theta)
        ind_min_theta = ind_min_theta[0][0]

        if W > 19:
            ind_min_theta = len(w_plot_ind)


        theta_smooth = gaussian_filter(theta_ind,sigma = 1)[:-1]

        ax.plot(w_plot_ind,theta_ind,"-",linewidth=0.5,alpha=0.5,color=cmap(norm(W)))
        ax.fill_between(w_plot_ind, theta_ind - theta_err_ind,theta_ind + theta_err_ind,color=cmap(norm(W)), alpha=0.3)

        if W>19:
            ax.plot(w_plot_ind[ind_max_theta:ind_min_theta-1],theta_smooth[ind_max_theta:ind_min_theta],"-",color=cmap(norm(W)))
        else:
            ax.plot(w_plot_ind[ind_max_theta:ind_min_theta+1],theta_smooth[ind_max_theta:ind_min_theta+1],"-",color=cmap(norm(W)))

# Create scalar mappable
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add colorbar above the plot
cbar_ax = fig.add_axes([0.2, 0.82, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"$W$", labelpad=2)

# === Choose which values to label ===
W_labels = [1, 5, 10, 15, 20, 25]

# Set major ticks and labels (only for selected W)
cbar.set_ticks(W_labels)
cbar.set_ticklabels([str(w) for w in W_labels])

# Add minor ticks for all W_vec values (unlabeled)
cbar.ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(W_vec))
cbar.ax.tick_params(axis='x', which='minor', length=3, width=0.6)

# === Move ticks and labels ABOVE the bar ===
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(axis='x', direction='out', top=True, bottom=False, labeltop=True, labelbottom=False)

plt.subplots_adjust(top=0.8)

x = np.linspace(0.1,1,1000)
#plt.plot(x,0.1*x**(exp),"-",c="orange",label = r"$\theta_0 = 1-(0.5 + 1/(z \log(2)))^{-1}$")

ax.hlines(0,0.001,10,colors="black",linewidth=0.5)
#plt.hlines(1- 1/(0.5 + 1/(z*np.log(2))),0.001,1,linestyles="--",colors="orange",label = r"$\theta_0$")
#plt.hlines(0,0.001,10,colors="black",linewidth=0.5)
ax.set_ylim(-3,1)
ax.set_xlim(0.001,0.2)

#plt.yscale("log")
ax.set_xscale("log")
ax.set_xlabel("$w$")
#plt.ylabel(r"$\rho(\log w)$")
#ax.set_ylabel(r"$\theta$")
#plt.legend(fontsize=10,frameon=False,loc=3,ncols=2)

ax.set_yticks([])

# Remove y-axis label
ax.set_ylabel('')

ax.text(0.04, 0.04, "RRG",
        transform=ax.transAxes,   # use axes fraction coordinates
        ha='left', va='bottom')

#plt.savefig("Plots/rho_dec_Bethe_W%.2f.pdf"%W)

plt.savefig("Plots/theta_RRG_bin.pdf",bbox_inches="tight")

plt.show()
