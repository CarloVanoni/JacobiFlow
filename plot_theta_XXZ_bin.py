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
import matplotlib.colors as mcolors
from scipy.special import binom
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


W_vec = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.2, 3.5, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.5, 6, 6.2, 6.5, 6.8, 7, 7.2, 7.5, 8])

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=np.min(W_vec), vmax=np.max(W_vec))

for L in range(14,15):

    N=int(binom(L,L//2))
    
    dis = 2999

    k=0
    for iw, W in enumerate(W_vec):

        w_start = 1.0   # Starting value
        w_end = 0.01    # Ending value
        factor = 1.1    # Geometric progression factor

        # Compute the number of steps needed
        num_steps = int(np.log(w_end / w_start) / np.log(1/1.1))

        print(num_steps)

        # Generate the sequence using geomspace
        w_values = [w_start/(factor**i) for i in range(num_steps)]

        filename = "Results_summary_bin/theta_XXZ_Jac_L%d_W%.2f_dis%d_test.txt"%(L,W,dis)
        theta = np.loadtxt(filename)

        filename = "Results_summary_bin/theta_err_XXZ_Jac_L%d_W%.2f_dis%d_test.txt"%(L,W,dis)
        theta_err = np.loadtxt(filename)

        filename = "Results_summary_bin/rho_dec_XXZ_Jac_L%d_W%.2f_dis%d_test.txt"%(L,W,dis)
        mean_nN = np.loadtxt(filename)

        w_values = np.array(w_values)
        w_plot = (w_values[2:-1]+w_values[3:])/2

        gauss = gaussian_filter(mean_nN*N,sigma = 0.8)

        theta_smooth = 1 + np.divide(np.diff(np.log(gauss)),np.diff(np.log(w_values)))
        
        ax.plot(w_plot,theta[:-2],"-",color=cmap(norm(W)))
        ax.fill_between(w_plot, theta[:-2] - theta_err[:-2],theta[:-2] + theta_err[:-2],color=cmap(norm(W)), alpha=0.3)


# Create scalar mappable
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add colorbar above the plot
cbar_ax = fig.add_axes([0.2, 0.82, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"$W$", labelpad=2)

# === Choose which values to label ===
W_labels = [1, 3, 5, 8]

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

#plt.hlines(1- 1/(0.5 + 1/(z*np.log(2))),0.001,1,linestyles="--",colors="orange",label = r"$\theta_0$")
ax.hlines(0,0.001,10,colors="black",linewidth=0.5)
ax.set_ylim(-3,1)
ax.set_xlim(0.01,0.2)

#plt.yscale("log")
ax.set_xscale("log")
ax.set_xlabel("$w$")
#plt.ylabel(r"$\rho(\log w)$")
#ax.set_ylabel(r"$\theta$")

ax.set_yticks([])

# Remove y-axis label
ax.set_ylabel('')

ax.text(0.04, 0.04, "XXZ",
        transform=ax.transAxes,   # use axes fraction coordinates
        ha='left', va='bottom')


plt.savefig("Plots/theta_XXZ_bin.pdf",bbox_inches="tight")

plt.show()
