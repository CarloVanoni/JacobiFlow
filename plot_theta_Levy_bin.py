#!/usr/bin/env python3
"""
plot_theta_Levy_bin.py

This script:
- loads theta and theta error data previously produced and saved to text files,
- smooths theta with a Gaussian filter,
- plots theta vs w for a range of mu values using a reversed viridis colormap,
- adds a horizontal colorbar for mu and saves the figure.
"""

import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib as mpl

# Figure / plotting defaults
plt.rcParams["figure.figsize"] = [5, 5]
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})

fig, ax = plt.subplots()

# Parameters (these control which files are read and how the plot is produced)
dis = 7999
gamma = 1
mu_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2]
W = 1

w_start = 10
w_end = 0.001
factor = 1.1  # geometric step factor used to construct w_plot

# Colormap and normalization for mu values
cmap = plt.cm.viridis_r
norm = mpl.colors.Normalize(vmin=np.min(mu_vec), vmax=np.max(mu_vec))

# Loop over system size(s). The original code used enumerate(range(10,11)) but only n is needed.
for n in range(10, 11):
    N = 2 ** n

    # Loop over mu values and load corresponding theta files
    for iw, mu in enumerate(mu_vec):

        # number of steps used to build w_plot (keeps original numeric behavior)
        num_steps = int(np.log(w_end / w_start) / np.log(1 / 1.1))

        # w values plotted (geometrically decreasing from w_start)
        w_plot = np.array([w_start / (factor ** i) for i in range(num_steps)])
        w_plot = w_plot[:-1]  # match original slicing

        # Load theta and its error from precomputed summary files
        filename = "Results_summary_bin/theta_Levy_Jac_N%d_gamma%.2f_mu%.2f_W%.2f_dis%d_bin.txt" % (
            N,
            gamma,
            mu,
            W,
            dis,
        )
        theta = np.loadtxt(filename)

        filename = "Results_summary_bin/theta_err_Levy_Jac_N%d_gamma%.2f_mu%.2f_W%.2f_dis%d_bin.txt" % (
            N,
            gamma,
            mu,
            W,
            dis,
        )
        theta_err = np.loadtxt(filename)

        # Select indices for plotting where w is in the desired window
        ind_plot = np.where(w_plot > 0.0001)[0]
        ind_plot = np.where(w_plot[ind_plot] < 0.1)[0]

        # Find the index of the minimum theta within the selected range
        min_theta = np.min(theta[ind_plot])
        ind_min_theta = np.where(theta == min_theta)[0][0]

        # Smooth theta for plotting (preserve original slicing behavior)
        theta_smooth = gaussian_filter(theta, sigma=1)[:-1]

        # Plotting logic:
        # - For small mu (0.1, 0.2) plot full curves and fill error band.
        # - For other mu values, plot the portion up to the local minimum using the
        #   smoothed theta and afterwards plot the raw theta with reduced alpha.
        if mu == 0.1 or mu == 0.2:
            ax.plot(w_plot[:-1], theta_smooth, "-", color=cmap(norm(mu)))
            ax.plot(w_plot, theta, "-", alpha=0.5, linewidth=0.5, color=cmap(norm(mu)))
            ax.fill_between(w_plot, theta - theta_err, theta + theta_err, color=cmap(norm(mu)), alpha=0.3)
        else:
            ax.plot(w_plot[:ind_min_theta], theta_smooth[:ind_min_theta], "-", color=cmap(norm(mu)))
            ax.plot(
                w_plot[ind_min_theta - 1 : -1],
                theta[ind_min_theta - 1 : -1],
                "-",
                alpha=0.5,
                linewidth=0.5,
                color=cmap(norm(mu)),
            )
            ax.fill_between(
                w_plot[:ind_min_theta],
                theta[:ind_min_theta] - theta_err[:ind_min_theta],
                theta[:ind_min_theta] + theta_err[:ind_min_theta],
                color=cmap(norm(mu)),
                alpha=0.3,
            )

# Create scalar mappable for the colorbar and add horizontal colorbar above the plot
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([0.2, 0.82, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label(r"$\mu$", labelpad=2)

# Choose which mu values to label explicitly on the colorbar
mu_labels = [0.1, 0.5, 0.8, 1.2]
cbar.set_ticks(mu_labels)
cbar.set_ticklabels([str(w) for w in mu_labels])

# Add minor tick marks for every mu in mu_vec (unlabeled)
cbar.ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(mu_vec))
cbar.ax.tick_params(axis="x", which="minor", length=3, width=0.6)

# Move ticks and labels above the bar
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.tick_params(axis="x", direction="out", top=True, bottom=False, labeltop=True, labelbottom=False)

plt.subplots_adjust(top=0.8)

# Horizontal reference line and axis scaling/limits
ax.hlines(0, 0.001, 10, colors="black", linewidth=0.5)
ax.set_xscale("log")
ax.set_xlabel("$w$")
ax.set_ylabel(r"$\theta$")
ax.set_xlim(1e-3, 1)
ax.set_ylim(-3, 1)

# Label inside axes and save figure
ax.text(0.04, 0.04, "Levy RP", transform=ax.transAxes, ha="left", va="bottom")

plt.savefig("Plots/theta_Levy_W%d_bin.pdf" % W, bbox_inches="tight")
plt.show()
