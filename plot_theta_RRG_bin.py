#!/usr/bin/env python3
"""
Plot theta for RRG from precomputed result files.

"""

import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib as mpl

# Figure default size and LaTeX-style fonts
plt.rcParams["figure.figsize"] = [2.33, 2.33]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8
})

fig, ax = plt.subplots()

# Fixed parameters used to build filenames and iterate Ws
dis = 9999
W_vec = np.arange(1, 26)

# Colormap and normalization for the W colorbar
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=np.min(W_vec), vmax=np.max(W_vec))

# Iterate over system sizes (only L=10 in original script)
for L in range(10, 11):
    N = 2**L

    # Loop over W values in reversed order (so colors map consistently)
    for iw, W in enumerate(W_vec[::-1]):

        # Build a geometric sequence of w values (matching original behavior)
        w_start = 1.0    # Starting value
        w_end = 0.001    # Ending value
        factor = 1.1     # Geometric progression factor

        # Compute (integer) number of steps (keeps original logic)
        num_steps = int(np.log(w_end / w_start) / np.log(1/1.1))

        # Generate the sequence using the geometric factor
        w_values = [w_start / (factor**i) for i in range(num_steps)]
        w_values = np.array(w_values)

        # Filenames for the precomputed results (unchanged)
        filename = "Results_summary_bin/theta_RRG_Jac_L%d_W%.2f_dis%d_bin.txt" % (L, W, dis)
        theta = np.loadtxt(filename)

        filename = "Results_summary_bin/theta_err_RRG_Jac_L%d_W%.2f_dis%d_bin.txt" % (L, W, dis)
        theta_err = np.loadtxt(filename)

        filename = "Results_summary_bin/rho_dec_RRG_Jac_L%d_W%.2f_dis%d_bin.txt" % (L, W, dis)
        mean_nN = np.loadtxt(filename)

        # Prepare w values for plotting: midpoints of neighboring entries (matching original)
        w_plot = (w_values[2:-1] + w_values[3:]) / 2

        # Original selection logic (kept as-is to preserve behavior)
        ind_plot = np.where(w_plot > 0.001)[0]
        ind_plot = np.where(w_plot[ind_plot] < 0.2)[0]

        theta_ind = theta[ind_plot]
        theta_err_ind = theta_err[ind_plot]
        w_plot_ind = w_plot[ind_plot]

        # Identify peak and subsequent minimum index used for smoothing and highlighting
        ind_max = np.where(w_plot_ind > 0.035)[0]

        max_theta = np.max(theta_ind[ind_max])
        ind_max_theta = np.where(theta_ind == max_theta)[0]
        ind_max_theta = ind_max_theta[0]

        min_theta = np.min(theta_ind[ind_max_theta:])
        ind_min_theta = np.where(theta_ind == min_theta)
        ind_min_theta = ind_min_theta[0][0]

        # For large W the code originally extends the range to the end
        if W > 19:
            ind_min_theta = len(w_plot_ind)

        # Smooth theta for plotting the main highlighted segment
        theta_smooth = gaussian_filter(theta_ind, sigma=1)[:-1]

        # Plot raw theta and error band
        ax.plot(w_plot_ind, theta_ind, "-", linewidth=0.5, alpha=0.5, color=cmap(norm(W)))
        ax.fill_between(w_plot_ind,
                        theta_ind - theta_err_ind,
                        theta_ind + theta_err_ind,
                        color=cmap(norm(W)), alpha=0.3)

        # Plot smoothed portion between the identified max and min indices
        if W > 19:
            ax.plot(w_plot_ind[ind_max_theta:ind_min_theta-1],
                    theta_smooth[ind_max_theta:ind_min_theta],
                    "-", linewidth=0.8, color=cmap(norm(W)))
        else:
            ax.plot(w_plot_ind[ind_max_theta:ind_min_theta+1],
                    theta_smooth[ind_max_theta:ind_min_theta+1],
                    "-", linewidth=0.8, color=cmap(norm(W)))

# Create scalar mappable for the colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add horizontal colorbar above the plot and configure ticks/labels
cbar_ax = fig.add_axes([0.2, 0.82, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"$W$", labelpad=2)

# Choose which W values to label on the colorbar
W_labels = [1, 5, 10, 15, 20, 25]
cbar.set_ticks(W_labels)
cbar.set_ticklabels([str(w) for w in W_labels])

# Add minor ticks for all W_vec values (unlabeled)
cbar.ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(W_vec))
cbar.ax.tick_params(axis='x', which='minor', length=3, width=0.6)

# Move ticks and label above the bar
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(axis='x', direction='out', top=True, bottom=False, labeltop=True, labelbottom=False)

plt.subplots_adjust(top=0.8)

# Visual guide and axes limits (kept as in original script)
ax.hlines(0, 0.001, 10, colors="black", linewidth=0.5)
ax.set_ylim(-3, 1)
ax.set_xlim(0.001, 0.2)

ax.set_xscale("log")
ax.set_xlabel("$w$")

# Remove y-axis ticks and label (original styling)
ax.set_yticks([])
ax.set_ylabel('')

# Annotations
ax.text(0.96, 0.04, "RRG", transform=ax.transAxes, ha='right', va='bottom')
ax.text(0.96, 0.90, "(b)", transform=ax.transAxes, ha='right', va='bottom')

# Save and show plot (unchanged)
plt.savefig("Plots/theta_RRG_bin.pdf", bbox_inches="tight")
plt.show()
