"""
Plot theta(w) for the XXZ Jacobi flow.

Expected input files (per W and L):
 - Results_summary_bin/theta_XXZ_Jac_L{L}_W{W:.2f}_dis{dis}_test.txt
 - Results_summary_bin/theta_err_XXZ_Jac_L{L}_W{W:.2f}_dis{dis}_test.txt

Produces:
 - Plots/theta_XXZ_bin.pdf
"""

import numpy as np
from scipy.ndimage import gaussian_filter  # kept in case smoothing is later re-enabled
from scipy.special import binom
import matplotlib as mpl
from matplotlib import pyplot as plt

# Figure and global matplotlib configuration
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
    "axes.linewidth": 0.8,
})
fig, ax = plt.subplots()

# Vector of disorder strengths and colormap normalization for the colorbar
W_vec = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.2, 3.5, 3.8, 4, 4.2, 4.4,
                  4.6, 4.8, 5, 5.2, 5.5, 6, 6.2, 6.5, 6.8, 7, 7.2,
                  7.5, 8, 9, 10, 11, 12, 13, 14, 15])

cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=np.min(W_vec), vmax=np.max(W_vec))

# Loop over system sizes (here the original code used range(14, 15) so only L=14)
for L in range(14, 15):
    dis = 2999  # disorder realization index used in filenames

    # Loop over disorder strengths and load precomputed theta, theta_err files
    for iw, W in enumerate(W_vec):
        # Generate the sequence of w values used in the original analysis.
        # The original code used a geometric progression: w_start/(factor**i)
        w_start = 1.0
        w_end = 0.01
        factor = 1.1

        # Calculate number of steps (same formula as original)
        # Ensure positive integer number of steps
        num_steps = int(np.log(w_end / w_start) / np.log(1.0 / factor))
        if num_steps < 4:
            # safe-guard, though not expected for the physical parameters used
            num_steps = 4

        w_values = np.array([w_start / (factor ** i) for i in range(num_steps)])

        # Load theta and its error for the given L, W, dis
        theta_fname = "Results_summary_bin/theta_XXZ_Jac_L%d_W%.2f_dis%d_test.txt" % (L, W, dis)
        theta_err_fname = "Results_summary_bin/theta_err_XXZ_Jac_L%d_W%.2f_dis%d_test.txt" % (L, W, dis)

        theta = np.loadtxt(theta_fname)
        theta_err = np.loadtxt(theta_err_fname)

        # w_plot is the mid-point between consecutive w steps used for plotting
        # (keeps the original indexing used in the script)
        w_plot = (w_values[2:-1] + w_values[3:]) / 2.0

        # Plot theta with shaded error band; match original slicing (theta[:-2])
        ax.plot(w_plot, theta[:-2], "-", linewidth=0.8, color=cmap(norm(W)))
        ax.fill_between(w_plot,
                        theta[:-2] - theta_err[:-2],
                        theta[:-2] + theta_err[:-2],
                        color=cmap(norm(W)), alpha=0.3)

# Create scalar mappable and horizontal colorbar above the plot
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([0.2, 0.82, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"$W$", labelpad=2)

# Choose which values to label on the colorbar
W_labels = [1, 3, 5, 8, 12, 15]
cbar.set_ticks(W_labels)
cbar.set_ticklabels([str(w) for w in W_labels])

# Add minor ticks for all W_vec values (unlabeled)
cbar.ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(W_vec))
cbar.ax.tick_params(axis='x', which='minor', length=3, width=0.6)

# Move ticks and labels ABOVE the bar (as in original)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(axis='x', direction='out', top=True, bottom=False, labeltop=True, labelbottom=False)

plt.subplots_adjust(top=0.8)

# Axis settings and decorations
ax.hlines(0, 0.001, 10, colors="black", linewidth=0.5)
ax.set_ylim(-3, 1)
ax.set_xlim(0.01, 0.2)
ax.set_xscale("log")
ax.set_xlabel("$w$")
ax.set_yticks([])      # no tick labels on y-axis in the original
ax.set_ylabel('')      # remove y-axis label

# Text annotations (kept from original)
ax.text(0.96, 0.04, "XXZ", transform=ax.transAxes, ha='right', va='bottom')
ax.text(0.96, 0.90, "(c)", transform=ax.transAxes, ha='right', va='bottom')

# Save and display the figure
plt.savefig("Plots/theta_XXZ_bin.pdf", bbox_inches="tight")
plt.show()
