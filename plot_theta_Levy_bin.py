"""
plot_theta_Levy_bin.py

Reads precomputed "theta" curves (and their errors) from the
Results_summary_bin directory and plots theta(w) for various
values of the Levy parameter mu. The script produces a PDF
"Plots/theta_Levy_W{W}_bin.pdf" and shows the figure.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import matplotlib as mpl

# ---------- Figure and plotting defaults ----------
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

# ---------- Parameters (same as original) ----------
dis = 7999            # disorder identifier used in filenames
gamma = 1             # parameter used in filenames
mu_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2]  # Levy exponents to plot
W = 1                 # another parameter used in filenames

# w range construction: start and end for the geometric sequence
w_start = 10
w_end = 0.001
factor = 1.1          # geometric step factor used to build w_plot

# Colormap and normalization across mu values for consistent coloring
cmap = plt.cm.viridis_r
norm = mpl.colors.Normalize(vmin=np.min(mu_vec), vmax=np.max(mu_vec))

# ---------- Main data-loading and plotting loop ----------
# The original code looped over range(10,11) and used n=10.
# Preserving the original behavior: N = 2**10
n = 10
N = 2**n

for iw, mu in enumerate(mu_vec):
    # compute number of steps for the geometric sequence of w
    num_steps = int(np.log(w_end / w_start) / np.log(1 / factor))

    # build the w values (decreasing geometric series)
    w_plot = np.array([w_start / (factor**i) for i in range(num_steps)])
    # original code removed the last element; keep same slicing
    w_plot = w_plot[:-1]

    # Load precomputed theta and theta_err from Results_summary_bin
    filename = "Results_summary_bin/theta_Levy_Jac_N%d_gamma%.2f_mu%.2f_W%.2f_dis%d_bin.txt" % (
        N, gamma, mu, W, dis)
    theta = np.loadtxt(filename)

    filename_err = "Results_summary_bin/theta_err_Levy_Jac_N%d_gamma%.2f_mu%.2f_W%.2f_dis%d_bin.txt" % (
        N, gamma, mu, W, dis)
    theta_err = np.loadtxt(filename_err)

    # Select the indices in w_plot that are between 1e-4 and 1e-1 (as in original code).
    ind_plot = np.where(w_plot > 0.0001)[0]
    ind_plot = np.where(w_plot[ind_plot] < 0.1)[0]

    # Find the minimum theta within the previously computed indices and its index
    min_theta = np.min(theta[ind_plot])
    ind_min_theta = np.where(theta == min_theta)[0][0]

    # Smooth theta (Gaussian smoothing) and drop the last element as in original code
    theta_smooth = gaussian_filter(theta, sigma=1)[:-1]

    # Plot behavior:
    # - for the smallest mu values (0.1 and 0.2) plot the full (smoothed) curve
    #   and overlay the raw theta with reduced alpha and a shaded error band.
    # - for other mu values, plot only up to the index of the minimum (smoothed),
    #   then plot (part of) the raw curve similarly, and shade the error region
    #   for the portion up to the minimum (matching original logic).
    color = cmap(norm(mu))
    if mu == 0.1 or mu == 0.2:
        ax.plot(w_plot[:-1], theta_smooth, "-", linewidth=0.8, color=color)
        ax.plot(w_plot, theta, "-", alpha=0.5, linewidth=0.5, color=color)
        ax.fill_between(w_plot, theta - theta_err, theta + theta_err,
                        color=color, alpha=0.3)
    else:
        ax.plot(w_plot[:ind_min_theta], theta_smooth[:ind_min_theta], "-",
                linewidth=0.8, color=color)
        ax.plot(w_plot[ind_min_theta - 1:-1], theta[ind_min_theta - 1:-1], "-",
                alpha=0.5, linewidth=0.5, color=color)
        ax.fill_between(w_plot[:ind_min_theta],
                        theta[:ind_min_theta] - theta_err[:ind_min_theta],
                        theta[:ind_min_theta] + theta_err[:ind_min_theta],
                        color=color, alpha=0.3)

# ---------- Colorbar setup (horizontal, above the plot) ----------
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # needed for colorbar

cbar_ax = fig.add_axes([0.2, 0.82, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r"$\mu$", labelpad=2)

# Choose which values to label on the colorbar and add minor ticks for all mu values
mu_labels = [0.1, 0.5, 0.8, 1.2]
cbar.set_ticks(mu_labels)
cbar.set_ticklabels([str(w) for w in mu_labels])
cbar.ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(mu_vec))
cbar.ax.tick_params(axis='x', which='minor', length=3, width=0.6)

# Move ticks and label above the colorbar
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(axis='x', direction='out', top=True, bottom=False,
                    labeltop=True, labelbottom=False)

plt.subplots_adjust(top=0.8)

# ---------- Axes formatting ----------
ax.hlines(0, 0.001, 10, colors="black", linewidth=0.5)
ax.set_xscale("log")
ax.set_xlabel("$w$")
ax.set_ylabel(r"$\theta$")
ax.set_xlim(1e-3, 1)
ax.set_ylim(-3, 1)

# small labels within the axes (preserve original placement/text)
ax.text(0.96, 0.04, "Levy RP", transform=ax.transAxes, ha='right', va='bottom')
ax.text(0.96, 0.90, "(a)", transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0'))

# Save and show the figure (same filename as original)
plt.savefig("Plots/theta_Levy_W%d_bin.pdf" % W, bbox_inches="tight")
plt.show()
