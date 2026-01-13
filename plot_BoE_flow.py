"""
plot_BoE_flow.py

Compute and plot streamlines (trajectories) of a 2D ODE (a flow) in the (x,y) plane.

Notes:
- The ODE is defined so x decreases at constant rate -1 (dx/dt = -1), while y evolves
  according to dydt = 0.1 * (y - 1) * exp(x * y).
- The script integrates trajectories starting from a vertical line of initial points
  (x=0, y=y0) and plots them with arrows indicating direction of the flow.
- A special trajectory with initial y = yc is plotted as a dashed black line.
- The script uses a dense time evaluation grid (100000 points) to have smooth curves
  and to allow picking nearby points for arrows.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Figure configuration: square figure and LaTeX-style text for labels
plt.rcParams["figure.figsize"] = [5, 5]
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})

# ---------------------------------------------------------------------
# Vector field (ODE system)
# ---------------------------------------------------------------------
def flow(t, z):
    """
    Right-hand side of the ODE system.

    Parameters
    ----------
    t : float
        Independent variable (time). The ODE is autonomous (no explicit t-dependence),
        but solve_ivp still requires it.
    z : sequence-like, length 2
        State vector [x, y].

    Returns
    -------
    list [dxdt, dydt]
        Time derivatives of x and y.
    """
    x, y = z
    # x decreases at constant rate -1 (so x = x0 - t).
    dxdt = -1

    # y evolves nonlinearly, scaled by 0.1. When y = 1 the factor (y - 1) vanishes.
    # The exponential factor depends on the product x*y.
    dydt = 0.1 * (y - 1) * np.exp(x * y)

    # Return an iterable accepted by solve_ivp
    return [dxdt, dydt]

# ---------------------------------------------------------------------
# Integration domain and evaluation grid
# ---------------------------------------------------------------------
t_span = (0, 10)                           # integrate from t=0 to t=10 (forward time)
t_eval = np.linspace(0, 10, 100000)       # dense sampling of solution for smooth plotting

# ---------------------------------------------------------------------
# Initial conditions: seed multiple streamlines
# ---------------------------------------------------------------------
# Note: original inline comment said "seed along the right boundary x=1" but the code
# actually seeds at x=0. We preserve the code behavior and seed at x = 0.
y0_vals = np.linspace(-0.25, 1, 30)   # 30 seed points for y along the vertical line x=0
trajectories = []                      # will store the returned OdeResult objects

# Integrate each streamline from the same x=0 starting point with different y
for y0 in y0_vals:
    # solve_ivp returns an object with arrays in sol.y (shape (2, len(t_eval)))
    sol = solve_ivp(flow, t_span, [0, y0], t_eval=t_eval)
    trajectories.append(sol)

# Choose a colormap for plotting multiple streamlines
color = plt.get_cmap("seismic", len(trajectories))

# ---------------------------------------------------------------------
# Special/reference trajectory (with initial y = yc)
# ---------------------------------------------------------------------
yc = 0.434916
solc = solve_ivp(flow, t_span, [0, yc], t_eval=t_eval)

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
for i, sol in enumerate(trajectories):
    # sol.y contains two rows: x (row 0) and y (row 1)
    x, y = sol.y

    # Plot the trajectory curve. We pick colors from the colormap.
    plt.plot(x, y, color=color(len(trajectories) - i))

    # Add a small arrow along the trajectory to indicate direction.
    # We pick a midpoint index and a point slightly further along the trajectory.
    # Because t_eval is dense (100000 samples), using idx+50 advances a small amount.
    idx = len(x) // 2
    # Defensive check: ensure the forward index exists (should be fine with current t_eval)
    forward_idx = min(idx + 50, len(x) - 1)

    x0, y0 = x[idx], y[idx]
    x1, y1 = x[forward_idx], y[forward_idx]

    plt.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color(len(trajectories) - i),
            lw=1
        )
    )

# Plot the special trajectory as a dashed black line for emphasis/comparison
plt.plot(solc.y[0], solc.y[1], "--", color="black")

# Axis labels (using LaTeX)
plt.xlabel("$\log w$", fontsize=12)
plt.ylabel("$\\theta(w)$", fontsize=12)

# Add axes lines at x=0 and y=0 for reference
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

# Set the viewing window (x from -10 to 0, y from -1 to 1)
plt.xlim(-10, 0)
plt.ylim(-1, 1)

# Save figure to PDF (directory "Plots" should exist or this will raise an error)
plt.savefig("Plots/BoE_flow.pdf", bbox_inches="tight")

# Show interactive plot (or blocking window)
plt.show()
