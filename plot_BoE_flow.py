import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams["figure.figsize"] = [5,5]
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})

# Vector field (ODE system)
def flow(t, z):
    x, y = z
    dxdt = -1
    dydt = 0.1 * (y - 1) * np.exp(x * y)
    return [dxdt, dydt]

# Integration domain
t_span = (0, 10)     # time goes forward
t_eval = np.linspace(0, 10, 100000)

# Initial conditions for multiple streamlines
y0_vals = np.linspace(-0.25, 1, 30)   # seed along the right boundary x=1
trajectories = []

for y0 in y0_vals:
    sol = solve_ivp(flow, t_span, [0, y0], t_eval=t_eval)
    trajectories.append(sol)

color = plt.get_cmap("seismic", len(trajectories))

yc = 0.434916

solc = solve_ivp(flow, t_span, [0, yc], t_eval=t_eval)

# Plot
for i, sol in enumerate(trajectories):
    #plt.plot(sol.y[0], sol.y[1], color=color(len(trajectories)-i))

    x, y = sol.y
    plt.plot(x, y, color=color(len(trajectories)-i))

    # pick a midpoint for the arrow
    idx = len(x)//2
    x0, y0 = x[idx], y[idx]
    x1, y1 = x[idx+50], y[idx+50]  # a little ahead along the trajectory

    plt.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color(len(trajectories)-i),
            lw=1
        )
    )

plt.plot(solc.y[0], solc.y[1],"--", color="black")

plt.xlabel("$\log w$", fontsize=12)
plt.ylabel("$\\theta(w)$", fontsize=12)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlim(-10, 0)
plt.ylim(-1, 1)

plt.savefig("Plots/BoE_flow.pdf",bbox_inches="tight")

plt.show()

