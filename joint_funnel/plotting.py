import jax
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA  # noqa: N812
from util import const as ct

jax.config.update("jax_enable_x64", True)

T = ct.T
N = ct.N
time_traj = ct.time_traj


def data_plotting(x_traj_sim, u_traj_sim, K_traj, Q_traj):  # noqa: ARG001
    ## construct the funnel bounds
    x1_bounds = np.zeros([T, 2])
    x2_bounds = np.zeros([T, 2])
    for t in range(T):
        ## plotting the ellipsoid
        Q_t = Q_traj[t, 0:2, 0:2]
        # Eigen-decomposition (ascending order from eigh)
        vals, vecs = LA.eigh(Q_t)
        order = vals.argsort()[::-1]  # sort descending so index 0 is largest
        vals = vals[order]
        ## x1 upper and lower bounds
        x1_bounds[t, 0] = x_traj_sim[0, t, 0] + np.sqrt(vals[0])
        x1_bounds[t, 1] = x_traj_sim[0, t, 0] - np.sqrt(vals[0])
        ## x2 upper and lower bounds
        x2_bounds[t, 0] = x_traj_sim[0, t, 1] + np.sqrt(vals[1])
        x2_bounds[t, 1] = x_traj_sim[0, t, 1] - np.sqrt(vals[1])

    plt.subplot(2, 1, 1)
    ## plot state
    for i in range(N + 1):
        plt.plot(time_traj[0 : T - 1], x_traj_sim[i, 0 : T - 1, 0])
    plt.plot(time_traj[0 : T - 1], x1_bounds[0 : T - 1, 0], "r", linewidth=2, label="x upper bound")
    plt.plot(time_traj[0 : T - 1], x1_bounds[0 : T - 1, 1], "r", linewidth=2, label="x lower bound")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("x")
    plt.subplot(2, 1, 2)
    for i in range(N + 1):
        plt.plot(time_traj[0 : T - 1], x_traj_sim[i, 0 : T - 1, 1])
    plt.plot(time_traj[0 : T - 1], x2_bounds[0 : T - 1, 0], "r", linewidth=2, label="y upper bound")
    plt.plot(time_traj[0 : T - 1], x2_bounds[0 : T - 1, 1], "r", linewidth=2, label="y lower bound")
    plt.xlabel("time")
    plt.ylabel("y")
    plt.legend()
    plt.show()
