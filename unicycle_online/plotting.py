import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm

from unicycle_online.util.const import W_traj_s
from util import Integrator, dynamics
import jax
import cvxpy as cp
from util import const as ct

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

T = ct.T
N = ct.N
time_traj = ct.time_traj
n = ct.n
W_traj = ct.W_traj


def data_plotting(x_traj_sim, u_traj_sim, K_traj, Q_traj):
    ## construct the funnel bounds
    x1_bounds = np.zeros([T, 2])
    x2_bounds = np.zeros([T, 2])
    for t in range(T):
        ## plotting the ellipsoid
        Q_t = Q_traj[t, 0:2, 0:2]
        Q_half = la.sqrtm(Q_t)
        # Eigen-decomposition (ascending order from eigh)
        vals, vecs = LA.eigh(Q_t)
        order = vals.argsort()[::-1]  # sort descending so index 0 is largest
        vals = vals[order]
        ## x1 upper and lower bounds
        x1_bounds[t, 0] = x_traj_sim[0, t, 0] + np.sqrt(vals[0])
        x1_bounds[t, 0] = x_traj_sim[0, t, 0] + np.sqrt(Q_t[0, 0])
        # x1_bound_2 = x_traj_sim[0, t, 0] + np.sqrt(Q_t[0,0])
        # x1_bound_3 = find_funnel_bound(x_traj_sim[0,t],u_traj_sim[t],Q_t,K_traj[t])
        # print("Method 1: ", x1_bounds[t,0], "Method 2: ", x1_bound_2, "Method 3:", x1_bound_3[0])
        x1_bounds[t, 1] = x_traj_sim[0, t, 0] - np.sqrt(vals[0])
        x1_bounds[t, 1] = x_traj_sim[0, t, 0] - np.sqrt(Q_t[0, 0])
        ## x2 upper and lower bounds
        x2_bounds[t, 0] = x_traj_sim[0, t, 1] + np.sqrt(vals[1])
        x2_bounds[t, 1] = x_traj_sim[0, t, 1] - np.sqrt(vals[1])
        x2_bounds[t, 0] = x_traj_sim[0, t, 1] + np.sqrt(Q_t[1, 1])
        x2_bounds[t, 1] = x_traj_sim[0, t, 1] - np.sqrt(Q_t[1, 1])
    plt.subplot(2, 1, 1)
    ## plot state
    for i in range(N + 1):
        plt.plot(time_traj[0:T - 1], x_traj_sim[i, 0:T - 1, 0])
    plt.plot(time_traj[0:T - 1], x1_bounds[0:T - 1, 0], "r", linewidth=2, label="x1 upper bound")
    plt.plot(time_traj[0:T - 1], x1_bounds[0:T - 1, 1], "r-.", linewidth=2, label="x1 lower bound")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("State (x1)")
    plt.subplot(2, 1, 2)
    for i in range(N + 1):
        plt.plot(time_traj[0:T - 1], x_traj_sim[i, 0:T - 1, 1])
    plt.plot(time_traj[0:T - 1], x2_bounds[0:T - 1, 0], "r", linewidth=2, label="x2 upper bound")
    plt.plot(time_traj[0:T - 1], x2_bounds[0:T - 1, 1], "r-.", linewidth=2, label="x2 lower bound")
    plt.xlabel("time")
    plt.ylabel("State (x2)")
    plt.legend()
    plt.show()

    ## plot the controls
    Q_u_traj = np.zeros([T - 1, ct.m, ct.m])
    u1_bounds = np.zeros([T - 1, 2])  ## upper and lower
    u2_bounds = np.zeros([T - 1, 2])
    for t in range(T - 1):
        Q_u_traj[t] = K_traj[t] @ Q_traj[t] @ K_traj[t].T
        u1_bounds[t, 0] = u_traj_sim[0, t, 0] + np.sqrt(Q_u_traj[t, 0, 0])  ## upper
        u1_bounds[t, 1] = u_traj_sim[0, t, 0] - np.sqrt(Q_u_traj[t, 0, 0])  ## lower

        u2_bounds[t, 0] = u_traj_sim[0, t, 1] + np.sqrt(Q_u_traj[t, 1, 1])  ## upper
        u2_bounds[t, 0] = np.maximum(u2_bounds[t, 0], np.max(u_traj_sim[:, t, 1]))
        u2_bounds[t, 1] = u_traj_sim[0, t, 1] - np.sqrt(Q_u_traj[t, 1, 1])  ## lower
        u2_bounds[t, 1] = np.minimum(u2_bounds[t, 1], np.min(u_traj_sim[:, t, 1]))
    plt.subplot(2, 1, 1)
    plt.plot(time_traj[0:T - 2], u1_bounds[0:T - 2, 0], "r", linewidth=2, label="u1 upper bound")
    plt.plot(time_traj[0:T - 2], u1_bounds[0:T - 2, 1], "r-.", linewidth=2, label="u1 lower bound")
    for i in range(N):
        plt.plot(time_traj[0:T - 2], u_traj_sim[i, 0:T - 2, 0])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("Control (u1)")

    plt.subplot(2, 1, 2)
    plt.plot(time_traj[0:T - 2], u2_bounds[0:T - 2, 0], "r", linewidth=2, label="u2 upper bound")
    plt.plot(time_traj[0:T - 2], u2_bounds[0:T - 2, 1], "r-.", linewidth=2, label="u2 lower bound")
    for i in range(N):
        plt.plot(time_traj[0:T - 2], u_traj_sim[i, 0:T - 2, 1])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("Control (u2)")

    plt.show()

    ##plot the process noise
    plt.subplot(2, 1, 1)
    print(W_traj_s[0, 0:T - 1, 0])
    for i in range(ct.N):
        plt.plot(time_traj[0:T - 1], W_traj_s[i, 0:T - 1, 0])

    plt.xlabel("time")
    plt.ylabel("Noise (w1)")
    plt.subplot(2, 1, 2)
    for i in range(ct.N):
        plt.plot(time_traj[0:T - 1], W_traj_s[i, 0:T - 1, 1])
    plt.xlabel("time")
    plt.ylabel("Noise (w2)")
    plt.show()
    return
