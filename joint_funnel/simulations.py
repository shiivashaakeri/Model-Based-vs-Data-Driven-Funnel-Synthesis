import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
from util import Integrator, dynamics
import jax
import cvxpy as cp
from util import const as ct

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

T = ct.T
n = ct.n
m = ct.m
dt = ct.dt
obs = ct.obs
obs_r = ct.obs_r
num_obs = ct.num_obs
x_des = ct.x_des

N = ct.N


def traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, is_multi, is_test, is_plotting):
    test_t = 15
    x_traj_sim = np.zeros([N + 1, T, n])
    ## simulate the nominal traj
    x_traj_sim[0] = np.zeros([T, n])
    u_traj_sim = np.zeros([T - 1, m])
    x_traj_sim[0, 0] = ct.x_0
    for t in range(T - 1):
        u_t = u_traj[t] + K_traj[t] @ (x_traj_sim[0, t] - x_traj[t])
        u_traj_sim[t] = u_t
        x_traj_sim[0, t + 1] = Integrator.RK4(dt, x_traj_sim[0, t], u_t, np.zeros(ct.nw))
    ## simulate the traj boundles
    if is_test == False:
        Q = Q_traj[0, 0:2, 0:2]
    else:
        Q = Q_traj[test_t, 0:2, 0:2]
    # Eigen-decomposition (ascending order from eigh)
    vals, vecs = LA.eigh(Q)
    order = vals.argsort()[::-1]  # sort descending so index 0 is largest
    vals = vals[order]
    vecs = vecs[:, order]
    vmax = vecs[:, 0]
    angle_deg = np.degrees(np.arctan2(vmax[1], vmax[0]))
    x_0s = np.zeros([N, n])
    phi = np.arctan2(vmax[1], vmax[0])
    theta = np.linspace(0, 2 * np.pi, N)
    ## rotate the coordinate by phi (ellipsoid rotation)
    DCM = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi), np.cos(phi)]])
    ## semi axes length
    a = np.sqrt(vals[0]) - 0.01
    b = np.sqrt(vals[1]) - 0.01
    for i in range(N):
        idx = i + 1
        x_0s[i, 0:2] = np.array([a * np.cos(theta[i]), b * np.sin(theta[i])])
        x_0s[i, 0:2] = DCM @ x_0s[i, 0:2]
        if is_test == False:
            x_0s[i, 0] += ct.x_0[0]
            x_0s[i, 1] += ct.x_0[1]
            x_0s[i, 2] = ct.x_0[2]
        else:
            x_0s[i, 0] += x_traj[test_t, 0]
            x_0s[i, 1] += x_traj[test_t, 1]
            x_0s[i, 2] = x_traj[test_t, 2]
        ## simulate the traj boundles
        x_traj_sim[idx, 0] = x_0s[i]

        for t in range(T - 1):
            u_t = u_traj[t] + K_traj[t] @ (x_traj_sim[idx, t] - x_traj[t])
            if is_plotting == True:
                x_traj_sim[idx, t + 1] = Integrator.RK4(dt, x_traj_sim[idx, t], u_t, W_traj[t])
            else:
                x_traj_sim[idx, t + 1] = Integrator.RK4(dt, x_traj_sim[idx, t], u_t, np.zeros(ct.nw))
    if is_plotting == True:
        if is_multi == True:
            plotting_fcn(x_traj_sim, u_traj, Q_traj, True)
        else:
            plotting_fcn(x_traj_sim, u_traj, Q_traj, False)

    return x_traj_sim, u_traj_sim


def plotting_fcn(x_traj, u_traj, Q_traj, is_multi):
    fig, ax = plt.subplots()
    ## plot traj boundles
    if is_multi == True:
        for i in range(N):
            ax.plot(x_traj[i + 1, :, 0], x_traj[i + 1, :, 1], "r")
    for t in range(T - 1):
        ## plot the nominal traj
        theta = x_traj[0, t, 2]
        leng = 0.4
        heading_vector = np.array([[x_traj[0, t, 0], x_traj[0, t, 1]],
                                   [x_traj[0, t, 0] + leng * np.cos(theta), x_traj[0, t, 1] + leng * np.sin(theta)]])

        ax.plot(heading_vector[:, 0], heading_vector[:, 1], "g")
        ax.plot(x_traj[0, t, 0], x_traj[0, t, 1], "g.")
        ax.plot(x_des[0], x_des[1], "go")

        ## plotting circles
        theta_grid = np.linspace(0, 2 * np.pi, 101)
        for j in range(num_obs):
            x_theta_j = obs[j, 0] + np.cos(theta_grid) * obs_r
            y_theta_j = obs[j, 1] + np.sin(theta_grid) * obs_r
            ax.plot(x_theta_j, y_theta_j)

        ## plotting the ellipsoid
        Q_t = Q_traj[t, 0:2, 0:2]
        # Eigen-decomposition (ascending order from eigh)
        vals, vecs = LA.eigh(Q_t)
        order = vals.argsort()[::-1]  # sort descending so index 0 is largest
        vals = vals[order]
        vecs = vecs[:, order]
        vmax = vecs[:, 0]
        angle_deg = np.degrees(np.arctan2(vmax[1], vmax[0]))
        ell = Ellipse(xy=(x_traj[0, t, 0], x_traj[0, t, 1]), width=2 * np.sqrt(vals[0]), height=2 * np.sqrt(vals[1]),
                      angle=angle_deg, fill=True, alpha=0.5)
        ax.add_patch(ell)
        ax.set_aspect('equal', adjustable='box')

        plt.pause(0.01)
        if t == T - 2:
            plt.show()
    plt.clf()
