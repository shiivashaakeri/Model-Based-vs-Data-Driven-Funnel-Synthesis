import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from util import Integrator, dynamics, linearization
import jax
import cvxpy as cp
from traj_upt import traj_gen, plotting_fcn
from funnel_upt import funnel_gen
from lipschitz import lipschitz_estimator

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from util import const as ct
from simulations import traj_sim

max_iter = 30
m = ct.m
n = ct.n
T = ct.T
dt = ct.dt
W_traj = ct.W_traj
K_traj = ct.K0_traj
Q_traj = ct.Q0_traj
########## select model modes
mode = 0

########### channel selections
C = ct.C_u
D = ct.D_u
E = ct.E_u
G = ct.G_u1


def K0_fcn(x_traj, u_traj):
    K0_traj = np.zeros([T - 1, m, n])
    for t in range(T - 1):
        K0_t = cp.Variable([m, n])
        ## u = BKx
        f = cp.norm((u_traj[t] - ct.u_traj[t]) - K0_t @ (x_traj[t] - ct.x_traj[t]), 2)
        problem = cp.Problem(cp.Minimize(f))
        problem.solve(solver=cp.CLARABEL)
        K0_traj[t] = K0_t.value

    return K0_traj


## Main loop
for iter in range(max_iter):
    print("Main iteration", iter + 1)
    if iter == 0:
        #####################initialize traj and K0############################################
        x_traj = ct.x_traj
        u_traj = ct.u_traj
        ## traj gen
        [x_traj, u_traj, A_list, B_list, F_list] = traj_gen(x_traj, u_traj, Q_traj, K_traj, iter)
        ## get K0
        K_traj = K0_fcn(x_traj, u_traj)
        ## simulate trajs
        [x_traj_sim, u_traj_sim] = traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, True)
        ## get true matrices
        [A_list_sim, B_list_sim, F_list_sim] = linearization.linearize(x_traj_sim, u_traj_sim, W_traj)
        A_list = np.array(A_list)
        A_list_sim = np.array(A_list_sim)
        ## compute Y_traj
        Y_traj = np.zeros((T, m, n))
        for t in range(T - 1):
            Y_traj[t] = K_traj[t] @ Q_traj[t]
        ## find local Lip const
        gamma_traj = lipschitz_estimator(x_traj, u_traj, mode)

        #####################end of initialization############################################
    ## funnel update
    [Q_traj, Y_traj, K_traj] = funnel_gen(x_traj, u_traj, A_list_sim, B_list_sim, F_list_sim, Q_traj, Y_traj, C, D, E,
                                          G, gamma_traj)
    ## simulate trajs
    [x_traj_sim, u_traj_sim] = traj_sim(x_traj, u_traj, W_traj, K_traj, Q_traj, True)
    ## get true matrices
    [A_list_sim, B_list_sim, F_list_sim] = linearization.linearize(x_traj_sim, u_traj_sim, W_traj)
    ## traj update
    [x_traj, u_traj, A_list, B_list, F_list] = traj_gen(x_traj, u_traj, Q_traj, K_traj, iter)
    ## find local Lip const
    gamma_traj = lipschitz_estimator(x_traj_sim, u_traj_sim, mode)
