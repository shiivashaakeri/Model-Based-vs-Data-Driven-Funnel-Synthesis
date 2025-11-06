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
        [x_traj, u_traj, A_list, B_list, F_list] = traj_gen(x_traj, u_traj, Q_traj, K_traj, iter)
        K_traj = K0_fcn(x_traj, u_traj)

        ## simulate the traj by K0
        x_traj_sim = np.zeros((T, n))
        u_traj_sim = np.zeros((T-1,m))
        x_traj_sim[0] = ct.x_0
        for t in range(T - 1):
            u_t = u_traj[t] + K_traj[t] @ (x_traj_sim[t] - x_traj[t])
            u_traj_sim[t] = u_t
            x_traj_sim[t + 1] = Integrator.RK4(dt, x_traj_sim[t], u_t, W_traj[t])
        # plotting_fcn(x_traj_sim, u_traj, Q_traj)
        [A_list_sim, B_list_sim, F_list_sim] = linearization.linearize(x_traj_sim,u_traj_sim,W_traj)
        A_list = np.array(A_list)
        A_list_sim = np.array(A_list_sim)
        ## compute Y_traj
        Y_traj = np.zeros((T, m, n))
        for t in range(T - 1):
            Y_traj[t] = K_traj[t] @ Q_traj[t]
        #
        ## find local Lip const
        gamma_traj = lipschitz_estimator(x_traj, u_traj, mode)

        #####################end of initialization############################################
    ## funnel update
    [Q_traj, Y_traj, K_traj] = funnel_gen(x_traj, u_traj, A_list_sim, B_list_sim, F_list_sim, Q_traj, Y_traj, C, D, E, G,gamma_traj)

    ## simulate the traj
    if iter % 1 == 0 and iter >= 0:
        x_traj_sim = np.zeros((T, n))
        x_traj_sim[0] = ct.x_0
        for t in range(T - 1):
            u_t = u_traj[t] + K_traj[t] @ (x_traj_sim[t] - x_traj[t])
            x_traj_sim[t + 1] = Integrator.RK4(dt, x_traj_sim[t], u_t, W_traj[t])
        plotting_fcn(x_traj_sim, u_traj, Q_traj)

    ## traj update
    [x_traj, u_traj, A_list, B_list, F_list] = traj_gen(x_traj, u_traj, Q_traj, K_traj, iter)

    ## find local Lip const
    gamma_traj = lipschitz_estimator(x_traj,u_traj,mode)

    ## traj update
