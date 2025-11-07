import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA

from util import Integrator, dynamics
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from util import const as ct

T = ct.T
n = ct.n
m = ct.m
## funnel constants
alpha = 0.9
lambda_omega = 0.1
## for maximize
# w_Q = -1
## for minimize
w_Q = 1
w_K = 1
w_tr = 1
nw = ct.nw
n_p = ct.n_p
n_q = ct.n_q
num_obs = ct.num_obs
obs = ct.obs
obs_r = ct.obs_r
gamma1 = ct.gamma1


def funnel_cost(Q, Q_traj, Y, Y_traj, mu_Q, mu_K,s):
    f = 0

    for t in range(T):
        ## state funnel penalty
        f += w_Q * mu_Q[t]

        if t < T - 1:
            ## control funnel penalty
            f += w_K * mu_K[t]
            ## regularization for the funnels
            f += w_tr * (cp.norm(Q[t] - Q_traj[t], "fro") + cp.norm(Y[t] - Y_traj[t], "fro"))
            ## slack penalty
            f += 1000000 *  -(s[t])

    return f







def funnel_gen(x_traj, u_traj, A_traj, B_traj, F_traj, Q_traj, Y_traj, C, D, E, G,gamma_traj):
    Q = cp.Variable([T, n, n])
    Y = cp.Variable([T - 1, m, n])  ## Y = BK
    s = cp.Variable(T-1 , nonpos=True)
    mu_Q = cp.Variable(T, nonneg=True)
    mu_K = cp.Variable(T - 1, nonneg=True)
    mu_P = cp.Variable(T - 1, pos=True)
    ## Initial constraints
    constraints = [Q[0] >> ct.Q0_traj[0]]

    ## terminal constraints
    # constraints.append(Q[-2] << ct.Q0_traj[-1])  ## fixed final funnel
    # constraints.append(mu_Q[-1]<= 0.8)
    # print(gamma_traj)
    # print(u_traj[:,0])
    for t in range(T - 1):
        x_t = x_traj[t]
        u_t = u_traj[t]
        Q_t = Q[t]
        Q_tp1 = Q[t + 1]
        Y_t = Y[t]
        mu_Q_t = mu_Q[t]
        mu_K_t = mu_K[t]
        mu_P_t = mu_P[t]
        A_t = A_traj[t]
        B_t = B_traj[t]
        F_t = F_traj[t]
        s_t = s[t]
        gamma_t = gamma_traj[t]
        ## to insure the invertibility of Q
        eps = 0.000000001
        # constraints.append(Q_t >> np.eye(n) * eps)
        # constraints.append(Q_tp1 >> np.eye(n) * eps)
        ## state funnel constraints
        I_Q = np.eye(n)
        ## for maximize
        # constraints.append(Q_t - mu_Q_t * I_Q >> 0)
        ## for minimize
        constraints.append(Q_t - mu_Q_t * I_Q << 0)
        constraints.append(mu_Q_t <= 0.5)

        ## control funnel constraints
        I_K = np.eye(m)
        C_row1 = cp.hstack((I_K * mu_K_t, Y_t))
        C_row2 = cp.hstack((Y_t.T, Q_t))
        C_matrix = cp.vstack((C_row1, C_row2))
        constraints.append(C_matrix >> eps * np.eye(m + n))

        ## Linear DLMI constraints
        LMI11 = alpha * Q_t - lambda_omega * Q_t  ## n by n
        LMI21 = np.zeros([nw, n])
        LMI31 = A_t @ Q_t + B_t @ Y_t
        LMI22 = lambda_omega * np.eye(nw)  ## nw b nw
        LMI32 = F_t
        LMI33 = Q_tp1  ## n by n
        LMI = cp.bmat([[LMI11, LMI21.T, LMI31.T],
                       [LMI21, LMI22, LMI32.T],
                       [LMI31, LMI32, LMI33]])
        # if u_t[0] >= 0.:
        #     constraints.append(LMI >>0* s_t * np.eye(2 * n + nw))

        ## Nonlinear DLMI constraints
        LMI11 = alpha * Q_t - lambda_omega * Q_t
        LMI21 = np.zeros((n_p, n))
        LMI31 = np.zeros((nw, n))
        LMI41 = A_t @ Q_t + B_t @ Y_t
        LMI51 = C @ Q_t + D @ Y_t

        LMI22 = mu_P_t * np.eye(n_p)
        LMI32 = np.zeros((nw, n_p))
        LMI42 = mu_P_t * E
        LMI52 = np.zeros((n_q, n_p))

        LMI33 = lambda_omega * np.eye(nw)
        LMI43 = F_t
        LMI53 = G

        LMI44 = Q_tp1
        LMI54 = np.zeros((n_q, n))

        LMI55 = mu_P_t * 1 / (gamma_t ** 2) * np.eye(n_q)

        row1 = cp.hstack((LMI11, LMI21.T, LMI31.T, LMI41.T, LMI51.T))
        row2 = cp.hstack((LMI21, LMI22, LMI32.T, LMI42.T, LMI52.T))
        row3 = cp.hstack((LMI31, LMI32, LMI33, LMI43.T, LMI53.T))
        row4 = cp.hstack((LMI41, LMI42, LMI43, LMI44, LMI54.T))
        row5 = cp.hstack((LMI51, LMI52, LMI53, LMI54, LMI55))
        LMI = cp.vstack((row1, row2, row3, row4, row5))
        I_lmi = np.eye(n + n_p + nw + n + n_q)
        if u_t[0] > 0.0001:
            constraints.append(LMI >> 1*s_t * I_lmi)

        ## obs constraints
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:2] - obs_j, 2) ** 2
            a_t = - 2 * (x_t[0:2] - obs_j)
            a_t_col = np.reshape(a_t, [2, 1])
            b_t = a_t @ x_t[0:2] - h_j
            # B_row1 = cp.hstack(((b_t - a_t @ x_t[0:2]) ** 2, a_t @ Q_t[0:2, 0:2].T))
            # B_row2 = cp.hstack((Q_t[0:2, 0:2] @ a_t_col, Q_t[0:2, 0:2]))
            # B_matrix = cp.vstack((B_row1, B_row2))
            # constraints.append(B_matrix >> 0)

            Q2 = Q_t[0:2, 0:2]
            a_row = cp.Constant(a_t).reshape((1, 2))  # 1×2
            x2 = x_t[0:2].reshape((2, 1))  # 2×1 numpy → 2×1

            B11 = (b_t - a_row @ x2) ** 2  # 1×1
            B12 = a_row @ Q2.T  # 1×2

            B_row1 = cp.hstack((B11, B12))  # 1×3
            B_row2 = cp.hstack((Q2 @ a_row.T, Q2))  # (2×1) hstack (2×2) → 2×3

            B_matrix = cp.vstack((B_row1, B_row2))  # 3×3 PSD
            constraints.append(B_matrix >> 0)
        ## control constraints
        ## u >= 0
        a_t = np.array([[-1],[0]])
        b_t = 0.0
        BB11 = np.array([(b_t - a_t.T @ u_t) ])
        BB_row1 = cp.hstack((BB11, a_t.T @ Y_t))
        BB_row2 = cp.hstack((Y_t.T @ a_t, Q_t))
        BB_matrix = cp.vstack((BB_row1, BB_row2))
        constraints.append(BB_matrix >> 0)



    f0 = funnel_cost(Q, Q_traj, Y, Y_traj, mu_Q, mu_K,s)
    problem = cp.Problem(cp.Minimize(f0), constraints)
    problem.solve(solver=cp.CLARABEL)
    if Q.value.any() != None:
        Q_traj = Q.value
        Y_traj = Y.value
    K_traj = np.zeros([T - 1, m, n])
    for t in range(T - 1):
        K_traj[t] = Y_traj[t] @ LA.inv(Q_traj[t])
    print("funnel problem cost: ",problem.value)

    return Q_traj, Y_traj, K_traj
