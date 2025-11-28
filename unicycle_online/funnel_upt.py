import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from unicycle_online.util.const import t_horizon, t_steps, W_traj_s
from util import Integrator, dynamics
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from util import const as ct

T = ct.T
n = ct.n
m = ct.m
## funnel constants
alpha = 0.999
lambda_omega = 0.1
## for maximize
# w_Q = -1
## for minimize
w_Q = 0
w_K = 0
w_tr = 1
nw = ct.nw
n_p = ct.n_p
n_q = ct.n_q
num_obs = ct.num_obs
obs = ct.obs
obs_r = ct.obs_r
gamma1 = ct.gamma1


def funnel_cost(Q, Q_traj, Y, Y_traj, mu_Q, mu_K, s, s0, sf):
    f = 0
    f += 1000 * (s0 + sf)
    for t in range(t_steps):
        ## state funnel penalty
        f += w_Q * mu_Q[t]
        f += w_tr * (cp.norm(Q[t] - Q_traj[t], "fro"))
        if t < t_steps - 1:
            ## control funnel penalty
            f += w_K * mu_K[t]
            ## regularization for the funnels
            f += w_tr * (cp.norm(Y[t] - Y_traj[t], "fro"))
            ## slack penalty
            f += 1000000 * -(s[t])

    return f


def funnel_problem(current_t, x_traj, u_traj, A_traj, B_traj, F_traj, Q_traj, Y_traj, C, D, E, G, gamma_traj):
    Q = cp.Variable([t_steps, n, n])
    Y = cp.Variable([t_steps - 1, m, n])  ## Y = BK
    s0 = cp.Variable(nonneg=True)
    sf = cp.Variable(nonneg=True)
    s_LMI = cp.Variable(t_steps - 1, nonpos=True)
    mu_Q = cp.Variable(t_steps, nonneg=True)
    mu_K = cp.Variable(t_steps - 1, nonneg=True)
    mu_P = cp.Variable(t_steps - 1, pos=True)
    ## Initial constraints
    if current_t == 0:
        constraints = [Q[0] >> ct.Q0_traj[0]]
    else:
        constraints = []
    # constraints = [ct.Q0_traj[0] - Q[0] << s0 * np.eye(n)]
    # constraints.append(mu_Q[0] <= 0.11)
    # constraints = []
    ## terminal constraints
    if current_t >= T - 2:
        constraints.append(Q[-1] - ct.Q0_traj[-1] << sf * np.eye(n))  ## fixed final funnel
    # constraints.append(Q[-2] << ct.Q0_traj[-1])  ## fixed final funnel
    # constraints.append(Q[-2] - ct.Q0_traj[-1] << sf * np.eye(n))  ## fixed final funnel
    # constraints.append(mu_Q[-2]<= 0.05)
    # print(gamma_traj)
    # print(u_traj[:,0])
    for t in range(t_steps - 1):
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
        s_LMI_t = s_LMI[t]
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
        # constraints.append(mu_Q_t <= 0.5)
        ## funnel size constraints
        constraints.append(mu_Q_t <= 0.5)

        ## control funnel constraints
        I_K = np.eye(m)
        C_row1 = cp.hstack((I_K * mu_K_t, Y_t))
        C_row2 = cp.hstack((Y_t.T, Q_t))
        C_matrix = cp.vstack((C_row1, C_row2))
        constraints.append(C_matrix >> eps * np.eye(m + n))

        LMI11 = alpha * Q_t - lambda_omega * Q_t  ## n by n
        LMI21 = np.zeros([nw, n])
        LMI31 = A_t @ Q_t + B_t @ Y_t
        LMI22 = lambda_omega * np.eye(nw)  ## nw b nw
        LMI32 = F_t
        LMI33 = Q_tp1  ## n by n
        LMI = cp.bmat([[LMI11, LMI21.T, LMI31.T],
                       [LMI21, LMI22, LMI32.T],
                       [LMI31, LMI32, LMI33]])
        if u_t[0] >= 0.001:
            constraints.append(LMI >> 1 * s_LMI_t * np.eye(2 * n + nw))

        ## add the compatibility constraint for the past funnel
        if current_t != 0 and t == 0:
            LMI11 = alpha * Q_traj[current_t] - lambda_omega * Q_traj[current_t]  ## n by n
            LMI = cp.bmat([[LMI11, LMI21.T, LMI31.T],
                           [LMI21, LMI22, LMI32.T],
                           [LMI31, LMI32, LMI33]])
            constraints.append(LMI >> 1 * s_LMI_t * np.eye(2 * n + nw))

        ## Nonlinear DLMI constraints for prediction
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
        # if u_t[0] > 0.000001:
        #     constraints.append(LMI >> 1 * s_LMI_t * I_lmi)

        ## obs constraints
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:2] - obs_j, 2) ** 2
            a_t = - 2 * (x_t[0:2] - obs_j)
            a_t_col = np.reshape(a_t, [2, 1])
            b_t = a_t @ x_t[0:2] - h_j

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
        a_t = np.array([[-1], [0]])
        b_t = ct.u1_min
        BB11 = np.array([(b_t - a_t.T @ u_t)])
        BB_row1 = cp.hstack((BB11, a_t.T @ Y_t))
        BB_row2 = cp.hstack((Y_t.T @ a_t, Q_t))
        BB_matrix = cp.vstack((BB_row1, BB_row2))
        constraints.append(BB_matrix >> 0)

        ## u <= 2
        a_t2 = np.array([[1], [0]])
        b_t2 = ct.u1_max + 0.1
        BB211 = np.array([(b_t2 - a_t2.T @ u_t)])
        BB_row21 = cp.hstack((BB211, a_t2.T @ Y_t))
        BB_row22 = cp.hstack((Y_t.T @ a_t2, Q_t))
        BB2_matrix = cp.vstack((BB_row21, BB_row22))
        constraints.append(BB2_matrix >> 0)

        ## omega <=  2
        a_t3 = np.array([[1], [0]])
        b_t3 = ct.u2_max + 0.01
        BB311 = np.array([(b_t3 - a_t3.T @ u_t)])
        BB_row31 = cp.hstack((BB311, a_t3.T @ Y_t))
        BB_row32 = cp.hstack((Y_t.T @ a_t3, Q_t))
        BB3_matrix = cp.vstack((BB_row31, BB_row32))
        constraints.append(BB3_matrix >> 0)

        ## omega >= -2
        a_t4 = np.array([[-1], [0]])
        b_t4 = ct.u2_max + 0.01
        BB411 = np.array([(b_t4 - a_t4.T @ u_t)])
        BB_row41 = cp.hstack((BB411, a_t4.T @ Y_t))
        BB_row42 = cp.hstack((Y_t.T @ a_t4, Q_t))
        BB4_matrix = cp.vstack((BB_row41, BB_row42))
        constraints.append(BB4_matrix >> 0)

    f0 = funnel_cost(Q, Q_traj, Y, Y_traj, mu_Q, mu_K, s_LMI, s0, sf)
    problem = cp.Problem(cp.Minimize(f0), constraints)
    problem.solve(solver=cp.CLARABEL)
    Q_traj_t = Q.value
    Y_traj_t = Y.value
    K_traj_t = np.zeros([t_horizon, m, n])
    for t in range(t_horizon):
        K_traj_t[t] = Y_traj_t[t] @ LA.inv(Q_traj_t[t])
        control_funnel_t = K_traj_t[t] @ Q_traj_t[t] @ K_traj_t[t].T
    #     print("Control funnel size: ", LA.norm(control_funnel_t,2))
    # for t in range(t_steps):
    #     print("State funnel size: ",LA.norm(Q_traj_t[t], 2))
    return Q_traj_t, Y_traj_t, K_traj_t, problem.value


def funnel_gen(x_traj, u_traj, A_traj, B_traj, F_traj, Q_traj, Y_traj, C, D, E, G, gamma_traj):
    x_traj_bar = x_traj.copy()
    fig, ax = plt.subplots()
    K_traj = ct.K0_traj

    ## true system traj
    x_traj_true = np.zeros([T, n])
    x_traj_true[0] = x_traj[0] + np.ones(n) * 0.02
    # x_traj_true[0,1] = x_traj[0,1] + 0.1
    for t in range(T - 1):
        print("progress: ", t / T)
        steps_to_concat = t + t_steps - T
        ## references trajs
        idx_step = np.minimum(t + t_steps, T)
        idx_hori = np.minimum(t + t_horizon, T - 1)
        x_traj_t = x_traj[t:idx_step]
        u_traj_t = u_traj[t:idx_hori]
        A_traj_t = A_traj[t:idx_hori]
        B_traj_t = B_traj[t:idx_hori]
        F_traj_t = F_traj[t:idx_hori]
        Q_traj_t = Q_traj[t:idx_step]
        Y_traj_t = Y_traj[t:idx_hori]
        gamma_traj_t = gamma_traj[t:idx_hori]
        if steps_to_concat > 0:
            for s in range(steps_to_concat):
                x_traj_t = np.vstack((x_traj_t, x_traj[-1]))
                u_traj_t = np.vstack((u_traj_t, u_traj[-1]))
                A_traj_t = np.vstack((A_traj_t, A_traj[-1:]))
                B_traj_t = np.vstack((B_traj_t, B_traj[-1:]))
                F_traj_t = np.vstack((F_traj_t, F_traj[-1:]))
                Q_traj_t = np.vstack((Q_traj_t, Q_traj[-1:]))
                Y_traj_t = np.vstack((Y_traj_t, Y_traj[-1:]))
                gamma_traj_t = np.hstack((gamma_traj_t, gamma_traj[-1]))
        ######## Get predicted funnel with time horizon
        ## Q_traj_t is hor + 1 and Y,K_traj_t is hor
        [Q_traj_t, Y_traj_t, K_traj_t, prob_cost] = funnel_problem(t, x_traj_t, u_traj_t, A_traj_t, B_traj_t, F_traj_t,
                                                                   Q_traj,
                                                                   Y_traj_t, C,
                                                                   D, E, G, gamma_traj_t)
        ## update the funnel
        idx_step_cat = np.minimum(t_steps, T - t - t_steps)
        if idx_step_cat >= 0:
            Q_traj[t:t + t_steps] = Q_traj_t[0:0+t_steps]
            Y_traj[t:t + t_horizon] = Y_traj_t
            K_traj[t:t + t_horizon] = K_traj_t

        ## propagate system
        u_traj[t] = u_traj[t] + K_traj_t[0] @ (x_traj_true[t] - x_traj[t])
        x_traj_true[t + 1] = Integrator.RK4(ct.dt, x_traj_true[t], u_traj[t], W_traj_s[0, t])

        ## plotting the circles
        theta_grid = np.linspace(0, 2 * np.pi, 101)
        for r in range(num_obs):
            x_grid = np.cos(theta_grid) * ct.obs_r + ct.obs[r, 0]
            y_grid = np.sin(theta_grid) * ct.obs_r + ct.obs[r, 1]
            ax.plot(x_grid, y_grid)

        ## plotting the realtime funnel generation
        for tt in range(t_steps):
            Q_t = Q_traj_t[tt, 0:2, 0:2]
            # Eigen-decomposition (ascending order from eigh)
            vals, vecs = LA.eigh(Q_t)
            order = vals.argsort()[::-1]  # sort descending so index 0 is largest
            vals = vals[order]
            vecs = vecs[:, order]
            vmax = vecs[:, 0]
            angle_deg = np.degrees(np.arctan2(vmax[1], vmax[0]))
            ell = Ellipse(xy=(x_traj_t[tt, 0], x_traj_t[tt, 1]), width=2 * np.sqrt(vals[0]),
                          height=2 * np.sqrt(vals[1]),
                          angle=angle_deg, fill=True, alpha=0.5)
            ax.add_patch(ell)
        ax.plot(x_traj_true[0:t, 0], x_traj_true[0:t, 1], "r.", label="true traj")
        ax.plot(x_traj_bar[0:t, 0], x_traj_bar[0:t, 1], "g.", label="ref traj")
        print(x_traj_true[t], x_traj_bar[t])
        ax.plot(ct.x_0[0], ct.x_0[1], "b.", markersize=10)
        ax.plot(ct.x_des[0], ct.x_des[1], "r.", markersize=10)
        ax.legend()
        ax.set_xlim([0, 12])
        ax.set_ylim([0, 7])
        if t == T - 2:
            plt.show()
        else:
            plt.pause(0.01)
        ax.clear()
        ## update the current traj
        x_traj[t] = x_traj_true[t].copy()

    return Q_traj, Y_traj, K_traj
