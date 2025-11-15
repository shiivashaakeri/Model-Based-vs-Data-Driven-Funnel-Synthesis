import jax
import numpy as np
import scipy.linalg as la
from numpy import linalg as LA  # noqa: N812
from util import Integrator
from util import const as ct

jax.config.update("jax_enable_x64", True)

T = ct.T
n = ct.n
m = ct.m
nw = ct.nw
C = ct.C_u
D = ct.D_u
gamma_min = 0.0001
gamma_default = 0.001
gamma_max = 0.08


# input_list = [A_list_sim, B_list_sim, F_list_sim, x_traj_sim, K_traj, Q_traj, u_traj_sim]
def lipschitz_estimator(input_list, mode):  # noqa: ARG001
    RK4_jit = jax.jit(Integrator.RK4, static_argnums=0)
    ## unpack data
    A_list_sim = input_list[0]
    B_list_sim = input_list[1]
    F_list_sim = input_list[2]
    x_traj_sim = input_list[3]
    K_traj = input_list[4]
    Q_traj = input_list[5]
    u_traj_sim = input_list[6]

    gamma_traj = np.zeros(T - 1)

    for t in range(T - 1):
        gamma_traj[t] = gamma_default
        ## default value if K is all zeros
        if (K_traj[t] == np.zeros([m, n])).all():
            continue
        [eta_samples_t, w_samples_t] = get_samples_t(Q_traj[t])
        A_t = A_list_sim[t]
        B_t = B_list_sim[t]
        F_t = F_list_sim[t]
        K_t = K_traj[t]


        ## loop over samples at t
        gamma_t = np.zeros(100)
        ## use jax to compute the propagation
        x_sample_t = x_traj_sim[t] + eta_samples_t
        u_sample_t = np.zeros([len(eta_samples_t[:, 0]),m])
        for s in range(len(eta_samples_t[:, 0])):
            u_sample_t[s] = u_traj_sim[t] + K_t @ eta_samples_t[s]
        x_sample_tp1 = jax.vmap(
            lambda x, u, w: RK4_jit(ct.dt, x, u, w),
            in_axes=(0, 0, 0)
        )(x_sample_t, u_sample_t, w_samples_t)

        x_sample_tp1 = np.array(x_sample_tp1)
        for s in range(len(eta_samples_t[:, 0])):
            eta_sample_t_s = eta_samples_t[s]
            w_sample_t_s = w_samples_t[s]

            ## propagate the sample point
            x_sample_tp1_s = x_sample_tp1[s]
            # x_prop = Integrator.RK4(ct.dt, x_traj_sim[t], u_traj_sim[t], np.zeros(nw))
            eta_sample_tp1_s = x_sample_tp1_s - x_traj_sim[t + 1]
            # LHS = eta_sample_tp1_s - (A_t + B_t @ K_t) @ eta_sample_t_s - F_t @ w_sample_t_s + (
            #             x_traj_sim[t + 1] - x_prop)
            LHS = eta_sample_tp1_s - (A_t + B_t @ K_t) @ eta_sample_t_s - F_t @ w_sample_t_s
            mu = (C + D @ K_t) @ eta_sample_t_s
            gamma_t[s] = LA.norm(LHS) / LA.norm(mu)
        gamma_traj[t] = np.max(gamma_t) / 1
        gamma_traj[t] = max(gamma_traj[t], gamma_min)
        gamma_traj[t] = min(gamma_max, gamma_traj[t])
        print("Lip est progress: ", t / T * 100, "Lip_t: ", gamma_traj[t])
    return gamma_traj / 1


def get_samples_t(Q_t):
    Q_sqrt = la.sqrtm(Q_t)
    N = 5  ## sample number
    ## generate the sample points on the unit sphere
    eta_samples_t = np.zeros([N * N * (N - 1), n])  ## state sample
    w_samples_t = np.zeros([(2 * N) * (2 * N), nw])  ## noise sample
    count_eta = 0
    eps = 0.0001
    for i in np.linspace(-1, 1, N):
        for j in np.linspace(-1, 1, N):
            for k in np.linspace(-1, 1, N - 1):
                eta_samples_t[count_eta] = np.array([eps + i, j, k]) / LA.norm(np.array([eps + i, j, k]), 2)
                ## project back to the ellipsoid
                eta_samples_t[count_eta] = Q_sqrt @ eta_samples_t[count_eta]
                count_eta += 1
    count_w = 0
    for i in np.linspace(-1, 1, 2 * N):
        for j in np.linspace(-1, 1, 2 * N):
            w_samples_t[count_w] = np.array([i, j + eps]) / LA.norm(np.array([i, j + eps]), 2)
            count_w += 1
    return eta_samples_t, w_samples_t
