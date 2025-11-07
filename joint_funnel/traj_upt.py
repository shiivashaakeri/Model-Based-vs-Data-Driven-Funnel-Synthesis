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

## obstacles
num_obs = ct.num_obs
obs = ct.obs
obs_r = ct.obs_r

## global variables
n = ct.n
m = ct.m
dt = ct.dt
T = ct.T
x_0 = ct.x_0
x_des = ct.x_des
x_traj = np.zeros([T, n])
x_traj[0] = x_0
x_traj_euler = np.zeros([T, n])
x_traj_euler[0] = x_0
u_traj = np.zeros([T - 1, m])


def ini(x_0, x_des, x_traj):
    x_traj = np.linspace(x_0, x_des, T)
    return x_traj


def linearization(integrator, dt, x_t, u_t, W_t) -> tuple:
    f = lambda x, u, W: integrator(dt, x, u, W)
    # Compute the Jacobian of f(x, u) with respect to x (A matrix)
    A_k = jax.jacobian(lambda x: f(x, u_t, W_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    B_k = jax.jacobian(lambda u: f(x_t, u, W_t))(u_t)
    # Compute the Jacobian of f(x, u) with respect to W (F matrix)
    W_k = jax.jacobian(lambda W: f(x_t, u_t, W))(W_t)
    return A_k, B_k, W_k


linearization_jit = jax.jit(linearization, static_argnums=0)


def cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s):
    ## terminal cost
    f0 = 1000 * cp.norm(x_traj[T - 1, 0:2] + d[T - 1, 0:2] - x_des[0:2], 2)

    ## running cost
    for t in range(T - 1):
        ## for objective
        # f0 += cp.norm(x_traj[t,0:2] + d[t,0:2],2)
        f0 += cp.norm(u_traj[t, 0] + w[t, 0], 2)
        ## for constraints
        f0 += 1000 * cp.norm(v[t], 1)
        ## for obs
        for j in range(num_obs):
            f0 += 1000 * cp.abs(s[t, j])
        ## regularization
        f0 += 10 * cp.norm(w[t], 2)

    return f0


def solve_subproblem(A_list, B_list, trajs, x_des) -> tuple:
    x_traj = trajs[0]
    u_traj = trajs[1]
    W_traj = trajs[2]
    Q_traj = trajs[3]
    K_traj = trajs[4]
    d = cp.Variable([T, n])
    w = cp.Variable([T - 1, m])
    v = cp.Variable([T, n])
    s = cp.Variable([T, num_obs])
    ## starting constraints
    constraints = [d[0] == np.zeros(n)]
    ## terminal constraints
    # constraints.append(d[-1]+x_traj[-1] == x_des)

    for t in range(T - 1):
        x_t = x_traj[t]
        x_tp1 = x_traj[t + 1]
        u_t = u_traj[t]
        w_t = w[t]
        W_t = W_traj[t]
        d_t = d[t]
        d_tp1 = d[t + 1]
        v_t = v[t]
        s_t = s[t]
        A_t = A_list[t]
        B_t = B_list[t]
        Q_t = Q_traj[t]
        K_t = K_traj[t]
        f_t = Integrator.RK4(dt, x_t, u_t, W_t)
        constraints.append(x_tp1 + d_tp1 == A_t @ d_t + B_t @ w_t + f_t + np.diag(np.ones(n)) @ v_t)

        ## control rate constraints
        control_funnel = sqrtm(K_t @ Q_t @ K_t.T)
        a_t = np.array([1, 0])
        constraints.append(LA.norm(control_funnel @ a_t, 2) + cp.norm(u_t + w_t, 1) <= 2)

        ## control constraints
        # a_t = np.array([[-1], [0]])
        # rate_LHS = -u_t[0] - w_t[0]
        # rate_LHS += LA.norm(control_funnel @ a_t, 2)
        # constraints.append(rate_LHS <= 0)

        constraints.append(u_t[0] + w_t[0] >= 0)
        ## obs constraints
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:2] - obs_j, 2) ** 2
            a = - 2 * (x_t[0:2] - obs_j)
            LHS = h_j + a @ d_t[0:2]  ## obs constraints
            Q_t_root = sqrtm(Q_t[0:2, 0:2])
            LHS += LA.norm(Q_t_root @ a, 2)
            constraints.append(LHS <= s_t[j])
            # constraints.append(h_j - 2 * (x_t[0:2] - obs_j) @ d_t[0:2] <= s_t[j])
            constraints.append(s_t[j] >= 0)

    f0 = cost_subproblem_fun(x_traj, u_traj, x_des, d, w, v, s)
    problem = cp.Problem(cp.Minimize(f0), constraints)
    problem.solve(solver=cp.CLARABEL)
    d_traj = d.value
    w_traj = w.value
    true_cost = 0
    subproblem_cost = problem.value

    return d_traj, w_traj, true_cost, subproblem_cost


def traj_gen(x_traj, u_traj, Q_traj, K_traj, main_iter) -> tuple:
    if main_iter == 0:
        max_iter = 20
    else:
        max_iter = 20
    W_traj = np.zeros([T - 1, ct.nw])
    subproblem_cost_old = 0
    for iter in range(max_iter):
        [A_list, B_list, F_list] = jax.vmap(
            lambda x, u, W: linearization_jit(Integrator.RK4, dt, x, u, W),
            in_axes=(0, 0, 0)
        )(x_traj[0:T - 1, :], u_traj, W_traj)

        trajs = [x_traj, u_traj, W_traj, Q_traj, K_traj]
        [d_traj, w_traj, true_cost, subproblem_cost] = solve_subproblem(A_list, B_list, trajs, x_des)
        print("Traj upt iteration:  ", iter + 1, "subproblem cost:    ", subproblem_cost)
        ## update
        if d_traj.any() != None:
            x_traj += d_traj
            u_traj += w_traj
        print("cost diff:   ", np.abs(subproblem_cost_old - subproblem_cost))
        if np.abs(subproblem_cost_old - subproblem_cost) <= 0.1:
            break
        subproblem_cost_old = subproblem_cost
        ## plotting
        # if iter % 5 == 0:
        #     plotting_fcn(x_traj, u_traj)
    # plotting_fcn(x_traj, u_traj, Q_traj)
    return x_traj, u_traj, A_list, B_list, F_list


def plotting_fcn(x_traj, u_traj, Q_traj):
    fig, ax = plt.subplots()
    for t in range(T - 1):
        theta = x_traj[t, 2]
        leng = 0.4
        heading_vector = np.array([[x_traj[t, 0], x_traj[t, 1]],
                                   [x_traj[t, 0] + leng * np.cos(theta), x_traj[t, 1] + leng * np.sin(theta)]])

        ax.plot(heading_vector[:, 0], heading_vector[:, 1], "r")
        ax.plot(x_traj[t, 0], x_traj[t, 1], "g.")
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
        ell = Ellipse(xy=(x_traj[t, 0], x_traj[t, 1]), width=2 * np.sqrt(vals[0]), height=2 * np.sqrt(vals[1]),
                      angle=angle_deg, fill=True, alpha=0.5)
        ax.add_patch(ell)
        ax.set_aspect('equal', adjustable='box')

        plt.pause(0.01)
        if t == T - 2:
            plt.show()
    plt.clf()

# [x_traj, u_traj,A_list, B_list] = traj_gen()

# ## simulate the traj and check linearization
# for t in range(T - 1):
#     u_traj[t] = np.array([-0.3 * t * dt, -0.5])
#     x_traj[t + 1] = Integrator.RK4(dt, x_traj[t], u_traj[t])
#     x_traj_euler[t + 1] = Integrator.Euler(dt, x_traj[t], u_traj[t])
# [A_list, B_list] = jax.vmap(
#     lambda x, u: linearization_jit(Integrator.RK4, dt, x, u),
#     in_axes=(0, 0)
# )(x_traj[0:T - 1, :], u_traj)
# for t in range(T - 1):
#     plt.plot(x_traj[t, 0], x_traj[t, 1], "r.")
#     plt.pause(0.1)
# plt.show()
