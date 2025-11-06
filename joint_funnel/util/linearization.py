import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
from .Integrator import RK4
from .const import dt,T
import jax
import cvxpy as cp
jax.config.update('jax_enable_x64', True)



def linearization_fun(integrator, dt, x_t, u_t, W_t) -> tuple:
    f = lambda x, u, W: integrator(dt, x, u, W)
    # Compute the Jacobian of f(x, u) with respect to x (A matrix)
    A_k = jax.jacobian(lambda x: f(x, u_t, W_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    B_k = jax.jacobian(lambda u: f(x_t, u, W_t))(u_t)
    # Compute the Jacobian of f(x, u) with respect to W (F matrix)
    W_k = jax.jacobian(lambda W: f(x_t, u_t, W))(W_t)
    return A_k, B_k, W_k

linearization_jit = jax.jit(linearization_fun, static_argnums=0)

def linearize(x_traj,u_traj,W_traj):

    [A_list, B_list, F_list] = jax.vmap(
        lambda x, u, W: linearization_jit(RK4, dt, x, u, W),
        in_axes=(0, 0, 0)
    )(x_traj[0:T - 1, :], u_traj, W_traj)

    return A_list, B_list, F_list