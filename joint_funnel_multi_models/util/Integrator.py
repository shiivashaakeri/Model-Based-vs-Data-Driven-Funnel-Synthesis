import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
import jax.numpy as jnp
from .dynamics import unicycle, quadrotor


def RK4(dt: jnp.array, x_t: jnp.array, u_t: jnp.array, W_t: jnp.array) -> jnp.array:
    ## zoh for control
    n = len(x_t)
    m = len(u_t)
    k1 = unicycle(x_t, u_t, W_t)
    k2 = unicycle(x_t + dt * k1 / 2, u_t, W_t)
    k3 = unicycle(x_t + dt * k2 / 2, u_t, W_t)
    k4 = unicycle(x_t + dt * k3, u_t, W_t)
    # k1 = quadrotor(x_t, u_t, W_t)
    # k2 = quadrotor(x_t + dt * k1 / 2, u_t, W_t)
    # k3 = quadrotor(x_t + dt * k2 / 2, u_t, W_t)
    # k4 = quadrotor(x_t + dt * k3, u_t, W_t)
    x_tp1 = x_t + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    q_tp1 = x_tp1[6:10] / jnp.linalg.norm(x_tp1[6:10], 2)
    x_tp1 = x_tp1.at[6:10].set(q_tp1)
    return x_tp1


def Euler(dt: jnp.array, x_t: jnp.array, u_t: jnp.array, W_t: jnp.array) -> jnp.array:
    x_tp1 = x_t + dt * unicycle(x_t, u_t, W_t)
    return x_tp1
