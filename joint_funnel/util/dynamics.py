import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
import jax.numpy as jnp


def unicycle(x_t: jnp.array, u_t: jnp.array, W_t: jnp.array) -> jnp.array:
    ## state: px, py, theta
    ## control: v, omega
    theta = x_t[2]
    v = u_t[0]
    omega = u_t[1]
    ## type 1 uncertainty
    x_dot = jnp.array([v * jnp.cos(theta) + 0.1 * W_t[0],
                       v * jnp.sin(theta) + 0.1 * W_t[1],
                       omega ])
    return x_dot
