import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from util import Integrator, dynamics
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from util import const as ct

T = ct.T


## for unicycles
def lipschitz_estimator(x_traj, u_traj, mode):
    gamma_traj = np.zeros(T - 1)
    if mode == 0:
        for t in range(T - 1):
            x_t = x_traj[t]
            u_t = u_traj[t]
            dphi = np.array([[-u_t[0] * np.sin(x_t[2]) , np.cos(x_t[2])],
                             [u_t[0] * np.cos(x_t[2]) , np.sin(x_t[2])]])
            gamma_traj[t] = LA.norm(dphi,2) * ct.dt **2 /2
    return gamma_traj
