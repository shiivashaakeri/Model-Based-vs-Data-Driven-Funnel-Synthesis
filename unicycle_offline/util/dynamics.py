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
                       v * jnp.sin(theta) + 0.2 * W_t[1],
                       omega])
    return x_dot


def quat_conj(q: jnp.array) -> jnp.array:
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(p: jnp.array, q: jnp.array) -> jnp.array:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q: jnp.array, v: jnp.array) -> jnp.array:
    # rotate 3-vector v from body to inertial using quaternion q
    v_quat = jnp.concatenate([jnp.array([0.0]), v])
    v_rot = quat_mul(quat_mul(q, v_quat), quat_conj(q))
    return v_rot[1:]


def quadrotor(x_t: jnp.array, u_t: jnp.array, W_t: jnp.array) -> jnp.array:
    ## states: position (NED), velocity(NED), quaternion (body), angular rates (body)
    ## note that the z axis is pointing down
    ## control: thrust, torque

    ## constants and matrices
    m = 0.0293
    J = jnp.diag(jnp.array([1.8203e-5, 1.8186e-5, 3.4484e-5]))*100

    ## current states
    pos = x_t[0:3]
    vel = x_t[3:6]
    q = x_t[6:10]
    omega = x_t[10:13]

    ## current control
    T = u_t[0]
    tau = u_t[1:4]

    ## forces in the NED frame
    f_g_i = jnp.array([0, 0, 9.81 * m])
    f_T_b = jnp.array([0, 0, -T])  ## in body frame
    f_T_i = quat_rotate(q, f_T_b)
    f_net_i = f_T_i + f_g_i
    # f_net_i = f_T_b+ f_g_i

    ## states rates
    Omega = jnp.array([[0, -omega[0], -omega[1], -omega[2]],
                       [omega[0], 0, omega[2], -omega[1]],
                       [omega[1], -omega[2], 0, omega[0]],
                       [omega[2], omega[1], -omega[0], 0]])
    v_dot_i = f_net_i / m
    omega_dot_b = jnp.linalg.inv(J) @ (tau - jnp.cross(omega, (J @ omega)))
    q_dot_i = 0.5 * Omega @ q
    pos_dot_i = vel

    x_dot = jnp.hstack((pos_dot_i, v_dot_i, q_dot_i, omega_dot_b))

    return x_dot
