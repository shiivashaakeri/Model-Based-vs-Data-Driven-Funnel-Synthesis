import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA

## select dynamcis
run = "unicycle"
# run = "quadrotor"

####### global variables
## simulation samples
if run == "unicycle":
    ## obstacles
    num_obs = 2
    obs = np.array([[4, 3], [8, 3]])
    obs_r = 1

    N = 10
    n = 3
    m = 2
    nw = 2
    n_p = 2
    n_q = 2
    tf = 8
    T = 61
    dt = tf / T
    time_traj = np.linspace(0, tf, T)
    gamma1 = 0.4
    ## initial and final states
    x_0 = np.array([1.0, 1.0, 0])
    x_des = np.array([10, 5.5, 0])
    ## initial funnel
    Q0_traj = np.zeros([T, n, n])
    K0_traj = np.zeros([T - 1, m, n])
    for t in range(T):
        Q0_traj[t] = np.diag([1, 1, 0.1]) * 0.1
    Q0_traj[-1] = np.diag([1, 1, 0.1]) * 0.1
    ## control constraints
    u1_max = 2
    u1_min = 0
    u2_max = 2
    ########### channel selections
    ## Unicycle
    C_u = np.array([[0, 0, 1],
                    [0, 0, 0]])
    D_u = np.array([[0, 0],
                    [1, 0]])
    E_u = np.array([[1, 0],
                    [0, 1],
                    [0, 0]])
    ## Unicycle 1
    G_u1 = np.zeros([2, 2])
## quadrotor
elif run == "quadrotor":
    ## obstacles
    num_obs = 2
    obs = np.array([[2, 2, -1.5], [4, 4, -3.5]])
    obs_r = 0.5

    n = 13
    m = 4
    g = 9.81
    nw = 2
    n_p = 2
    n_q = 2
    tf = 4
    T = 51
    dt = tf / T
    time_traj = np.linspace(0, tf, T)
    ## initial and final states
    x_0 = np.array([1.0, 1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    x_des = np.array([5.0, 5.0, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    ## initial funnel
    Q0_traj = np.zeros([T, n, n])
    K0_traj = np.zeros([T - 1, m, n])

x_traj = np.zeros([T, n])
x_traj[0] = x_0
## initial control
u_traj = np.zeros([T - 1, m])
## process noise for the nominal trajectory
W_traj = np.zeros([T - 1, nw])
rng = np.random.default_rng(123456789)
W_traj[:, 0] = rng.uniform(low=-1.0, high=1.0, size=(T - 1))
rng = np.random.default_rng(987654321)
W_traj[:, 1] = rng.uniform(low=-1.0, high=1.0, size=(T - 1))

## noise for samples
W_traj_s = rng.uniform(low=-1.0, high=1.0, size=(N, T - 1, nw))