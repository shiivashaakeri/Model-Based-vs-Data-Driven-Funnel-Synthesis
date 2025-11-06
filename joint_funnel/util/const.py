import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA

## obstacles
num_obs = 2
obs = np.array([[4, 4], [8, 3]])
obs_r = 1

####### global variables
## unicycle
n = 3
m = 2
nw = 2
n_p = 2
n_q = 2
tf = 8
T = 61
dt = tf / T

gamma1 = 0.4
## initial and final states
x_0 = np.array([1.0, 1.0, 0.0])
x_des = np.array([10, 4, -np.pi])
x_traj = np.zeros([T, n])
x_traj[0] = x_0
## initial control
u_traj = np.zeros([T - 1, m])
## process noise
W_traj = np.zeros([T - 1, nw])
## initial funnel
Q0_traj = np.zeros([T, n, n])
K0_traj = np.zeros([T-1,m,n])
for t in range(T):
    Q0_traj[t] = np.diag([1, 1, 0.1]) * 0.1
Q0_traj[T-1] = np.diag([1, 1, 0.1]) * 0.01


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
G_u1 = np.zeros([2,2])