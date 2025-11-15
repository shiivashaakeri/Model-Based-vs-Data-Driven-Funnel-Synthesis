import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

# Problem data (given)
n = 4  ## number of states
m = 2  ## number of inputs

# These are the result of discretizing 2D double integrator dynamics with zero-order hold and dt = 0.1
A = np.array([[1, 0, 0.1, 0],
              [0, 1, 0, 0.1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

B = np.array([[0.005, 0],
              [0, 0.005],
              [0.1, 0],
              [0, 0.1]])

# LQR cost matrices
Q = 3 * np.eye(4)
R = np.eye(2)
Q_N = 10 * np.eye(4)

# Time horizon
N = 100

# Initial state
x_init = np.array([1, 2, -0.25, 0.5])

# LQR implementation goes in this cell

# P_matrices = []
# K_matrices = []

P_matrices = np.zeros((N, n, n))
K_matrices = np.zeros((N - 1, m, n))

x_trajectory_lqr = np.zeros((N, 4))
u_history_lqr = np.zeros((N - 1, 2))

###### FILL CODE HERE ######
P_matrices[N - 1, :, :] = Q_N  ## terminal cost

for i in range(N - 1):
    # starting from N-1
    k = N - 2 - i  ## i = 0, k = 8 / i = N-2, k = 0
    P_kp1 = P_matrices[k + 1, :, :]
    P_k = Q + A.T @ P_kp1 @ A - A.T @ P_kp1 @ B @ inv(R + B.T @ P_kp1 @ B) @ B.T @ P_kp1 @ A
    K_k = -inv(R + B.T @ P_kp1 @ B) @ B.T @ P_kp1 @ A
    P_matrices[k, :, :] = P_k
    K_matrices[k, :, :] = K_k

## simulate the trajectory
x_trajectory_lqr[0, :] = x_init  ## set the initial condition
for i in range(N - 1):
    u_k = K_matrices[i, :] @ x_trajectory_lqr[i, :]
    u_history_lqr[i, :] = u_k
    x_trajectory_lqr[i + 1, :] = A @ x_trajectory_lqr[i, :] + B @ u_k
# as a result of your code, the P_matrices list defined above should contain the ten P matrices in increasing order of time,  # noqa: E501
# and the K_matrices list defined above should contain the nine K matrices in increasing order of time.
#
# x_trajectory_lqr should contain the complete state trajectory (with the k-th row of x_trajectory_lqr containing
# the state at time k), and likewise u_history_lqr should contain the complete control history (with the k-th row
# of u_history_lqr containing the control at time k).


##########################################

# Provided plotting code
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(x_trajectory_lqr[:, 0], x_trajectory_lqr[:, 1])
plt.title("State trajectory lqr")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(N - 1), u_history_lqr[:, 0], label="u1")
plt.plot(range(N - 1), u_history_lqr[:, 1], label="u2")
plt.legend()
plt.title("Control vs. k lqr")
plt.xlabel("k")
plt.ylabel("control")
plt.axis("equal")
plt.grid()


# CVX implementation goes in this cell

x_trajectory_cvx = np.zeros((N, 4))
u_history_cvx = np.zeros((N - 1, 2))

###### FILL CODE HERE ######
x_all = cp.Variable((N, n))
u_all = cp.Variable((N - 1, m))

## terminal cost
f = 0
x_N = x_all[N - 1, :]
f_terminal = cp.quad_form(x_N, Q_N)
f += f_terminal
## initial condition
constraints = [x_all[0,:] == x_init]
for i in range(N - 1):
    x_t = x_all[i, :]
    x_tp1 = x_all[i + 1, :]
    u_t = u_all[i, :]
    ## running cost
    f += cp.quad_form(x_t, Q) + cp.quad_form(u_t, R)
    constraints.append(x_tp1 == A @ x_t + B @ u_t)

problem = cp.Problem(cp.Minimize(f), constraints)

# Solve the optimization problem
problem.solve(solver=cp.CLARABEL)
x_trajectory_cvx = x_all.value
u_history_cvx = u_all.value
######################################################

# Provided plotting code
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(x_trajectory_cvx[:, 0], x_trajectory_cvx[:, 1])
plt.title("State trajectory cvx")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(N - 1), u_history_cvx[:, 0], label="u1")
plt.plot(range(N - 1), u_history_cvx[:, 1], label="u2")
plt.legend()
plt.title("Control vs. k cvx")
plt.xlabel("k")
plt.ylabel("control")
plt.axis("equal")
plt.grid()
plt.show()


print("States match:", np.allclose(x_trajectory_cvx, x_trajectory_lqr))
print("Controls match:", np.allclose(u_history_cvx, u_history_lqr))
