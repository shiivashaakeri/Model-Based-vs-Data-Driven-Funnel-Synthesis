"""
Mathematical Utilities for DDFS.

This module provides common mathematical operations including:
- Matrix operations (norms, eigenvalues, positive definiteness)
- Ellipsoid computations (volume, containment)
- Quaternion operations (multiply, rotate, normalize, conversions)
- Rotation matrix utilities
- Numerical differentiation (Jacobian computation)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.linalg import LinAlgError

# =============================================================================
# Constants
# =============================================================================

EPS = 1e-12  # Small constant for numerical stability


# =============================================================================
# Matrix Norms and Properties
# =============================================================================


def spectral_norm(M: np.ndarray) -> float:
    """
    Compute the spectral norm (induced 2-norm) of a matrix.

    Parameters
    ----------
    M : np.ndarray
        Input matrix.

    Returns
    -------
    float
        Spectral norm (largest singular value).
    """
    return np.linalg.norm(M, ord=2)


def frobenius_norm(M: np.ndarray) -> float:
    """
    Compute the Frobenius norm of a matrix.

    Parameters
    ----------
    M : np.ndarray
        Input matrix.

    Returns
    -------
    float
        Frobenius norm.
    """
    return np.linalg.norm(M, ord="fro")


def vector_norm(v: np.ndarray, ord: int = 2) -> float:  # noqa: A002
    """
    Compute the norm of a vector.

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    ord : int, optional
        Order of the norm (default: 2 for Euclidean).

    Returns
    -------
    float
        Vector norm.
    """
    return np.linalg.norm(v, ord=ord)


def inf_norm_sequence(z: np.ndarray, axis: int = 0) -> float:
    """
    Compute infinity norm over a sequence: max_k ||z(k)||.

    Parameters
    ----------
    z : np.ndarray
        Sequence array, shape (n_steps, n_dim) or (n_dim, n_steps).
    axis : int
        Axis along which steps are indexed.

    Returns
    -------
    float
        Maximum norm over all steps.
    """
    norms = np.linalg.norm(z, axis=1) if axis == 0 else np.linalg.norm(z, axis=0)
    return np.max(norms)


# =============================================================================
# Eigenvalue Operations
# =============================================================================


def eigenvalues(M: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of a matrix.

    Parameters
    ----------
    M : np.ndarray
        Square matrix.

    Returns
    -------
    np.ndarray
        Array of eigenvalues.
    """
    return np.linalg.eigvals(M)


def eigenvalues_symmetric(M: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of a symmetric matrix (real eigenvalues).

    Parameters
    ----------
    M : np.ndarray
        Symmetric matrix.

    Returns
    -------
    np.ndarray
        Array of real eigenvalues, sorted ascending.
    """
    return np.linalg.eigvalsh(M)


def lambda_min(M: np.ndarray) -> float:
    """
    Compute minimum eigenvalue of a symmetric matrix.

    Parameters
    ----------
    M : np.ndarray
        Symmetric matrix.

    Returns
    -------
    float
        Minimum eigenvalue.
    """
    return np.min(np.linalg.eigvalsh(M))


def lambda_max(M: np.ndarray) -> float:
    """
    Compute maximum eigenvalue of a symmetric matrix.

    Parameters
    ----------
    M : np.ndarray
        Symmetric matrix.

    Returns
    -------
    float
        Maximum eigenvalue.
    """
    return np.max(np.linalg.eigvalsh(M))


def condition_number(M: np.ndarray) -> float:
    """
    Compute condition number of a matrix.

    Parameters
    ----------
    M : np.ndarray
        Input matrix.

    Returns
    -------
    float
        Condition number.
    """
    return np.linalg.cond(M)


# =============================================================================
# Positive Definiteness
# =============================================================================


def is_positive_definite(M: np.ndarray, tol: float = EPS) -> bool:
    """
    Check if a matrix is positive definite.

    Uses Cholesky decomposition for efficiency.

    Parameters
    ----------
    M : np.ndarray
        Square symmetric matrix.
    tol : float, optional
        Tolerance for eigenvalue check.

    Returns
    -------
    bool
        True if positive definite.
    """
    if M.shape[0] != M.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(M, M.T, atol=tol):
        return False

    try:
        np.linalg.cholesky(M)
        return True
    except LinAlgError:
        return False


def is_positive_semidefinite(M: np.ndarray, tol: float = EPS) -> bool:
    """
    Check if a matrix is positive semidefinite.

    Parameters
    ----------
    M : np.ndarray
        Square symmetric matrix.
    tol : float, optional
        Tolerance for eigenvalue check.

    Returns
    -------
    bool
        True if positive semidefinite.
    """
    if M.shape[0] != M.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(M, M.T, atol=tol):
        return False

    # Check eigenvalues
    eigvals = np.linalg.eigvalsh(M)
    return np.all(eigvals >= -tol)


def enforce_positive_definite(M: np.ndarray, min_eigenvalue: float = EPS) -> np.ndarray:
    """
    Enforce positive definiteness by adjusting eigenvalues.

    Projects matrix to nearest positive definite matrix.

    Parameters
    ----------
    M : np.ndarray
        Square symmetric matrix.
    min_eigenvalue : float, optional
        Minimum eigenvalue to enforce.

    Returns
    -------
    np.ndarray
        Positive definite matrix.
    """
    # Symmetrize
    M_sym = 0.5 * (M + M.T)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M_sym)

    # Clip eigenvalues
    eigvals_clipped = np.maximum(eigvals, min_eigenvalue)

    # Reconstruct
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def enforce_positive_semidefinite(M: np.ndarray, tol: float = EPS) -> np.ndarray:  # noqa: ARG001
    """
    Enforce positive semidefiniteness by clipping negative eigenvalues.

    Parameters
    ----------
    M : np.ndarray
        Square symmetric matrix.
    tol : float, optional
        Tolerance (eigenvalues below -tol are clipped to 0).

    Returns
    -------
    np.ndarray
        Positive semidefinite matrix.
    """
    # Symmetrize
    M_sym = 0.5 * (M + M.T)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M_sym)

    # Clip negative eigenvalues to zero
    eigvals_clipped = np.maximum(eigvals, 0.0)

    # Reconstruct
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


# =============================================================================
# Matrix Square Root and Decompositions
# =============================================================================


def matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """
    Compute matrix square root of a positive semidefinite matrix.

    Returns S such that S @ S.T = M.

    Parameters
    ----------
    M : np.ndarray
        Positive semidefinite matrix.

    Returns
    -------
    np.ndarray
        Matrix square root.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0.0)  # Numerical safety
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def matrix_sqrt_inv(M: np.ndarray, regularization: float = EPS) -> np.ndarray:
    """
    Compute inverse matrix square root of a positive definite matrix.

    Returns S such that S @ M @ S = I.

    Parameters
    ----------
    M : np.ndarray
        Positive definite matrix.
    regularization : float, optional
        Small value added to eigenvalues for numerical stability.

    Returns
    -------
    np.ndarray
        Inverse matrix square root.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, regularization)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def cholesky_safe(M: np.ndarray, regularization: float = EPS) -> np.ndarray:
    """
    Compute Cholesky decomposition with regularization for numerical stability.

    Parameters
    ----------
    M : np.ndarray
        Positive definite matrix.
    regularization : float, optional
        Small value added to diagonal if decomposition fails.

    Returns
    -------
    np.ndarray
        Lower triangular Cholesky factor L such that M = L @ L.T.
    """
    try:
        return np.linalg.cholesky(M)
    except LinAlgError:
        # Add regularization and retry
        M_reg = M + regularization * np.eye(M.shape[0])
        return np.linalg.cholesky(M_reg)


# =============================================================================
# Ellipsoid Operations
# =============================================================================


def ellipsoid_volume(P: np.ndarray) -> float:
    """
    Compute volume of ellipsoid E(P) = {x : x^T P x <= 1}.

    Parameters
    ----------
    P : np.ndarray
        Positive definite matrix defining the ellipsoid.

    Returns
    -------
    float
        Volume of the ellipsoid.
    """
    n = P.shape[0]
    # Volume = (pi^(n/2) / Gamma(n/2 + 1)) * det(P)^(-1/2)
    # Using log for numerical stability
    log_det_P = np.linalg.slogdet(P)[1]

    # Volume of unit ball in n dimensions
    if n % 2 == 0:
        # Even dimension
        k = n // 2
        log_unit_ball = (n / 2) * np.log(np.pi) - np.sum(np.log(np.arange(1, k + 1)))
    else:
        # Odd dimension
        k = (n - 1) // 2
        log_unit_ball = (
            ((n + 1) / 2) * np.log(np.pi)
            + np.sum(np.log(np.arange(1, k + 1)))
            - np.sum(np.log(np.arange(1, n + 1, 2)))
            - np.log(2) * k
        )

    log_volume = log_unit_ball - 0.5 * log_det_P
    return np.exp(log_volume)


def ellipsoid_contains(P: np.ndarray, x: np.ndarray) -> bool:
    """
    Check if point x is inside ellipsoid E(P) = {x : x^T P x <= 1}.

    Parameters
    ----------
    P : np.ndarray
        Positive definite matrix defining the ellipsoid.
    x : np.ndarray
        Point to check.

    Returns
    -------
    bool
        True if x is inside or on the boundary.
    """
    return x @ P @ x <= 1.0 + EPS


def ellipsoid_boundary_points(P: np.ndarray, n_points: int = 100, dims: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Generate points on the boundary of a 2D projection of an ellipsoid.

    Parameters
    ----------
    P : np.ndarray
        Positive definite matrix defining the ellipsoid.
    n_points : int, optional
        Number of boundary points.
    dims : tuple of int, optional
        Dimensions to project onto.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) with boundary points.
    """
    # Extract 2x2 submatrix for projection
    idx = np.array(dims)
    P_2d = P[np.ix_(idx, idx)]

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(P_2d)

    # Semi-axes lengths
    axes = 1.0 / np.sqrt(eigvals)

    # Generate points on unit circle and transform
    theta = np.linspace(0, 2 * np.pi, n_points)
    unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])

    # Transform: scale by axes, rotate by eigenvectors
    boundary = unit_circle @ np.diag(axes) @ eigvecs.T

    return boundary


# =============================================================================
# Quaternion Operations
# =============================================================================


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz] (scalar-first convention).

    Returns
    -------
    np.ndarray
        Normalized quaternion.
    """
    norm = np.linalg.norm(q)
    if norm < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz].

    Returns
    -------
    np.ndarray
        Conjugate quaternion [qw, -qx, -qy, -qz].
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion inverse.

    For unit quaternions, inverse equals conjugate.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz].

    Returns
    -------
    np.ndarray
        Inverse quaternion.
    """
    norm_sq = np.dot(q, q)
    if norm_sq < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat_conjugate(q) / norm_sq


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.

    Parameters
    ----------
    p : np.ndarray
        First quaternion [pw, px, py, pz].
    q : np.ndarray
        Second quaternion [qw, qx, qy, qz].

    Returns
    -------
    np.ndarray
        Product quaternion p * q.
    """
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a 3D vector by a quaternion.

    Rotates vector from body frame to inertial frame.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz].
    v : np.ndarray
        3D vector to rotate.

    Returns
    -------
    np.ndarray
        Rotated 3D vector.
    """
    # Convert vector to quaternion form [0, vx, vy, vz]
    v_quat = np.array([0.0, v[0], v[1], v[2]])

    # Rotate: q * v * q^*
    v_rot = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))

    return v_rot[1:4]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a 3D vector by the inverse of a quaternion.

    Rotates vector from inertial frame to body frame.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz].
    v : np.ndarray
        3D vector to rotate.

    Returns
    -------
    np.ndarray
        Rotated 3D vector.
    """
    return quat_rotate(quat_conjugate(q), v)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create quaternion from axis-angle representation.

    Parameters
    ----------
    axis : np.ndarray
        Unit axis of rotation (3D).
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Quaternion [qw, qx, qy, qz].
    """
    axis = axis / (np.linalg.norm(axis) + EPS)
    half_angle = angle / 2.0
    return np.array(
        [
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
        ]
    )


def quat_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert quaternion to axis-angle representation.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz].

    Returns
    -------
    axis : np.ndarray
        Unit axis of rotation.
    angle : float
        Rotation angle in radians.
    """
    q = quat_normalize(q)

    # Handle near-zero rotation
    sin_half = np.linalg.norm(q[1:4])
    if sin_half < EPS:
        return np.array([1.0, 0.0, 0.0]), 0.0

    angle = 2.0 * np.arctan2(sin_half, q[0])
    axis = q[1:4] / sin_half

    return axis, angle


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create quaternion from Euler angles (ZYX convention).

    Parameters
    ----------
    roll : float
        Roll angle (rotation about x-axis) in radians.
    pitch : float
        Pitch angle (rotation about y-axis) in radians.
    yaw : float
        Yaw angle (rotation about z-axis) in radians.

    Returns
    -------
    np.ndarray
        Quaternion [qw, qx, qy, qz].
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (ZYX convention).

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz].

    Returns
    -------
    roll : float
        Roll angle in radians.
    pitch : float
        Pitch angle in radians.
    yaw : float
        Yaw angle in radians.
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# =============================================================================
# Rotation Matrix Operations
# =============================================================================


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [qw, qx, qy, qz].

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    q = quat_normalize(q)
    qw, qx, qy, qz = q

    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.

    Uses Shepperd's method for numerical stability.

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix.

    Returns
    -------
    np.ndarray
        Quaternion [qw, qx, qy, qz].
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return quat_normalize(np.array([qw, qx, qy, qz]))


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (ZYX convention).

    Parameters
    ----------
    roll : float
        Roll angle (rotation about x-axis) in radians.
    pitch : float
        Pitch angle (rotation about y-axis) in radians.
    yaw : float
        Yaw angle (rotation about z-axis) in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles from rotation matrix (ZYX convention).

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix.

    Returns
    -------
    roll : float
        Roll angle in radians.
    pitch : float
        Pitch angle in radians.
    yaw : float
        Yaw angle in radians.
    """
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    if np.abs(sy) < 1.0 - EPS:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return roll, pitch, yaw


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.

    Parameters
    ----------
    v : np.ndarray
        3D vector.

    Returns
    -------
    np.ndarray
        3x3 skew-symmetric matrix.
    """
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def vee(S: np.ndarray) -> np.ndarray:
    """
    Extract 3D vector from skew-symmetric matrix.

    Parameters
    ----------
    S : np.ndarray
        3x3 skew-symmetric matrix.

    Returns
    -------
    np.ndarray
        3D vector.
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# =============================================================================
# Numerical Differentiation
# =============================================================================


def numerical_jacobian(
    f: callable,
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute Jacobian of f at x using central finite differences.

    Parameters
    ----------
    f : callable
        Function f: R^n -> R^m.
    x : np.ndarray
        Point at which to compute Jacobian.
    eps : float, optional
        Finite difference step size.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (m, n).
    """
    n = len(x)
    f0 = f(x)
    m = len(f0)

    J = np.zeros((m, n))

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        J[:, i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return J


def numerical_jacobian_xu(
    f: callable,
    x: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobians of f(x, u) with respect to x and u.

    Parameters
    ----------
    f : callable
        Function f: R^n x R^m -> R^n.
    x : np.ndarray
        State vector.
    u : np.ndarray
        Input vector.
    eps : float, optional
        Finite difference step size.

    Returns
    -------
    A : np.ndarray
        Jacobian df/dx of shape (n, n).
    B : np.ndarray
        Jacobian df/du of shape (n, m).
    """
    n = len(x)
    m = len(u)

    # Jacobian with respect to x
    A = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        A[:, i] = (f(x_plus, u) - f(x_minus, u)) / (2 * eps)

    # Jacobian with respect to u
    B = np.zeros((n, m))
    for i in range(m):
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += eps
        u_minus[i] -= eps
        B[:, i] = (f(x, u_plus) - f(x, u_minus)) / (2 * eps)

    return A, B


# =============================================================================
# Block Matrix Operations
# =============================================================================


def block_diag(*matrices: np.ndarray) -> np.ndarray:
    """
    Create block diagonal matrix from input matrices.

    Parameters
    ----------
    *matrices : np.ndarray
        Variable number of matrices.

    Returns
    -------
    np.ndarray
        Block diagonal matrix.
    """
    from scipy.linalg import block_diag as scipy_block_diag  # noqa: PLC0415

    return scipy_block_diag(*matrices)


def stack_horizontal(*matrices: np.ndarray) -> np.ndarray:
    """
    Stack matrices horizontally.

    Parameters
    ----------
    *matrices : np.ndarray
        Variable number of matrices with same number of rows.

    Returns
    -------
    np.ndarray
        Horizontally stacked matrix.
    """
    return np.hstack(matrices)


def stack_vertical(*matrices: np.ndarray) -> np.ndarray:
    """
    Stack matrices vertically.

    Parameters
    ----------
    *matrices : np.ndarray
        Variable number of matrices with same number of columns.

    Returns
    -------
    np.ndarray
        Vertically stacked matrix.
    """
    return np.vstack(matrices)


# =============================================================================
# Miscellaneous
# =============================================================================


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Wrapped angle in [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """
    Wrap array of angles to [-pi, pi].

    Parameters
    ----------
    angles : np.ndarray
        Angles in radians.

    Returns
    -------
    np.ndarray
        Wrapped angles.
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """
    Safe division handling zeros in denominator.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator array.
    denominator : np.ndarray
        Denominator array.
    default : float, optional
        Value to use when denominator is zero.

    Returns
    -------
    np.ndarray
        Result of division.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = default
    return result
