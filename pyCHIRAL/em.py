import numpy as np
from tqdm import tqdm  # Progress bar
from numpy.linalg import inv, det
from scipy.linalg import solve
import pandas as pd
from scipy.special import iv  # Modified Bessel function of the first kind

# This file contains the functions for the EM algorithm
# in the loop over iterations, the functions are called in
# same order as they are defined in this file


def EM_Step():
    pass


def EM_initialization(E, sigma2=None, u=None, tau2=None, iterations=100):
    """
    Initializes parameters for a probabilistic model.

    Parameters:
    - E: The expression data matrix (2D NumPy array).
    - sigma2: Initial value for sigma squared. If None, calculated from E.
    - u: Initial value for u. If None, set to 0.2.
    - tau2: Initial value for tau squared. If None, calculated based on E.
    - pbar: Boolean indicating whether to show a progress bar.
    - iterations: Number of iterations for the progress bar.

    Returns:
    - sigma2: Initialized sigma squared.
    - T: Diagonal matrix of variances.
    - S: Precomputed outer products of columns of E.
    - W: Array of ones with length equal to the number of genes (Ng).
    """

    # Initialize parameters for probabilistic model
    if sigma2 is None:
        sigma2 = np.mean(np.var(E, axis=0))
    if u is None:
        u = 0.2
    if tau2 is None:
        tau2 = 4 / (24 + E.shape[0])

    sigma2_0 = sigma2
    T = np.diag([u**2, tau2, tau2])

    # Precompute some variables used in the EM loop
    Ng = E.shape[1]
    S = np.array([np.outer(E[:, l], E[:, l]) for l in range(Ng)])
    W = np.ones(Ng)

    return sigma2, u, tau2, T, S, W


def update_matrices(phi, T, E, sigma2, N):
    """
    Update matrices for the EM algorithm.
    """
    # Trigonometric components based on the current phi values
    phi_old = phi.copy()
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Design matrix X with columns: [1, cos(phi), sin(phi)]
    X = np.column_stack((np.ones(N), cos_phi, sin_phi))
    X_old = X.copy()

    # Matrix multiplication and inversion for parameter updates
    Nn = X.T @ X  # This is t(X) %*% X in R
    Nn_inv = inv(Nn)  # Inverse of Nn

    T_inv = inv(T)  # Inverse of the T matrix (covariance of gene's coefficients)
    M = Nn + sigma2 * T_inv  # M = Nn + sigma2 * Tinv
    M_inv = inv(M)  # Inverse of M

    # Update the alpha parameters
    alpha = M_inv @ X.T @ E
    alpha = alpha.T  # Transpose back to match R's convention

    return phi_old, alpha, M, M_inv, Nn, Nn_inv, X, X_old


def solve_lagrange(alpha, M_inv, E, sigma2, W, Tot, i):
    """
    First step to get the 4 zeros of teh derivative of the Q function.
    Find some matrix elements and construct the matrix K.
    """

    # Calculate intermediate variables for matrix K
    A = np.sum(W * (alpha[:, 1] ** 2 / sigma2 + M_inv[1, 1])) / Tot
    B = np.sum(W * (alpha[:, 1] * alpha[:, 2] / sigma2 + M_inv[1, 2])) / Tot
    C = np.sum(W * (alpha[:, 2] * alpha[:, 1] / sigma2 + M_inv[2, 1])) / Tot
    D = np.sum(W * (alpha[:, 2] ** 2 / sigma2 + M_inv[2, 2])) / Tot

    # Construct matrix K
    K = np.array([[A, B], [C, D]])

    # Return early if any elements of K are NaN
    if np.any(np.isnan(K)):
        return {"alpha": alpha, "weights": W, "iteration": i}

    # Calculate al and be, which depend on the current weight values and alpha
    al = np.apply_along_axis(
        lambda x: np.sum(W * (alpha[:, 1] * (x - alpha[:, 0]) / sigma2 - M_inv[0, 1]))
        / Tot,
        1,
        E,
    )
    be = np.apply_along_axis(
        lambda x: np.sum(W * (alpha[:, 2] * (x - alpha[:, 0]) / sigma2 - M_inv[0, 2]))
        / Tot,
        1,
        E,
    )

    # Create matrix O
    O = np.vstack([al, be])

    return K, O, A, B, C, D


def find_roots(x, A, B, C, D):
    """
    Computes the roots of a polynomial based on the input parameters.

    Parameters:
    - A, B, C, D: Coefficients of the polynomial.
    - x: A list or array containing the values for x[0] and x[1].

    Returns:
    - roots: Roots of the polynomial.
    """
    zero = (
        B**2 * C**2
        + A**2 * D**2
        - x[0] ** 2 * (D**2 + C**2)
        - x[1] ** 2 * (A**2 + B**2)
        + 2 * x[0] * x[1] * (A * B + C * D)
        - 2 * A * B * C * D
    )
    one = 2 * (
        (A + D) * (A * D - B * C)
        - x[0] ** 2 * D
        - x[1] ** 2 * A
        + x[0] * x[1] * (B + C)
    )
    two = A**2 + D**2 + 4 * A * D - x[0] ** 2 - x[1] ** 2 - 2 * B * C
    three = 2 * (A + D)
    four = 1

    # Roots of the polynomial (use numpy roots for equivalent of polyroot)
    return np.roots([four, three, two, one, zero])


def evaluate_possible_solutions(
    rooted, K, O, alpha, E, W, sigma2, M_inv, Tot, phi_old, j, Ng
):
    """
    This function evaluates the possible solutions for the roots of the polynomial.
    And eval Q(sol)
    """
    possible_solutions = []

    for y in rooted[:, j]:
        # Ignore complex roots
        if np.abs(np.imag(y)) > 1e-8:
            possible_solutions.append(np.array([0, 0, 100000, 100000]))
            continue

        # Take the real part of the root
        y = np.real(y)

        # Compute K - lambda*I
        K_lambda = K + np.eye(2) * y

        # Solve the system of equations: K_lambda * zet = O[:, j]
        zet = solve(K_lambda, O[:, j])

        # Compute new phi from zet
        phit = np.arctan2(zet[1], zet[0]) % (2 * np.pi)

        # Compute new Xt and Mt
        Xt = np.array([1, np.cos(phit), np.sin(phit)])
        Mt = np.outer(Xt, Xt)

        # Compute old Xt and Mt_old based on phi_old
        Xt_old = np.array([1, np.cos(phi_old[j]), np.sin(phi_old[j])])
        Mt_old = np.outer(Xt_old, Xt_old)

        # Compute Qs for the new and old values
        Qs = np.array(
            [
                alpha[k] @ Mt @ alpha[k]
                - 2 * alpha[k] @ Xt * E[j, k]
                + sigma2 * np.sum(np.diag(M_inv @ Mt))
                for k in range(Ng)
            ]
        )
        Qs_old = np.array(
            [
                alpha[k] @ Mt_old @ alpha[k]
                - 2 * alpha[k] @ Xt_old * E[j, k]
                + sigma2 * np.sum(np.diag(M_inv @ Mt_old))
                for k in range(Ng)
            ]
        )

        # Compute the Q function for new and old values
        Q = np.sum(Qs * W / sigma2) / Tot
        Q_old = np.sum(Qs_old * W / sigma2) / Tot

        # Append the solution
        possible_solutions.append(np.hstack((zet, Q, Q_old)))

    return possible_solutions


def evaluate_possible_solutions_vectorized(
    rooted, K, O, alpha, E, W, sigma2, M_inv, Tot, phi_old, j, Ng
):
    """
    Vectorized version of the function to evaluate possible solutions for the roots of the polynomial.
    """
    y = rooted[:, j]

    # Create a mask for roots with negligible imaginary parts
    mask = np.abs(np.imag(y)) <= 1e-8

    # Initialize possible_solutions with default values
    Nroots = y.shape[0]
    possible_solutions = np.zeros((Nroots, 4))
    possible_solutions[:, :2] = 0
    possible_solutions[:, 2:] = 100000

    # Process roots with negligible imaginary parts
    real_y = y[mask].real  # Shape: (Nroots_real,)

    # Compute K_lambda = K + y * I
    K_lambda = (
        K + np.eye(2)[None, :, :] * real_y[:, None, None]
    )  # Shape: (Nroots_real, 2, 2)

    # Solve K_lambda * zet = O[:, j]
    O_j = O[:, j]  # Shape: (2,)
    O_j_tiled = np.tile(O_j, (real_y.shape[0], 1))  # Shape: (Nroots_real, 2)
    zet = np.linalg.solve(K_lambda, O_j_tiled)  # Shape: (Nroots_real, 2)

    # Compute phit from zet
    phit = np.arctan2(zet[:, 1], zet[:, 0]) % (2 * np.pi)  # Shape: (Nroots_real,)

    # Compute Xt and Mt
    Xt = np.column_stack(
        (np.ones(real_y.shape[0]), np.cos(phit), np.sin(phit))
    )  # Shape: (Nroots_real, 3)
    Mt = np.einsum("ni,nj->nij", Xt, Xt)  # Shape: (Nroots_real, 3, 3)

    # Compute Xt_old and Mt_old
    phi_old_j = phi_old[j]
    Xt_old = np.array([1, np.cos(phi_old_j), np.sin(phi_old_j)])  # Shape: (3,)
    Mt_old = np.outer(Xt_old, Xt_old)  # Shape: (3, 3)

    # Prepare alpha and E for vectorized computations
    alpha = np.array(alpha)  # Shape: (Ng, 3)
    E_j = E[j, :]  # Shape: (Ng,)

    # Compute Qs for new values
    Qs_term1 = np.einsum("ki,nij,kj->nk", alpha, Mt, alpha)  # Shape: (Nroots_real, Ng)
    Term2 = -2 * np.dot(Xt, alpha.T) * E_j[None, :]  # Shape: (Nroots_real, Ng)
    Term3 = sigma2 * np.einsum("ij,nji->n", M_inv, Mt)  # Shape: (Nroots_real,)
    Qs = Qs_term1 + Term2 + Term3[:, None]  # Shape: (Nroots_real, Ng)

    # Compute Qs_old
    Term1_old = np.einsum("ki,ij,kj->k", alpha, Mt_old, alpha)  # Shape: (Ng,)
    Term2_old = -2 * np.dot(alpha, Xt_old) * E_j  # Shape: (Ng,)
    Term3_old = sigma2 * np.trace(M_inv @ Mt_old)  # Scalar
    Qs_old = Term1_old + Term2_old + Term3_old  # Shape: (Ng,)

    # Compute the Q function for new and old values
    W_sigma2 = W / sigma2  # Shape: (Ng,)
    Q = np.sum(Qs * W_sigma2[None, :], axis=1) / Tot  # Shape: (Nroots_real,)
    Q_old = np.sum(Qs_old * W_sigma2) / Tot  # Scalar

    # Stack zet, Q, and Q_old
    solutions = np.hstack(
        (zet, Q[:, None], np.full((real_y.shape[0], 1), Q_old))
    )  # Shape: (Nroots_real, 4)

    # Update possible_solutions with computed solutions
    possible_solutions[mask, :] = solutions

    return possible_solutions


def update_Q_hist(Q_hist, ze, i, Nc):
    """
    Updates the Q_hist DataFrame with the current iteration results.
    """
    # add ze.T  to values of first 4 columns
    # Create a new DataFrame for the current block of 30 rows
    Q_temp = pd.DataFrame(ze.T, columns=["cos", "sin", "Q", "Q_old"])
    # Set 'iteration' and 'sample' columns for the new block
    Q_temp["iteration"] = i + 1  # Assign iteration number
    Q_temp["sample"] = np.arange(Nc)  # Assign sample number
    Q_hist = pd.concat([Q_hist, Q_temp])  # Append the new iteration results
    return Q_hist


def update_weights(E, X, M_inv, sigma2, dTinv, M, q):
    Nc, Ng = E.shape
    P1_0 = np.array(
        [
            np.exp(E[:, p] @ X @ M_inv @ X.T @ E[:, p].T / (2 * sigma2))
            * np.sqrt(dTinv * sigma2**3 / np.linalg.det(M))
            for p in range(Ng)
        ]
    )
    W = q * P1_0 / (1 - q + q * P1_0)
    W[np.isnan(W)] = 1  # Replace NaN with 1
    return W


def update_EM_parameters(phi, Nn, X, X_old, S, E, T, sigma2, W, q, update_q):
    """
    Updates the parameters for the next EM step.
    """
    Nc, Ng = E.shape
    # Update parameters for the next EM step
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    X = np.column_stack(
        (np.ones(len(phi)), cos_phi, sin_phi)
    )  # Equivalent of cbind(1, cos(phi), sin(phi))

    Mold = Nn + sigma2 * inv(T)
    Moldinv = inv(Mold)  # Inverse of Mold

    # Update sigma2.m1
    sigma2_m1 = np.array(
        [
            np.sum(np.diag(S[s] - S[s] @ X_old @ Moldinv @ X.T)) / Nc + 0.01
            for s in range(Ng)
        ]
    )
    sigma2_m0 = np.var(E, axis=0)
    sigma2 = np.mean(sigma2_m1 * W + sigma2_m0 * (1 - W))
    sigma2_m1 = np.sum(sigma2_m1 * W) / np.sum(W)

    # Update q if needed
    if update_q:
        q = np.mean(W)
        q = max(0.05, min(q, 0.3))

    return cos_phi, sin_phi, X, Mold, Moldinv, sigma2_m1, sigma2_m0, sigma2, q
