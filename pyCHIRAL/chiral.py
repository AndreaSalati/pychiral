import numpy as np
from tqdm import trange  # Progress bar
from numpy.linalg import det
import pandas as pd

from .helper_fn import (
    ccg,
    process_expression_data,
)

from .stat_phys import (
    J_tilde,
    Zeta_mf_ordered,
)

from .em import (
    EM_initialization,
    update_matrices,
    solve_lagrange,
    find_roots,
    evaluate_possible_solutions,
    update_Q_hist,
    update_EM_parameters,
    update_weights,
)


def CHIRAL(
    E,
    clockgenes,
    layer=None,
    iterations=500,
    tau2=None,
    u=None,
    sigma2=None,
    TSM=True,
    standardize=True,
    q=0.1,
    update_q=False,
    phi_start=None,
):
    """
    Infer the circular phase of gene expression

    Parameters:
        - E (numpy.ndarray): Matrix of gene expression. Samples should be on columns, genes on rows
        - clockgenes (list): Set of clock genes (subset of .var_names), default is None, which uses core clock genes.
        - layer (string): Layer of the data to use, default is None, which useses adata.X
        - iterations (int): Number of maximum iterations. Default is 500.
        - tau2 (float): Tau parameter for the prior on gene coefficient, default is None.
        - u (float): u parameter for the prior on gene means, default is None.
        - sigma2 (float): Standard deviation of data points for prediction, default is None.
        - TSM (bool): Switches two-state model in EM, default is True.
        - mean_centre_E (bool): Whether to center data around the empirical mean, default is True.
        - q (float): Probability weight for EM procedure.
        - update_q (bool): Whether to update q during EM, default is False.
        - phi_start (numpy.ndarray): Initial guess for phases, default is None.
        - standardize (bool): Whether to standardize the matrix for inference, default is False.

    Returns:
        dict: Inferred phases, sigma, alpha, weights, iteration number, and other metrics.
    """

    E, E_full, clock_coord, Nc, Ng = process_expression_data(
        E, clockgenes, ccg, layer, standardize
    )

    phi = phi_start

    # Initializing phase using spin glass model (or any other method)
    if phi_start is None:
        # Use the spin glass initialization (for now, we skip implementation of J.tilde and Zeta.mf.ordered)
        beta = 1000
        J = J_tilde(E)
        Zeta = Zeta_mf_ordered(J, beta, E.shape[0])
        phi = Zeta[:, 1] + np.random.uniform(-0.5, 0.5, size=E.shape[0])

    sigma2, u, tau2, T, S, W = EM_initialization(E, sigma2, u, tau2, iterations)
    dTinv = 1 / det(T)

    # Start EM iterations
    for i in trange(iterations, desc="Progress", bar_format="{percentage:3.0f}%"):
        phi_old, alpha, M, M_inv, Nn, Nn_inv, X, X_old = update_matrices(
            phi, T, E, sigma2, Nc
        )

        # If two-state model is off, reset weights to initial values
        if not TSM:
            W = np.ones(Ng)

        Tot = np.sum(W)  # Total sum of weights

        K, Om, A, B, C, D = solve_lagrange(alpha, M_inv, E, sigma2, W, Tot, i)
        # Ensure no NaN values in Om
        if np.any(np.isnan(Om)):
            return {"alpha": alpha, "weights": W, "iteration": i}

        # Apply find_roots function for each column of Om
        rooted = np.apply_along_axis(find_roots, 0, Om, A, B, C, D)

        # Test roots to find the minimum (corresponds to the minimum of the Q function)
        # Vectorized version of the function
        possible_solutions = evaluate_possible_solutions(
            rooted, K, Om, alpha, E, W, sigma2, M_inv, Tot, phi_old, Ng
        )
        # Extract Q values (the third column)
        Q_values = possible_solutions[:, :, 2]  # Shape: [Nroots, Nc]

        # Find the indices of the minimum Q value for each observation
        min_indices = np.argmin(Q_values, axis=0)  # Shape: [Nc]

        # Extract the minimum Q values
        min_Q_values = Q_values[
            min_indices, np.arange(Q_values.shape[1])
        ]  # Shape: [Nc]

        # Check for invalid solutions
        if np.any(min_Q_values == 1e5):
            raise ValueError("No solution found on the circle for some observations")

        # Extract the corresponding solutions # Shape: [Nc, 4]
        ze = possible_solutions[min_indices, np.arange(Q_values.shape[1]), :]
        # Update phi using the best root solutions
        ze = np.array(ze).T
        phi = np.arctan2(ze[1, :], ze[0, :]) % (2 * np.pi)

        # Update weights
        if i == 0:
            Q_hist = pd.DataFrame(
                columns=["cos", "sin", "Q", "Q_old", "iteration", "sample"]
            )

        Q_hist = update_Q_hist(Q_hist, ze, i, Nc)
        W = update_weights(E, X, M_inv, sigma2, dTinv, M, q)

        # Exit the loop if convergence is reached
        if np.max(np.abs(phi - phi_old)) < 0.001:
            print("Algorithm has converged")
            return {
                "phi": phi,
                "sigma": sigma2,
                "alpha": alpha,
                "weights": W,
                "iteration": i + 1,
                "E": E_full,
                "Q_hist": Q_hist,
                "genes": clockgenes,
            }

        cos_phi, sin_phi, X, Mold, Moldinv, sigma2_m1, sigma2_m0, sigma2, q = (
            update_EM_parameters(phi, Nn, X, X_old, S, E, T, sigma2, W, q, update_q)
        )

    print("EM algorithm finished after maximum iterations")

    return {
        "phi": phi,
        "Q_hist": Q_hist,
        "sigma": sigma2,
        "alpha": alpha,
        "weights": W,
        "iteration": iterations,
        "sigma.m1": sigma2_m1,
        "genes": clockgenes,
        "E": E_full,
    }
