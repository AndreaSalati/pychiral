import numpy as np
from tqdm import trange  # Progress bar
from numpy.linalg import inv, det
import pandas as pd

from .helper_fn import (
    ccg,
    process_expression_data,
)

from .stat_phys import (
    J_tilde,
    Zeta_mf_ordered,
)

from .EM import (
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
    layer=None,
    iterations=500,
    clockgenes=None,
    tau2=None,
    u=None,
    sigma2=None,
    TSM=True,
    mean_centre_E=True,
    q=0.1,
    update_q=False,
    phi_start=None,
    standardize=False,
    GTEx_names=False,
):
    """
    Infer the circular phase of gene expression §.

    Parameters:
        E (numpy.ndarray): Matrix of gene expression. Samples should be on columns, genes on rows.
        iterations (int): Number of maximum iterations. Default is 500.
        clockgenes (list): Set of clock genes (subset of rownames), default is None.
        tau2 (float): Tau parameter for the prior on gene coefficient, default is None.
        u (float): u parameter for the prior on gene means, default is None.
        sigma2 (float): Standard deviation of data points for prediction, default is None.
        TSM (bool): Switches two-state model in EM, default is True.
        mean_centre_E (bool): Whether to center data around the empirical mean, default is True.
        q (float): Probability weight for EM procedure.
        update_q (bool): Whether to update q during EM, default is False.
        phi_start (numpy.ndarray): Initial guess for phases, default is None.
        standardize (bool): Whether to standardize the matrix for inference, default is False.
        GTEx_names (bool): Convert row names if analyzing GTEx data, default is False.

    Returns:
        dict: Inferred phases, sigma, alpha, weights, iteration number, and other metrics.
    """

    E, E_full, clock_coord, Nc, Ng = process_expression_data(
        E, clockgenes, ccg, layer, standardize, mean_centre_E
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

        K, O, A, B, C, D = solve_lagrange(alpha, M_inv, E, sigma2, W, Tot, i)
        # Ensure no NaN values in O
        if np.any(np.isnan(O)):
            return {"alpha": alpha, "weights": W, "iteration": i}

        # Apply find_roots function for each column of O
        rooted = np.apply_along_axis(find_roots, 0, O, A, B, C, D)

        # Test roots to find the minimum (corresponds to the minimum of the Q function)
        ze = []
        # Loop through each sample to find the best root
        for j in range(Nc):
            possible_solutions = evaluate_possible_solutions(
                rooted, K, O, alpha, E, W, sigma2, M_inv, Tot, phi_old, j, Ng
            )

            # Find the solution that minimizes the Q function
            solutions = np.vstack(possible_solutions)
            # choosing min of Q function
            min_index = np.argmin(solutions[:, 2])
            if solutions[min_index, 2] == 1e5:
                raise ValueError(f"No solution found on the circle at iteration {i}")

            ze.append(solutions[min_index, :])

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
