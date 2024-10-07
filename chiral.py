import numpy as np
from tqdm import tqdm  # Progress bar
from numpy.linalg import inv, det
from scipy.linalg import solve
import pandas as pd
from scipy.special import iv  # Modified Bessel function of the first kind

from helper_fn import (
    ccg,
    ind2,
    J_tilde,
    Zeta_mf_ordered,
    process_expression_data,
    EM_initialization,
    find_roots,
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
    pbar=True,
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
        pbar (bool): Whether to show progress bar, default is True.
        phi_start (numpy.ndarray): Initial guess for phases, default is None.
        standardize (bool): Whether to standardize the matrix for inference, default is False.
        GTEx_names (bool): Convert row names if analyzing GTEx data, default is False.

    Returns:
        dict: Inferred phases, sigma, alpha, weights, iteration number, and other metrics.
    """

    E, E_full, clock_coord, N, Ng = process_expression_data(
        E, clockgenes, ccg, layer, standardize, mean_centre_E
    )

    phi = phi_start

    # Initializing phase using spin glass model (or any other method)
    if phi_start is None:
        # Use the spin glass initialization (for now, we skip implementation of J.tilde and Zeta.mf.ordered)
        beta = 1000
        J = J_tilde(E)  # Placeholder function, needs definition
        ##################
        # There a re changes here, in both next lines, put to avoid
        ##################
        Zeta = Zeta_mf_ordered(J, beta, E.shape[0])
        phi = Zeta[:, 1]  # + np.random.uniform(-0.5, 0.5, size=E.shape[0])

    sigma2, u, tau2, T, S, W = EM_initialization(E, sigma2, u, tau2, iterations)
    dTinv = 1 / det(T)

    # Set up progress bar if requested
    if pbar:
        progress_bar = tqdm(total=iterations)

    # Start EM iterations
    for i in range(iterations):
        phi_old = phi.copy()

        # Trigonometric components based on the current phi values
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Design matrix X with columns: [1, cos(phi), sin(phi)]
        X = np.column_stack((np.ones(N), cos_phi, sin_phi))
        X_old = X.copy()

        # Matrix multiplication and inversion for parameter updates
        Nn = X.T @ X  # This is t(X) %*% X in R
        Nn_inv = inv(Nn)  # Inverse of Nn

        T_inv = inv(T)  # Inverse of the T matrix
        M = Nn + sigma2 * T_inv  # M = Nn + sigma2 * Tinv
        M_inv = inv(M)  # Inverse of M

        # Update the alpha parameters
        alpha = M_inv @ X.T @ E
        alpha = alpha.T  # Transpose back to match R's convention

        # If two-state model is off, reset weights to initial values
        if not TSM:
            W = np.ones(Ng)

        Tot = np.sum(W)  # Total sum of weights

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
            lambda x: np.sum(
                W * (alpha[:, 1] * (x - alpha[:, 0]) / sigma2 - M_inv[0, 1])
            )
            / Tot,
            1,
            E,
        )
        be = np.apply_along_axis(
            lambda x: np.sum(
                W * (alpha[:, 2] * (x - alpha[:, 0]) / sigma2 - M_inv[0, 2])
            )
            / Tot,
            1,
            E,
        )

        # Create matrix O
        O = np.vstack([al, be])

        # Ensure no NaN values in O
        if np.any(np.isnan(O)):
            return {"alpha": alpha, "weights": W, "iteration": i}

        # Apply find_roots function for each column of O
        rooted = np.apply_along_axis(find_roots, 0, O, A, B, C, D)

        # Test roots to find the minimum (corresponds to the minimum of the Q function)
        ze = []
        for j in range(N):
            possible_solutions = []
            for y in rooted[:, j]:
                if np.abs(np.imag(y)) > 1e-8:
                    continue  # Skip complex roots
                y = np.real(y)

                K_lambda = K + np.eye(2) * y
                zet = solve(K_lambda, O[:, j])  # Solve the system of equations

                phit = np.arctan2(zet[1], zet[0]) % (2 * np.pi)
                Xt = np.array([1, np.cos(phit), np.sin(phit)])
                Mt = np.outer(Xt, Xt)

                Xt_old = np.array([1, np.cos(phi_old[j]), np.sin(phi_old[j])])
                Mt_old = np.outer(Xt_old, Xt_old)

                # Compute the Q function for each root
                Qs = np.array(
                    [
                        alpha[k] @ Mt @ alpha[k]
                        ############################
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

                Q = np.sum(Qs * W / sigma2) / Tot
                Q_old = np.sum(Qs_old * W / sigma2) / Tot

                possible_solutions.append(np.hstack((zet, Q, Q_old)))

            # Find the solution that minimizes the Q function
            solutions = np.vstack(possible_solutions)
            min_index = np.argmin(solutions[:, 2])
            if solutions[min_index, 2] == 1e5:
                raise ValueError(f"No solution found on the circle at iteration {i}")

            ze.append(solutions[min_index, :])

        # Update phi using the best root solutions
        ze = np.array(ze).T
        phi = np.arctan2(ze[1, :], ze[0, :]) % (2 * np.pi)

        # Update weights
        Qhist = pd.DataFrame()  # Empty DataFrame to store Qhist

        # `ze` matrix construction
        # ze = np.array(ze).reshape(8, N)  # Equivalent of matrix(unlist(ze), 8, N)
        # Set rownames equivalent in Python (using Pandas DataFrame)
        # ze_df = pd.DataFrame(
        #     ze.T, columns=["cos", "sin", "Q", "Q_old", "Q1", "Q2", "Q3", "Q4"]
        # )

        # create empty dataset
        ze_df = pd.DataFrame(
            columns=["cos", "sin", "Q", "Q_old", "Q1", "Q2", "Q3", "Q4"]
        )

        # Store Qhist on the first iteration
        if i == 0:
            Qhist = ze_df.copy()
            Qhist["iteration"] = i + 1  # Iteration column
            Qhist["sample"] = np.arange(1, N + 1)  # Sample numbers starting from 1
        else:
            Qtemp = ze_df.copy()
            Qtemp["iteration"] = i + 1
            Qtemp["sample"] = np.arange(1, N + 1)
            Qhist = pd.concat([Qhist, Qtemp])  # Append the new iteration results

        # Update weights W
        P1_0 = np.array(
            [
                np.exp(E[:, p] @ X @ M_inv @ X.T @ E[:, p].T / (2 * sigma2))
                * np.sqrt(dTinv * sigma2**3 / np.linalg.det(M))
                for p in range(Ng)
            ]
        )
        W = q * P1_0 / (1 - q + q * P1_0)
        W[np.isnan(W)] = 1  # Replace NaN with 1

        # Exit the loop if convergence is reached
        if np.max(np.abs(phi - phi_old)) < 0.001:
            if pbar:
                print("\nAlgorithm has converged\n")
            return {
                "phi": phi,
                "sigma": sigma2,
                "alpha": alpha,
                "weights": W,
                "iteration": i + 1,
                "sigma.m1": sigma2_m1,
                "E": E_full,
                "Qhist": Qhist,
                "geni": clockgenes,
            }

        # Update parameters for the next EM step
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        X = np.column_stack(
            (np.ones(len(phi)), cos_phi, sin_phi)
        )  # Equivalent of cbind(1, cos(phi), sin(phi))

        Mold = Nn + sigma2 * T_inv
        Moldinv = np.linalg.inv(Mold)  # Inverse of Mold

        # Update sigma2.m1
        sigma2_m1 = np.array(
            [
                np.sum(np.diag(S[s] - S[s] @ X_old @ Moldinv @ X.T)) / N + 0.01
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

        if pbar:
            print(f"Iteration {i + 1} completed")

    if pbar:
        print("\nEM algorithm finished after maximum iterations\n")

    return {
        "phi": phi,
        "Qhist": Qhist,
        "sigma": sigma2,
        "alpha": alpha,
        "weights": W,
        "iteration": iterations,
        "sigma.m1": sigma2_m1,
        "E": E_full,
    }


# ADD THIS PART
