import numpy as np
from tqdm import tqdm  # Progress bar
from numpy.linalg import inv, det
from scipy.linalg import solve
import pandas as pd
from scipy.special import iv  # Modified Bessel function of the first kind

ccg = np.array(
    [
        "Arntl",
        "Npas2",
        "Cry1",
        "Cry2",
        "Per1",
        "Per2",
        "Nr1d1",
        "Nr1d2",
        "Tef",
        "Dbp",
        "Ciart",
        "Per3",
        "Bmal1",
    ]
)


def ind2(large_list, small_list, verbose=False):
    if isinstance(small_list, str):
        small_list = [small_list]
    large_array = np.array(large_list)
    small_array = np.array(small_list)
    indices = []

    element_to_index = {element: idx for idx, element in enumerate(large_array)}

    for element in small_array:
        if element in element_to_index:
            indices.append(element_to_index[element])
        else:
            # Optionally handle or notify when an element is not found
            if verbose:
                print(f"Warning: '{element}' not found in the large list.")

    return indices


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

    # Remove columns with any NA values
    # E = E[:, np.all(~np.isnan(E), axis=0)]

    # Standardize the matrix if needed
    if standardize:
        E = E / np.std(E, axis=0, keepdims=True)

    E_full = E.copy()

    if layer is None:
        E = E.X
    else:
        E = E.layers[layer]

    if clockgenes is None:
        clockgenes = ccg

    genes = E_full.var_names

    # Clock gene selection
    clock_coord = ind2(genes, clockgenes)
    E = E[:, clock_coord]

    N, Ng = E.shape

    # Mean center the expression data if needed
    if mean_centre_E:
        E = E - E.mean(axis=0, keepdims=True)

    # Initialize parameters for probabilistic model
    if sigma2 is None:
        sigma2 = np.mean(np.var(E, axis=0))
    if u is None:
        u = 0.2
    if tau2 is None:
        tau2 = 4 / (24 + E.shape[0])

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

    # Initialize matrices for the EM procedure

    sigma2_0 = sigma2
    T = np.diag([u**2, tau2, tau2])

    # Precompute some variables used in the EM loop
    S = np.array([np.outer(E[:, l], E[:, l]) for l in range(Ng)])
    W = np.ones(Ng)

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

        # Find numerical roots of the polynomial obtained from Lagrange multipliers
        def find_roots(x):
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
            return np.roots([zero, one, two, three, four])

        # Apply find_roots function for each column of O
        rooted = np.apply_along_axis(find_roots, 0, O)

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


# ADD THIS PART

# Qhist = pd.DataFrame()  # Empty DataFrame to store Qhist

#     for i in range(iterations):
#         # `ze` matrix construction
#         ze = np.array(ze).reshape(8, N)  # Equivalent of matrix(unlist(ze), 8, N)

#         # Set rownames equivalent in Python (using Pandas DataFrame)
#         ze_df = pd.DataFrame(ze.T, columns=["cos", "sin", "Q", "Q.old", "Q1", "Q2", "Q3", "Q4"])
#         phi = np.arctan2(ze_df['sin'], ze_df['cos']) % (2 * np.pi)

#         # Store Qhist on the first iteration
#         if i == 0:
#             Qhist = ze_df.copy()
#             Qhist['iteration'] = i + 1  # Iteration column
#             Qhist['sample'] = np.arange(1, N + 1)  # Sample numbers starting from 1
#         else:
#             Qtemp = ze_df.copy()
#             Qtemp['iteration'] = i + 1
#             Qtemp['sample'] = np.arange(1, N + 1)
#             Qhist = pd.concat([Qhist, Qtemp])  # Append the new iteration results

#         # Update weights W
#         P1_0 = np.array([np.exp(E[p, :] @ X @ Minv @ X.T @ E[p, :].T / (2 * sigma2)) * np.sqrt(dTinv * sigma2**3 / np.linalg.det(M)) for p in range(Ng)])
#         W = q * P1_0 / (1 - q + q * P1_0)
#         W[np.isnan(W)] = 1  # Replace NaN with 1

#         # Exit the loop if convergence is reached
#         if np.max(np.abs(phi - phi_old)) < 0.001:
#             if pbar:
#                 print("\nAlgorithm has converged\n")
#             return {
#                 "phi": phi, "sigma": sigma2, "alpha": alpha, "weights": W,
#                 "iteration": i + 1, "sigma.m1": sigma2_m1, "E": E_full, "Qhist": Qhist, "geni": geni
#             }

#         # Update parameters for the next EM step
#         cos_phi = np.cos(phi)
#         sin_phi = np.sin(phi)
#         X = np.column_stack((np.ones(len(phi)), cos_phi, sin_phi))  # Equivalent of cbind(1, cos(phi), sin(phi))

#         Mold = Nn + sigma2 * Tinv
#         Moldinv = np.linalg.inv(Mold)  # Inverse of Mold

#         # Update sigma2.m1
#         sigma2_m1 = np.array([np.sum(np.diag(S[s] - S[s] @ Xold @ Moldinv @ X.T)) / N + 0.01 for s in range(Ng)])
#         sigma2_m0 = np.var(E, axis=1)
#         sigma2 = np.mean(sigma2_m1 * W + sigma2_m0 * (1 - W))
#         sigma2_m1 = np.sum(sigma2_m1 * W) / np.sum(W)

#         # Update q if needed
#         if update_q:
#             q = np.mean(W)
#             q = max(0.05, min(q, 0.3))

#         if pbar:
#             print(f"Iteration {i + 1} completed")

#     if pbar:
#         print("\nEM algorithm finished after maximum iterations\n")

#     return {
#         "phi": phi, "Qhist": Qhist, "sigma": sigma2, "alpha": alpha, "weights": W,
#         "iteration": iterations, "sigma.m1": sigma2_m1, "E": E_full
#     }


def J_tilde(E, n_genes=0, n_samples=0):
    """
    Calculates the interaction matrix for the spin glass model.

    Parameters:
        E (numpy.ndarray): Expression matrix with genes on rows and samples on columns.
        n_genes (int): Number of genes (rows). If 0, use all genes.
        n_samples (int): Number of samples (columns). If 0, use all samples.

    Returns:
        numpy.ndarray: Interaction matrix (n_samples x n_samples).
    """
    # Convert E to numpy array if it isn't already
    E = np.array(E)

    # Default to using all genes and all samples if values are not provided
    if n_genes == 0:
        n_genes = E.shape[1]
    if n_samples == 0:
        n_samples = E.shape[0]

    # Initialize the interaction matrix with zeros
    Jtilde = np.zeros((n_samples, n_samples))

    # Calculate the interaction matrix
    for i in range(n_samples):
        for j in range(n_samples):
            Jtilde[i, j] = np.sum(E[i, :] * E[j, :]) / (n_samples * n_genes)

    # Set the diagonal elements to zero
    np.fill_diagonal(Jtilde, 0)

    return Jtilde


def Zeta_mf_ordered(J, beta, n_samples, A_0=0.1, iterations=1000):
    """
    Spin glass approximation to initialize the EM model.

    Parameters:
        J (numpy.ndarray): Interaction matrix.
        beta (float): Temperature-like parameter controlling phase interactions.
        n_samples (int): Number of samples (columns of expression matrix).
        A_0 (float): Initial condition for the order parameter A.
        iterations (int): Maximum number of iterations.

    Returns:
        numpy.ndarray: Matrix containing amplitudes (A) and phases (Theta).
    """
    A = np.full(n_samples, A_0)
    # Theta = np.random.uniform(0, 2 * np.pi, n_samples)
    Theta = np.linspace(0, 2 * np.pi, n_samples)

    for _ in range(iterations):
        A_cos = A * np.cos(Theta)
        A_sin = A * np.sin(Theta)

        for k in range(n_samples):
            u = beta * np.sum(A_cos * J[:, k])
            v = beta * np.sum(A_sin * J[:, k])
            mod = np.sqrt(u**2 + v**2)
            Zeta_k = [u / mod, v / mod] if mod > 0 else [1, 0]
            Theta[k] = np.arctan2(Zeta_k[1], Zeta_k[0])
            A[k] = iv(1, mod) / iv(0, mod) if mod <= 20 else 1

    return np.column_stack([A, Theta % (2 * np.pi)])
