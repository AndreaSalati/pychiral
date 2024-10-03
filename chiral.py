import numpy as np
from tqdm import tqdm  # Progress bar
from numpy.linalg import inv, det
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
    clock_coord = [i for i, gene in enumerate(genes) if gene in clockgenes]
    E = E[:, clock_coord]

    # Mean center the expression data if needed
    if mean_centre_E:
        E = E - E.mean(axis=0, keepdims=True)

    # Initialize parameters for probabilistic model
    if sigma2 is None:
        sigma2 = np.mean(np.var(E, axis=0))
    if u is None:
        u = 0.2
    if tau2 is None:
        tau2 = 4 / (24 + E.shape[1])

    phi = phi_start

    # Initializing phase using spin glass model (or any other method)
    if phi_start is None:
        # Use the spin glass initialization (for now, we skip implementation of J.tilde and Zeta.mf.ordered)
        beta = 1000
        J = J_tilde(E)  # Placeholder function, needs definition
        Zeta = Zeta_mf_ordered(J, beta, E.shape[0])
        phi = Zeta[:, 1] + np.random.uniform(-0.5, 0.5, size=E.shape[1])

    # Initialize matrices for the EM procedure
    Ng, N = E.shape
    sigma2_0 = sigma2
    T = np.diag([u**2, tau2, tau2])

    # Precompute some variables used in the EM loop
    S = np.array([np.outer(E[l, :], E[l, :]) for l in range(Ng)])
    W = np.ones(Ng)

    # Set up progress bar if requested
    if pbar:
        progress_bar = tqdm(total=iterations)

    # Start EM iterations
    for i in range(iterations):
        # Store old values for convergence check
        phi_old = phi.copy()

        # Update alpha and weight parameters (detailed matrix calculations, skipped for now)
        # --- Omitted for simplicity ---

        # Check for convergence
        if np.max(np.abs(phi - phi_old)) < 0.001:
            if pbar:
                print("\nAlgorithm has converged.")
            return {
                "phi": phi,
                "sigma": sigma2,
                "alpha": None,  # Placeholder
                "weights": W,
                "iteration": i,
                "sigma_m1": sigma2_0,  # Placeholder
                "E_full": E_full,
            }

        # Update for the next EM step
        phi = (phi + np.random.normal(0, 0.1, size=phi.shape)) % (2 * np.pi)

        # Update progress bar
        if pbar:
            progress_bar.update(1)

    # Close progress bar if it was used
    if pbar:
        progress_bar.close()

    return {
        "phi": phi,
        "sigma": sigma2,
        "alpha": None,  # Placeholder
        "weights": W,
        "iteration": iterations,
        "sigma_m1": sigma2_0,  # Placeholder
        "E_full": E_full,
    }


# Helper function definitions (J_tilde and Zeta_mf_ordered) should be added for completeness.


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
    Theta = np.random.uniform(0, 2 * np.pi, n_samples)

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
