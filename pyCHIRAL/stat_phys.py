import numpy as np
from scipy.special import iv  # Modified Bessel function of the first kind


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
    # Theta = np.linspace(0, 2 * np.pi, n_samples)

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
