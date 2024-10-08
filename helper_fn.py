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


def process_expression_data(
    E, clockgenes=None, ccg=None, layer=None, standardize=False, mean_centre_E=False
):
    """
    Processes the expression data based on the provided parameters.

    Parameters:
    - E: The expression data object (assumed to have X and layers attributes).
    - clockgenes: List of clock genes to select. If None, uses ccg.
    - ccg: Default list of clock genes if clockgenes is None.
    - layer: Specific layer of expression data to use.
    - standardize: Whether to standardize the expression data.
    - mean_centre_E: Whether to mean center the expression data.

    Returns:
    - E: Processed expression data.
    - clock_coord: Indices of the selected clock genes.
    - N: Number of samples.
    - Ng: Number of genes.
    """

    if clockgenes is None:
        clockgenes = ccg

    E_full = E.copy()
    genes = E_full.var_names

    if layer is None:
        E = E.X
    else:
        E = E.layers[layer]

    # Standardize the matrix if needed
    if standardize:
        E = E / np.std(E, axis=0, keepdims=True)

    # Mean center the expression data if needed
    if mean_centre_E:
        E = E - E.mean(axis=0, keepdims=True)

    # Clock gene selection
    clock_coord = ind2(genes, clockgenes)
    E = E[:, clock_coord]

    N, Ng = E.shape

    return E, E_full, clock_coord, N, Ng


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

    print("zero", zero)
    print("one", one)
    print("two", two)
    print("three", three)
    print("four", four)

    # Roots of the polynomial (use numpy roots for equivalent of polyroot)
    return np.roots([four, three, two, one, zero])
