import numpy as np


def check_frobenius_norm_lu(A: np.ndarray, L: np.ndarray, U: np.ndarray) -> float:
    """Returns the Frobenius norm of the difference between the product
    of the lower and upper triangular matrices and the original matrix A.
    If returns 0, the factorization is correct."""
    return np.linalg.norm(L @ U - A, "fro")
