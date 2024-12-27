import numpy as np
from numba import njit, prange


class DoolittleFactorization:
    """
    Doolittle's factorization of a square matrix A into
    a lower triangular matrix L and an upper triangular matrix U
    such that A = L*U.
    """

    def __init__(self, A: np.ndarray) -> None:
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        self.A = A.copy()
        self.n = A.shape[0]
        self.L = np.eye(self.n, dtype=A.dtype)
        self.U = np.zeros((self.n, self.n), dtype=A.dtype)

    @staticmethod
    def sequential(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform LU factorization sequentially using Numba.
        Returns lower and upper triangular matrices L and U.
        """
        L, U = lu_doolittle_numba(A)
        return L, U

    @staticmethod
    def parallel(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform LU factorization using Numba with parallelization.
        Only the computation of L[i, k] is parallelized to maintain correctness.
        Returns lower and upper triangular matrices L and U.
        """
        L, U = lu_doolittle_numba_parallel_corrected(A)
        return L, U


@njit(fastmath=True, cache=True)
def lu_doolittle_numba(A):
    """
    Perform Doolittle's LU factorization with Numba optimization (sequential).
    Returns lower and upper triangular matrices L and U.
    """
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    U = np.zeros((n, n), dtype=A.dtype)

    for k in range(n):
        # Compute U[k, j] for j = k to n-1
        for j in range(k, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[k, m] * U[m, j]
            U[k, j] = A[k, j] - sum_

        # Compute L[i, k] for i = k+1 to n-1
        for i in range(k + 1, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[i, m] * U[m, k]
            L[i, k] = (A[i, k] - sum_) / U[k, k]

    return L, U


@njit(parallel=True, fastmath=True, cache=True)
def lu_doolittle_numba_parallel_corrected(A):
    """
    Perform Doolittle's LU factorization with Numba's parallelization.
    Only the computation of L[i, k] is parallelized to maintain correctness.
    Returns lower and upper triangular matrices L and U.
    """
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    U = np.zeros((n, n), dtype=A.dtype)

    for k in range(n):
        # Compute U[k, j] sequentially
        for j in range(k, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[k, m] * U[m, j]
            U[k, j] = A[k, j] - sum_

        # Parallelize the computation of L[i, k] for i = k+1 to n-1
        for i in prange(k + 1, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[i, m] * U[m, k]
            L[i, k] = (A[i, k] - sum_) / U[k, k]

    return L, U
