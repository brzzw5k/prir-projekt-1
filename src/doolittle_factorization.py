import numpy as np
from numba import njit, prange


class DoolittleFactorization:
    """
    Doolittle's factorization of a square matrix A into
    a lower triangular matrix L and an upper triangular matrix U
    such that A = LU.

    Parameters:
        A: A square matrix of shape (n, n)
    Attributes:
        A: A square matrix of shape (n, n)
        n: The size of the square matrix A
        U: An upper triangular matrix of shape (n, n)
        L: A lower triangular matrix of shape (n, n)
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
        L, U = lu_doolittle_numba(A)
        return L, U

    @staticmethod
    def parallel(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        L, U = lu_doolittle_numba_parallel_corrected(A)
        return L, U


@njit(fastmath=True, cache=True)
def lu_doolittle_numba(A):
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    U = np.zeros((n, n), dtype=A.dtype)

    for k in range(n):
        for j in range(k, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[k, m] * U[m, j]
            U[k, j] = A[k, j] - sum_
        for i in range(k + 1, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[i, m] * U[m, k]
            L[i, k] = (A[i, k] - sum_) / U[k, k]
    return L, U


@njit(parallel=True, fastmath=True, cache=True)
def lu_doolittle_numba_parallel_corrected(A):
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    U = np.zeros((n, n), dtype=A.dtype)

    for k in range(n):
        for j in range(k, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[k, m] * U[m, j]
            U[k, j] = A[k, j] - sum_
        for i in prange(k + 1, n):
            sum_ = 0.0
            for m in range(k):
                sum_ += L[i, m] * U[m, k]
            L[i, k] = (A[i, k] - sum_) / U[k, k]
    return L, U
