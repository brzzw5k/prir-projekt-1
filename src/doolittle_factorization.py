import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def _initialize_lu(A: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    U = np.zeros((n, n), dtype=A.dtype)
    return n, L, U


class DoolittleFactorization:
    """
    Doolittle's factorization of a square matrix A into
    a lower triangular matrix L and an upper triangular matrix U
    such that A = L*U.
    """

    @staticmethod
    @njit(fastmath=True, cache=True)
    def sequential(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, L, U = _initialize_lu(A)

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

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def parallel(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, L, U = _initialize_lu(A)

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
