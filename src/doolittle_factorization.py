import numpy as np
from joblib import Parallel, delayed

class DoolittleFactorization:
    """
    Doolittle's factorization of a square matrix A into
    a lower triangular matrix L and an upper triangular matrix U
    such that A = LU.

    Parameters:delayed
        A: A square matrix of shape (n, n)

    Attributes:
        A: A square matrix of shape (n, n)
        n: The size of the square matrix A
        U: An upper triangular matrix of shape (n, n)
        L: A lower triangular matrix of shape (n, n)
    """

    def __init__(self, A: np.ndarray) -> None:
        self.A = A
        self.n = A.shape[0]
        self.U = np.zeros((self.n, self.n))
        self.L = np.eye(self.n)

    def _compute_U(self, k: int, j: int) -> None:
        self.U[k, j] = self.A[k, j]
        for m in range(k):
            self.U[k, j] -= self.L[k, m] * self.U[m, j]

    def _compute_L(self, k: int, i: int) -> None:
        self.L[i, k] = self.A[i, k]
        for m in range(k):
            self.L[i, k] -= self.L[i, m] * self.U[m, k]
        self.L[i, k] /= self.U[k, k]

    @staticmethod
    def sequential(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        df = DoolittleFactorization(A)
        for k in range(df.n):
            for j in range(k, df.n):
                df._compute_U(k, j)
            for i in range(k + 1, df.n):
                df._compute_L(k, i)
        return df.L, df.U

    @staticmethod
    def parallel(A: np.ndarray, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
        df = DoolittleFactorization(A)

        for k in range(df.n):
            Parallel(n_jobs=n_jobs)(
                delayed(df._compute_U)(k, j)
                for j in range(k, df.n)
            )
            Parallel(n_jobs=n_jobs)(
                delayed(df._compute_L)(k, i)
                for i in range(k + 1, df.n)
            )

        return df.L, df.U