import numpy as np
from joblib import Parallel, delayed


class DoolittleFactorization:
    """
    Doolittle's factorization of a square matrix A into
    a lower triangular matrix L and an upper triangular matrix U
    such that A = L*U.
    """

    def __init__(self, A: np.ndarray) -> None:
        self.A = A
        self.n = A.shape[0]
        self.U = np.zeros((self.n, self.n), dtype=A.dtype)
        self.L = np.eye(self.n, dtype=A.dtype)

    def _compute_U_element(self, k: int, j: int) -> None:
        """
        U[k,j].
        """
        tmp = self.A[k, j]
        for m in range(k):
            tmp -= self.L[k, m] * self.U[m, j]
        self.U[k, j] = tmp

    def _compute_L_element(self, k: int, i: int) -> None:
        """
        L[i,k].
        """
        tmp = self.A[i, k]
        for m in range(k):
            tmp -= self.L[i, m] * self.U[m, k]
        self.L[i, k] = tmp / self.U[k, k]

    @staticmethod
    def sequential(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        df = DoolittleFactorization(A)
        n = df.n
        for k in range(n):
            for j in range(k, n):
                df._compute_U_element(k, j)
            for i in range(k + 1, n):
                df._compute_L_element(k, i)
        return df.L, df.U

    @staticmethod
    def parallel(A: np.ndarray, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """
        Parallel method on rows and colls
        """
        df = DoolittleFactorization(A)
        n = df.n
        for k in range(n):
            Parallel(n_jobs=n_jobs)(
                delayed(df._compute_U_element)(k, j) for j in range(k, n)
            )
            Parallel(n_jobs=n_jobs)(
                delayed(df._compute_L_element)(k, i) for i in range(k + 1, n)
            )
        return df.L, df.U
