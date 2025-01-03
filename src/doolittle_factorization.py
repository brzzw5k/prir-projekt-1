import numpy as np
import pycuda.driver as drv
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor


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
    def parallel_numba(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def parallel_threads(
        A: np.ndarray, n_threads: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n, L, U = _initialize_lu(A)

        def compute_u(k: int):
            for j in range(k, n):
                sum_ = 0.0
                for m in range(k):
                    sum_ += L[k, m] * U[m, j]
                U[k, j] = A[k, j] - sum_

        def compute_l(k: int):
            for i in range(k + 1, n):
                sum_ = 0.0
                for m in range(k):
                    sum_ += L[i, m] * U[m, k]
                L[i, k] = (A[i, k] - sum_) / U[k, k]

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for k in range(n):
                executor.submit(compute_u, k)
                executor.submit(compute_l, k)

        return L, U

    @staticmethod
    def parallel_pycuda(
        A: np.ndarray, block_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        n, L, U = _initialize_lu(A)
        
        # for now that seems to be suboptimal solution, or pytest benchmark might be lying
        mod = SourceModule("""
        __global__ void doolittle(float *A, float *L, float *U, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                for (int k = 0; k < n; ++k) {
                    if (idx >= k && idx < n) {
                        float sum = 0.0;
                        for (int m = 0; m < k; ++m) {
                            sum += L[k * n + m] * U[m * n + idx];
                        }
                        U[k * n + idx] = A[k * n + idx] - sum;
                    }
                    __syncthreads();
                    if (idx > k && idx < n) {
                        float sum = 0.0;
                        for (int m = 0; m < k; ++m) {
                            sum += L[idx * n + m] * U[m * n + k];
                        }
                        L[idx * n + k] = (A[idx * n + k] - sum) / U[k * n + k];
                    }
                    __syncthreads();
                }
            }
        }
        """)

        doolittle = mod.get_function("doolittle")

        A = A.astype(np.float32)
        L = L.astype(np.float32)
        U = U.astype(np.float32)

        A_gpu = drv.mem_alloc(A.nbytes)
        L_gpu = drv.mem_alloc(L.nbytes)
        U_gpu = drv.mem_alloc(U.nbytes)

        drv.memcpy_htod(A_gpu, A)
        drv.memcpy_htod(L_gpu, L)
        drv.memcpy_htod(U_gpu, U)

        grid_size = (n + block_size - 1) // block_size

        doolittle(
            A_gpu,
            L_gpu,
            U_gpu,
            np.int32(n),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        drv.memcpy_dtoh(L, L_gpu)
        drv.memcpy_dtoh(U, U_gpu)

        return L, U
