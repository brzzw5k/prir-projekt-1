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
    def parallel_pycuda(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, L, U = _initialize_lu(A)

        mod = SourceModule("""
        __global__ void compute_u(float *A, float *L, float *U, int n, int k) {
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (j >= k && j < n) {
                float sum = 0.0;
                for (int m = 0; m < k; ++m) {
                    sum += L[k * n + m] * U[m * n + j];
                }
                U[k * n + j] = A[k * n + j] - sum;
            }
        }

        __global__ void compute_l(float *A, float *L, float *U, int n, int k) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i > k && i < n) {
                float sum = 0.0;
                for (int m = 0; m < k; ++m) {
                    sum += L[i * n + m] * U[m * n + k];
                }
                L[i * n + k] = (A[i * n + k] - sum) / U[k * n + k];
            }
        }
        """)

        compute_u = mod.get_function("compute_u")
        compute_l = mod.get_function("compute_l")

        A = A.astype(np.float32)
        L = L.astype(np.float32)
        U = U.astype(np.float32)

        A_gpu = drv.mem_alloc(A.nbytes)
        L_gpu = drv.mem_alloc(L.nbytes)
        U_gpu = drv.mem_alloc(U.nbytes)

        drv.memcpy_htod(A_gpu, A)
        drv.memcpy_htod(L_gpu, L)
        drv.memcpy_htod(U_gpu, U)

        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        for k in range(n):
            compute_u(
                A_gpu,
                L_gpu,
                U_gpu,
                np.int32(n),
                np.int32(k),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
            )
            drv.Context.synchronize()
            compute_l(
                A_gpu,
                L_gpu,
                U_gpu,
                np.int32(n),
                np.int32(k),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
            )

        drv.memcpy_dtoh(L, L_gpu)
        drv.memcpy_dtoh(U, U_gpu)

        return L, U
