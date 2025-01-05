import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor


@njit(fastmath=True, cache=True)
def _initialize_lu(A: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n = A.shape[0]
    L = np.eye(n, dtype=A.dtype)
    U = np.zeros_like(A, dtype=A.dtype)
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
                executor.submit(compute_u, k).result()
                executor.submit(compute_l, k).result()

        return L, U

    cuda_mod = SourceModule("""
    __global__ void compute_u(double *A, double *L, double *U, int n, int k) {
        int j = threadIdx.x + blockIdx.x * blockDim.x;
        if (j >= k && j < n) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += L[k * n + m] * U[m * n + j];
            }
            U[k * n + j] = A[k * n + j] - sum;
        }
    }

    __global__ void compute_l(double *A, double *L, double *U, int n, int k) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i > k && i < n) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += L[i * n + m] * U[m * n + k];
            }
            double U_kk = U[k * n + k];
            if (fabs(U_kk) > 1e-10) {
                L[i * n + k] = (A[i * n + k] - sum) / U_kk;
            } else {
                printf("Error: Division by zero in U[%d][%d]\\n", k, k);
            }
        }
    }
    """)

    compute_u_cuda = cuda_mod.get_function("compute_u")
    compute_l_cuda = cuda_mod.get_function("compute_l")

    @staticmethod
    def parallel_pycuda(
        A: np.ndarray, block_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        n, L, U = _initialize_lu(A)

        A_gpu = cuda.mem_alloc(A.nbytes)
        L_gpu = cuda.mem_alloc(A.nbytes)
        U_gpu = cuda.mem_alloc(A.nbytes)

        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(L_gpu, L)
        cuda.memcpy_htod(U_gpu, U)

        grid_size = (n + block_size - 1) // block_size

        for k in range(n):
            DoolittleFactorization.compute_u_cuda(
                A_gpu,
                L_gpu,
                U_gpu,
                np.int32(n),
                np.int32(k),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
            )
            DoolittleFactorization.compute_l_cuda(
                A_gpu,
                L_gpu,
                U_gpu,
                np.int32(n),
                np.int32(k),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
            )

        L_host = np.empty_like(A, dtype=np.float64)
        U_host = np.empty_like(A, dtype=np.float64)
        cuda.memcpy_dtoh(L_host, L_gpu)
        cuda.memcpy_dtoh(U_host, U_gpu)

        return L_host, U_host
