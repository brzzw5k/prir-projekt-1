import numpy as np
import pytest
from src.doolittle_factorization import DoolittleFactorization
import numba

from tests.helper import check_frobenius_norm_lu


@pytest.mark.benchmark(group="doolittle_factorization_seq")
@pytest.mark.parametrize("matrix_size", [6000])
def test_doolittle_factorization_sequential_performance(benchmark, matrix_size):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    def doolittle_sequential():
        return DoolittleFactorization.sequential(A)

    L, U = benchmark(doolittle_sequential)
    np.testing.assert_almost_equal(check_frobenius_norm_lu(A, L, U), 0, decimal=7)


@pytest.mark.benchmark(group="doolittle_factorization_parallel_numba")
@pytest.mark.parametrize("matrix_size", [6000])
@pytest.mark.parametrize("n_threads", [2, 4, 8, 12, 16])
def test_doolittle_factorization_parallel_numba_performance(
    benchmark, matrix_size, n_threads
):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    numba.set_num_threads(n_threads)

    def doolittle_parallel_numba():
        return DoolittleFactorization.parallel_numba(A)

    L, U = benchmark(doolittle_parallel_numba)
    np.testing.assert_almost_equal(check_frobenius_norm_lu(A, L, U), 0, decimal=7)


@pytest.mark.benchmark(group="doolittle_factorization_parallel_threads")
@pytest.mark.parametrize("matrix_size", [6000])
@pytest.mark.parametrize("n_threads", [2, 4, 8, 12, 16])
def test_doolittle_factorization_parallel_threads_performance(
    benchmark, matrix_size, n_threads
):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    def doolittle_parallel_threads():
        return DoolittleFactorization.parallel_threads(A, n_threads)

    L, U = benchmark(doolittle_parallel_threads)
    np.testing.assert_almost_equal(check_frobenius_norm_lu(A, L, U), 0, decimal=7)


@pytest.mark.benchmark(group="doolittle_factorization_pycuda")
@pytest.mark.parametrize("matrix_size", [6000])
@pytest.mark.parametrize("block_size", [4, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("grid_size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_doolittle_factorization_parallel_parallel_pycuda(
    benchmark, matrix_size, block_size, grid_size
):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    def doolittle_parallel_pycuda():
        return DoolittleFactorization.parallel_pycuda(A, block_size, grid_size)

    L, U = benchmark(doolittle_parallel_pycuda)
#     np.testing.assert_almost_equal(check_frobenius_norm_lu(A, L, U), 0, decimal=7)
