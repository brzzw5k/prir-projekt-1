import numpy as np
import pytest
from src.doolittle_factorization import DoolittleFactorization
import numba


@pytest.mark.benchmark(group="doolittle_factorization_seq")
@pytest.mark.parametrize("matrix_size", [4000])
def test_doolittle_factorization_sequential_performance(benchmark, matrix_size):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    def doolittle_sequential():
        return DoolittleFactorization.sequential(A)

    benchmark(doolittle_sequential)


@pytest.mark.benchmark(group="doolittle_factorization_parallel_numba")
@pytest.mark.parametrize("matrix_size", [7000])
@pytest.mark.parametrize("n_threads", [2, 3, 4])
def test_doolittle_factorization_parallel_numba_performance(
    benchmark, matrix_size, n_threads
):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    numba.set_num_threads(n_threads)

    def doolittle_parallel_numba():
        return DoolittleFactorization.parallel_numba(A)

    benchmark(doolittle_parallel_numba)


@pytest.mark.benchmark(group="doolittle_factorization_parallel_threads")
@pytest.mark.parametrize("matrix_size", [7000])
@pytest.mark.parametrize("n_threads", [2, 3, 4])
def test_doolittle_factorization_parallel_threads_performance(
    benchmark, matrix_size, n_threads
):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    def doolittle_parallel_threads():
        return DoolittleFactorization.parallel_threads(A, n_threads)

    benchmark(doolittle_parallel_threads)


@pytest.mark.skip(reason="Skipping parallel performance test on CI")
@pytest.mark.benchmark(group="doolittle_factorization_parallel_numba")
@pytest.mark.parametrize("matrix_size", [7000])
@pytest.mark.parametrize("n_threads", [2, 4, 6, 8, 10, 12, 14, 16])
def test_doolittle_factorization_parallel_numba_performance_more_threads(
    benchmark, matrix_size, n_threads
):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    numba.set_num_threads(n_threads)

    def doolittle_parallel_numba():
        return DoolittleFactorization.parallel_numba(A)

    benchmark(doolittle_parallel_numba)
