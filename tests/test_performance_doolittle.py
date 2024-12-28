import numpy as np
import pytest
from src.doolittle_factorization import DoolittleFactorization
import numba


@pytest.mark.benchmark(group="doolittle_factorization_seq")
@pytest.mark.parametrize("matrix_size", [7000])
def test_doolittle_factorization_sequential_performance(benchmark, matrix_size):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    def doolittle_sequential():
        return DoolittleFactorization.sequential(A)

    benchmark(doolittle_sequential)


@pytest.mark.benchmark(group="doolittle_factorization_parallel")
@pytest.mark.parametrize("matrix_size", [7000])
@pytest.mark.parametrize("n_threads", [2, 3, 4])
def test_doolittle_factorization_parallel_performance(
    benchmark, matrix_size, n_threads
):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)

    numba.set_num_threads(n_threads)

    def doolittle_parallel():
        return DoolittleFactorization.parallel(A)

    benchmark(doolittle_parallel)
