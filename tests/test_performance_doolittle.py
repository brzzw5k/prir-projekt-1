import numpy as np
import pytest
from src.doolittle_factorization import DoolittleFactorization


@pytest.mark.benchmark(group="doolittle_factorization_seq")
@pytest.mark.parametrize("matrix_size", [3, 10, 25, 100])
def test_doolittle_factorization_sequential_performance(benchmark, matrix_size):
    """
    Sequential version execution time measurement
    """
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size)

    def doolittle_sequential():
        DoolittleFactorization.sequential(A)

    benchmark(doolittle_sequential)


@pytest.mark.benchmark(group="doolittle_factorization_par")
@pytest.mark.parametrize("matrix_size", [3, 10, 25, 100])
def test_doolittle_factorization_parallel_performance(benchmark, matrix_size):
    """
    Parallel version execution time measurement
    """
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size)

    def doolittle_parallel():
        DoolittleFactorization.parallel(A, n_jobs=4)

    benchmark(doolittle_parallel)
