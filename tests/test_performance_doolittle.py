import numpy as np
import pytest
from src.doolittle_factorization import DoolittleFactorization

@pytest.mark.benchmark(group="doolittle_factorization_seq")
@pytest.mark.parametrize("matrix_size", [200])
def test_doolittle_factorization_sequential_performance(benchmark, matrix_size):
    """
    Sequential version execution time measurement
    """
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size)

    def doolittle_sequential():
        DoolittleFactorization.sequential(A)

    benchmark(doolittle_sequential)


@pytest.mark.benchmark(group="doolittle_factorization_parallel_rows_cols")
@pytest.mark.parametrize("matrix_size", [200])
@pytest.mark.parametrize("n_jobs", [4])
def test_doolittle_factorization_parallel(benchmark, matrix_size, n_jobs):
    """
    Parallel version (rows & cols) execution time measurement
    with different n_jobs (number of cores).
    """
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size)

    def doolittle_parallel():
        DoolittleFactorization.parallel(A, n_jobs=n_jobs)

    benchmark(doolittle_parallel)

