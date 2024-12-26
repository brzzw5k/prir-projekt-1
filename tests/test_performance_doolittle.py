import numpy as np
import pytest
from src.doolittle_factorization import DoolittleFactorization



@pytest.mark.benchmark(group="doolittle_factorization_seq")
@pytest.mark.parametrize("matrix_size", [500])
def test_doolittle_factorization_sequential_performance(benchmark, matrix_size):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size)

    def doolittle_sequential():
        return DoolittleFactorization.sequential(A)

    benchmark(doolittle_sequential)


@pytest.mark.benchmark(group="doolittle_factorization_parallel")
@pytest.mark.parametrize("matrix_size", [500])
@pytest.mark.parametrize("n_jobs", [2,4])
def test_doolittle_factorization_parallel_performance(benchmark, matrix_size, n_jobs):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size)

    def doolittle_parallel():
        return DoolittleFactorization.parallel(A, n_jobs=n_jobs)

    benchmark(doolittle_parallel)
