import numpy as np
import pytest
from src.doolittle_factorization import DoolittleFactorization
import time

@pytest.mark.benchmark(group="doolittle_factorization")
@pytest.mark.parametrize("matrix_size", [3, 10, 25, 100])
def test_doolittle_factorization_performance(benchmark, matrix_size):
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size)

    def doolittle_sequential():
        DoolittleFactorization.sequential(A)

    benchmark(doolittle_sequential)
