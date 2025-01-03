import pytest
import numpy as np
from src.doolittle_factorization import DoolittleFactorization
import numba


def check_frobenius_norm_lu(A: np.ndarray, L: np.ndarray, U: np.ndarray) -> float:
    """Returns the Frobenius norm of the difference between the product
    of the lower and upper triangular matrices and the original matrix A.
    If returns 0, the factorization is correct."""
    return np.linalg.norm(L @ U - A, "fro")


@pytest.mark.parametrize(
    "A, L_expected, U_expected",
    [
        (np.array([[2]]), np.array([[1]]), np.array([[2]])),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 0], [3, 1]]),
            np.array([[1, 2], [0, -2]]),
        ),
    ],
)
def test_doolittle_factorization_sequential(A, L_expected, U_expected):
    L, U = DoolittleFactorization.sequential(A)
    np.testing.assert_allclose(L, L_expected)
    np.testing.assert_allclose(U, U_expected)
    assert check_frobenius_norm_lu(A, L, U) == 0


@pytest.mark.parametrize(
    "A, L_expected, U_expected",
    [
        (np.array([[2]]), np.array([[1]]), np.array([[2]])),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 0], [3, 1]]),
            np.array([[1, 2], [0, -2]]),
        ),
    ],
)
@pytest.mark.parametrize("n_threads", [2, 3, 4])
def test_doolittle_factorization_parallel(A, L_expected, U_expected, n_threads):
    numba.set_num_threads(n_threads)

    L, U = DoolittleFactorization.parallel_numba(A)
    numba.set_num_threads(numba.config.NUMBA_DEFAULT_NUM_THREADS)
    np.testing.assert_allclose(L, L_expected)
    np.testing.assert_allclose(U, U_expected)
    assert check_frobenius_norm_lu(A, L, U) == 0


@pytest.mark.parametrize(
    "A, L_expected, U_expected",
    [
        (np.array([[2]]), np.array([[1]]), np.array([[2]])),
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 0], [3, 1]]),
            np.array([[1, 2], [0, -2]]),
        ),
    ],
)
@pytest.mark.parametrize("n_threads", [2, 3, 4])
def test_doolittle_factorization_parallel_futures(A, L_expected, U_expected, n_threads):
    L, U = DoolittleFactorization.parallel_threads(A, n_threads)
    np.testing.assert_allclose(L, L_expected)
    np.testing.assert_allclose(U, U_expected)
    assert check_frobenius_norm_lu(A, L, U) == 0
