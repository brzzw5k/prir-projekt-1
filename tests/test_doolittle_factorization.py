import pytest
import numpy as np
import numba
from src.doolittle_factorization import DoolittleFactorization
from tests.helper import check_frobenius_norm_lu


@pytest.mark.parametrize(
    "A, L_expected, U_expected",
    [
        (np.array([[2]], dtype=np.float64), np.array([[1]]), np.array([[2]])),
        (
            np.array([[1, 2], [3, 4]], dtype=np.float64),
            np.array([[1, 0], [3, 1]]),
            np.array([[1, 2], [0, -2]]),
        ),
        (
            np.array([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]], dtype=np.float64),
            np.array([[1, 0, 0], [-2, 1, 0], [-2, -1, 1]]),
            np.array([[2, -1, -2], [0, 4, -1], [0, 0, 3]]),
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
        (np.array([[2]], dtype=np.float64), np.array([[1]]), np.array([[2]])),
        (
            np.array([[1, 2], [3, 4]], dtype=np.float64),
            np.array([[1, 0], [3, 1]]),
            np.array([[1, 2], [0, -2]]),
        ),
        (
            np.array([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]], dtype=np.float64),
            np.array([[1, 0, 0], [-2, 1, 0], [-2, -1, 1]]),
            np.array([[2, -1, -2], [0, 4, -1], [0, 0, 3]]),
        ),
    ],
)
def test_doolittle_factorization_parallel(A, L_expected, U_expected):
    L, U = DoolittleFactorization.parallel_numba(A)
    numba.set_num_threads(numba.config.NUMBA_DEFAULT_NUM_THREADS)
    np.testing.assert_allclose(L, L_expected)
    np.testing.assert_allclose(U, U_expected)
    assert check_frobenius_norm_lu(A, L, U) == 0


@pytest.mark.parametrize(
    "A, L_expected, U_expected",
    [
        (np.array([[2]], dtype=np.float64), np.array([[1]]), np.array([[2]])),
        (
            np.array([[1, 2], [3, 4]], dtype=np.float64),
            np.array([[1, 0], [3, 1]]),
            np.array([[1, 2], [0, -2]]),
        ),
        (
            np.array([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]], dtype=np.float64),
            np.array([[1, 0, 0], [-2, 1, 0], [-2, -1, 1]]),
            np.array([[2, -1, -2], [0, 4, -1], [0, 0, 3]]),
        ),
    ],
)
@pytest.mark.parametrize("n_threads", [2, 3, 4])
def test_doolittle_factorization_parallel_futures(A, L_expected, U_expected, n_threads):
    L, U = DoolittleFactorization.parallel_threads(A, n_threads)
    np.testing.assert_allclose(L, L_expected)
    np.testing.assert_allclose(U, U_expected)
    assert check_frobenius_norm_lu(A, L, U) == 0


@pytest.mark.parametrize(
    "A, L_expected, U_expected",
    [
        (np.array([[2]], dtype=np.float64), np.array([[1]]), np.array([[2]])),
        (
            np.array([[1, 2], [3, 4]], dtype=np.float64),
            np.array([[1, 0], [3, 1]]),
            np.array([[1, 2], [0, -2]]),
        ),
        (
            np.array([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]], dtype=np.float64),
            np.array([[1, 0, 0], [-2, 1, 0], [-2, -1, 1]]),
            np.array([[2, -1, -2], [0, 4, -1], [0, 0, 3]]),
        ),
    ],
)
def test_doolittle_factorization_parallel_pycuda(A, L_expected, U_expected):
    L, U = DoolittleFactorization.parallel_pycuda(A)
    np.testing.assert_allclose(L, L_expected)
    np.testing.assert_allclose(U, U_expected)
    assert check_frobenius_norm_lu(A, L, U) == 0
