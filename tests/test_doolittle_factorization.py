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
    """
    Test Doolittle's LU factorization using the sequential method.
    """
    L, U = DoolittleFactorization.sequential(A)
    np.testing.assert_allclose(
        L, L_expected, rtol=1e-5, atol=1e-8, err_msg="Sequential L matrix mismatch."
    )
    np.testing.assert_allclose(
        U, U_expected, rtol=1e-5, atol=1e-8, err_msg="Sequential U matrix mismatch."
    )
    assert (
        check_frobenius_norm_lu(A, L, U) == 0
    ), "Sequential LU factorization is incorrect."


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
@pytest.mark.parametrize("n_threads", [2, 4, 8])
def test_doolittle_factorization_parallel(A, L_expected, U_expected, n_threads):
    """
    Test Doolittle's LU factorization using the parallel method with different thread counts.
    """
    # Set the number of threads for Numba
    numba.set_num_threads(n_threads)

    # Perform LU factorization in parallel
    L, U = DoolittleFactorization.parallel(A)

    # Reset Numba threads to default after the test to avoid affecting other tests
    numba.set_num_threads(numba.config.NUMBA_DEFAULT_NUM_THREADS)

    # Assertions
    np.testing.assert_allclose(
        L,
        L_expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Parallel L matrix mismatch with {n_threads} threads.",
    )
    np.testing.assert_allclose(
        U,
        U_expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Parallel U matrix mismatch with {n_threads} threads.",
    )
    assert (
        check_frobenius_norm_lu(A, L, U) == 0
    ), f"Parallel LU factorization is incorrect with {n_threads} threads."
