import equity_risk_model
import numpy
import pytest


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (numpy.diag([0.1, 0.2, 0.3]), True),  # Positive eigenvalues
        (numpy.diag([-0.1, -0.2, -0.3]), False),  # Negative eigenvalues
        (numpy.diag([[9, 7], [6, 14]]), False),  # Non-symmetric
    ],
)
def test_is_positive_semidefinite(matrix, expected):

    assert (
        equity_risk_model.correlation.is_positive_semidefinite(matrix)
        == expected
    )
