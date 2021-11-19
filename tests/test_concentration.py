import equity_risk_model
import numpy
import pytest

weights_concentrated = numpy.array([1, 0, 0, 0, 0])
weights_equal = numpy.array([0.2, 0.2, 0.2, 0.2, 0.2])
weights_longshort1 = numpy.array([-0.5, 0.5, 0.5, 0.5])
weights_longshort2 = numpy.array([-1, 2 / 3.0, 2 / 3.0, 2 / 3.0])


@pytest.mark.parametrize(
    "weights, expected",
    [
        (weights_concentrated, 1),
        (weights_equal, 5),
        (weights_longshort1, 1),
        (weights_longshort2, 3 / 7.0),
    ],
)
def test_enc(weights, expected):

    numpy.testing.assert_almost_equal(
        equity_risk_model.concentration.enc(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (weights_equal, 5),
    ],
)
def test_entropy(weights, expected):

    numpy.testing.assert_almost_equal(
        equity_risk_model.concentration.entropy(weights), expected
    )
