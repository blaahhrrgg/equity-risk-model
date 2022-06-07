import numpy
import pandas
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
def test_enc(weights, expected, concentration_calculator):

    numpy.testing.assert_almost_equal(
        concentration_calculator.enc(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (weights_equal, 5),
    ],
)
def test_entropy(weights, expected, concentration_calculator):

    numpy.testing.assert_almost_equal(
        concentration_calculator.entropy(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [(weights_equal, 3.7346305378867153), (weights_concentrated, 1)],
)
def test_effective_number_of_correlated_bets(
    weights, expected, concentration_calculator
):

    p = pandas.Series(
        data=weights,
        index=concentration_calculator.risk_calculator.factor_model.universe,
    )

    numpy.testing.assert_almost_equal(
        concentration_calculator.number_of_correlated_bets(p),
        expected,
    )


@pytest.mark.parametrize(
    "weights, expected",
    [(weights_equal, 3.9823008849557513), (weights_concentrated, 1)],
)
def test_effective_number_of_uncorrelated_bets(
    weights, expected, concentration_calculator
):

    p = pandas.Series(
        data=weights,
        index=concentration_calculator.risk_calculator.factor_model.universe,
    )

    numpy.testing.assert_almost_equal(
        concentration_calculator.number_of_uncorrelated_bets(p),
        expected,
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (weights_equal, 2),
        (weights_concentrated, 1),
    ],
)
def test_min_assets(weights, expected, concentration_calculator):

    p = pandas.Series(
        data=weights,
        index=concentration_calculator.risk_calculator.factor_model.universe,
    )

    numpy.testing.assert_almost_equal(
        concentration_calculator.min_assets_for_mcsr_threshold(p),
        expected,
    )


@pytest.mark.parametrize(
    "weights",
    [(weights_equal)],
)
def test_summarise(weights, concentration_calculator):

    p = pandas.Series(
        data=weights,
        index=concentration_calculator.risk_calculator.factor_model.universe,
    )

    out = concentration_calculator.summarise_portfolio(p)

    assert isinstance(out, dict)
    assert out["NAssets"] == 5
