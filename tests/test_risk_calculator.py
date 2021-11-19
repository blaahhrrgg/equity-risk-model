from dataclasses import dataclass

import equity_risk_model
import numpy
import pytest


@pytest.fixture
def factor_model():

    universe = numpy.array(["A", "B", "C", "D", "E"])
    factors = numpy.array(["foo", "bar", "baz"])

    factor_loadings = numpy.array(
        [
            [0.2, 0.3, -0.1, -0.2, 0.45],
            [0.01, -0.2, -0.23, -0.01, 0.4],
            [0.1, 0.05, 0.23, 0.15, -0.1],
        ]
    )

    covariance_factor = numpy.array(
        [[0.3, 0.05, 0.01], [0.05, 0.15, -0.10], [0.01, -0.10, 0.2]]
    )

    covariance_specific = numpy.diag([0.05, 0.04, 0.10, 0.02, 0.09])

    factor_model = equity_risk_model.model.FactorRiskModel(
        universe,
        factors,
        factor_loadings,
        covariance_factor,
        covariance_specific,
    )

    return equity_risk_model.calculator.RiskCalculator(factor_model)


@pytest.mark.parametrize(
    "weights, expected",
    [(numpy.ones(5) * 0.2, 0.13712548997177731), (numpy.zeros(5), 0.0)],
)
def test_total_risk(weights, expected, factor_model):

    numpy.testing.assert_almost_equal(
        factor_model.total_risk(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [(numpy.ones(5) * 0.2, 0.08248272546418432), (numpy.zeros(5), 0.0)],
)
def test_total_factor_risk(weights, expected, factor_model):

    numpy.testing.assert_almost_equal(
        factor_model.total_factor_risk(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [(numpy.ones(5) * 0.2, 0.10954451150103323), (numpy.zeros(5), 0.0)],
)
def test_total_specific_risk(weights, expected, factor_model):

    numpy.testing.assert_almost_equal(
        factor_model.total_specific_risk(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (
            numpy.ones(5) * 0.2,
            numpy.array([0.0712039, 0.0023238, 0.0384604]),
        ),
        (numpy.zeros(5), numpy.zeros(3)),
    ],
)
def test_factor_risks(weights, expected, factor_model):

    numpy.testing.assert_almost_equal(
        factor_model.factor_risks(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (numpy.ones(5) * 0.2, 0.015773395322504297),
        (numpy.zeros(5), 0.0),
    ],
)
def test_factor_covariance(weights, expected, factor_model):

    numpy.testing.assert_almost_equal(
        factor_model.factor_risk_covariance(weights), expected
    )


def test_marginal_contributions_to_total_risk(factor_model):

    weights = numpy.array([0.2] * 5)

    expected = numpy.array(
        [0.028867, 0.0312458, 0.0308141, -0.0014833, 0.0476819]
    )

    numpy.testing.assert_almost_equal(
        factor_model.marginal_contribution_to_total_risk(weights),
        expected,
    )


def test_marginal_contributions_to_total_factor_risk(factor_model):

    weights = numpy.array([0.2] * 5)

    expected = numpy.array(
        [0.0237432, 0.0325474, 0.0027327, -0.012165, 0.0356244]
    )

    numpy.testing.assert_almost_equal(
        factor_model.marginal_contribution_to_total_factor_risk(weights),
        expected,
    )


def test_marginal_contributions_to_total_specific_risk(factor_model):

    weights = numpy.array([0.2] * 5)

    expected = numpy.array(
        [0.0182574, 0.0146059, 0.0365148, 0.007303, 0.0328634]
    )

    numpy.testing.assert_almost_equal(
        factor_model.marginal_contribution_to_total_specific_risk(weights),
        expected,
    )


def test_marginal_contributions_to_factor_risks(factor_model):

    weights = numpy.array([0.2] * 5)

    numpy.testing.assert_almost_equal(
        numpy.sum(
            factor_model.marginal_contributions_to_factor_risks(weights), axis=1
        ),
        factor_model.factor_risks(weights),
    )


def test_effective_number_of_correlated_bets(factor_model):

    weights = numpy.array([0.2] * 5)

    numpy.testing.assert_almost_equal(
        factor_model.effective_number_of_correlated_bets(weights),
        3.7346305378867153,
    )


def test_effective_number_of_uncorrelated_bets(factor_model):

    weights = numpy.array([0.2] * 5)

    numpy.testing.assert_almost_equal(
        factor_model.effective_number_of_uncorrelated_bets(weights),
        3.9823008849557513,
    )
