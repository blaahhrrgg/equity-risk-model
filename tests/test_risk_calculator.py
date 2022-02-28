import numpy
import pytest


@pytest.mark.parametrize(
    "weights, expected",
    [(numpy.ones(5) * 0.2, 0.13712548997177731), (numpy.zeros(5), 0.0)],
)
def test_total_risk(weights, expected, risk_calculator):

    numpy.testing.assert_almost_equal(
        risk_calculator.total_risk(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [(numpy.ones(5) * 0.2, 0.08248272546418432), (numpy.zeros(5), 0.0)],
)
def test_total_factor_risk(weights, expected, risk_calculator):

    numpy.testing.assert_almost_equal(
        risk_calculator.total_factor_risk(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [(numpy.ones(5) * 0.2, 0.10954451150103323), (numpy.zeros(5), 0.0)],
)
def test_total_specific_risk(weights, expected, risk_calculator):

    numpy.testing.assert_almost_equal(
        risk_calculator.total_specific_risk(weights), expected
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
def test_factor_risks(weights, expected, risk_calculator):

    numpy.testing.assert_almost_equal(
        risk_calculator.factor_risks(weights), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (numpy.ones(5) * 0.2, 0.015773395322504297),
        (numpy.zeros(5), 0.0),
    ],
)
def test_factor_covariance(weights, expected, risk_calculator):

    numpy.testing.assert_almost_equal(
        risk_calculator.factor_risk_covariance(weights), expected
    )


def test_marginal_contributions_to_total_risk(risk_calculator):

    weights = numpy.array([0.2] * 5)

    expected = numpy.array(
        [0.028867, 0.0312458, 0.0308141, -0.0014833, 0.0476819]
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.marginal_contribution_to_total_risk(weights),
        expected,
    )


def test_marginal_contributions_to_total_factor_risk(risk_calculator):

    weights = numpy.array([0.2] * 5)

    expected = numpy.array(
        [0.0237432, 0.0325474, 0.0027327, -0.012165, 0.0356244]
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.marginal_contribution_to_total_factor_risk(weights),
        expected,
    )


def test_marginal_contributions_to_total_specific_risk(risk_calculator):

    weights = numpy.array([0.2] * 5)

    expected = numpy.array(
        [0.0182574, 0.0146059, 0.0365148, 0.007303, 0.0328634]
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.marginal_contribution_to_total_specific_risk(weights),
        expected,
    )


def test_marginal_contributions_to_factor_risks(risk_calculator):

    weights = numpy.array([0.2] * 5)

    numpy.testing.assert_almost_equal(
        numpy.sum(
            risk_calculator.marginal_contributions_to_factor_risks(weights),
            axis=0,
        ).values,
        risk_calculator.factor_risks(weights).values,
    )
