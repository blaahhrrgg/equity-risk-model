import numpy
import pandas
import pytest


@pytest.mark.parametrize(
    "weights, expected_total, expected_factor, expected_specific",
    [
        (
            [0.2] * 5,
            0.13712548997177731,
            0.08248272546418432,
            0.10954451150103323,
        ),
        ([0.0] * 5, 0.0, 0.0, 0.0),
    ],
)
def test_total_risk(
    weights,
    expected_total,
    expected_factor,
    expected_specific,
    risk_calculator,
):

    p = pandas.Series(
        data=weights, index=risk_calculator.factor_model.universe
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.total_risk(p), expected_total
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.total_factor_risk(p), expected_factor
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.total_specific_risk(p), expected_specific
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (
            [0.2] * 5,
            numpy.array([0.0712039, -0.0023238, 0.0384604]),
        ),
        ([0.0] * 5, numpy.zeros(3)),
    ],
)
def test_factor_risks(weights, expected, risk_calculator):

    p = pandas.Series(
        data=weights, index=risk_calculator.factor_model.universe
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.factor_risks(p), expected
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([0.2] * 5, 0.015773395322504297),
        ([0.0] * 5, 0.0),
    ],
)
def test_factor_covariance(weights, expected, risk_calculator):

    p = pandas.Series(
        data=weights, index=risk_calculator.factor_model.universe
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.factor_risk_covariance(p), expected
    )


def test_marginal_contributions_to_total_risk(risk_calculator):

    expected = numpy.array(
        [0.028867, 0.0312458, 0.0308141, -0.0014833, 0.0476819]
    )

    p = pandas.Series(
        data=[0.2] * 5, index=risk_calculator.factor_model.universe
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.marginal_contribution_to_total_risk(p),
        expected,
    )


def test_marginal_contributions_to_total_factor_risk(risk_calculator):

    p = pandas.Series(
        data=[0.2] * 5, index=risk_calculator.factor_model.universe
    )

    expected = numpy.array(
        [0.0237432, 0.0325474, 0.0027327, -0.012165, 0.0356244]
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.marginal_contribution_to_total_factor_risk(p),
        expected,
    )


def test_marginal_contributions_to_total_specific_risk(risk_calculator):

    p = pandas.Series(
        data=[0.2] * 5, index=risk_calculator.factor_model.universe
    )
    expected = numpy.array(
        [0.0182574, 0.0146059, 0.0365148, 0.007303, 0.0328634]
    )

    numpy.testing.assert_almost_equal(
        risk_calculator.marginal_contribution_to_total_specific_risk(p),
        expected,
    )


def test_marginal_contributions_to_factor_risks(risk_calculator):

    p = pandas.Series(
        data=[0.2] * 5, index=risk_calculator.factor_model.universe
    )

    numpy.testing.assert_almost_equal(
        numpy.sum(
            risk_calculator.marginal_contributions_to_factor_risks(p),
            axis=0,
        ).values,
        risk_calculator.factor_risks(p).values,
    )


def test_factor_group_risk(risk_calculator_with_factor_groups):

    p = pandas.Series(
        data=[0.2] * 5,
        index=risk_calculator_with_factor_groups.factor_model.universe,
    )

    factor_group_risk = risk_calculator_with_factor_groups.factor_group_risks(
        p
    )

    # Check structure
    assert isinstance(factor_group_risk, dict)
    assert set(factor_group_risk.keys()) == set(
        risk_calculator_with_factor_groups.factor_model.factor_groups
    )

    # Check covariance
    factor_group_covariance = (
        risk_calculator_with_factor_groups.factor_group_covariance(p)
    )

    numpy.testing.assert_almost_equal(factor_group_covariance, 0.0180776)


def test_factor_group_factor_risk(risk_calculator_with_factor_groups):

    p = pandas.Series(
        data=[0.2] * 5,
        index=risk_calculator_with_factor_groups.factor_model.universe,
    )

    factor_risks = risk_calculator_with_factor_groups.factor_risks(p)

    assert isinstance(factor_risks, pandas.Series)

    # Check index has correct structure
    assert isinstance(factor_risks.index, pandas.MultiIndex)

    assert set(factor_risks.index.get_level_values(0)) == set(
        risk_calculator_with_factor_groups.factor_model.factor_groups
    )
    assert set(factor_risks.index.get_level_values(1)) == set(
        risk_calculator_with_factor_groups.factor_model.factors
    )
