import numpy
import pandas
import pytest

import equity_risk_model


@pytest.fixture(scope="class")
def concentration_tearsheet(concentration_calculator):
    return equity_risk_model.tearsheet.ConcentrationTearsheet(
        concentration_calculator=concentration_calculator
    )


@pytest.fixture(scope="class")
def factor_risk_summary_tearsheet(risk_calculator_with_factor_groups):
    return equity_risk_model.tearsheet.FactorRiskSummaryTearsheet(
        risk_calculator=risk_calculator_with_factor_groups
    )


@pytest.fixture(scope="class")
def factor_group_risk_summary_tearsheet(risk_calculator_with_factor_groups):
    return equity_risk_model.tearsheet.FactorGroupRiskTearsheet(
        risk_calculator=risk_calculator_with_factor_groups
    )


@pytest.fixture(scope="class")
def factor_risk_tearsheet(risk_calculator_with_factor_groups):
    return equity_risk_model.tearsheet.FactorRiskTearsheet(
        risk_calculator=risk_calculator_with_factor_groups
    )


@pytest.fixture(scope="class")
def portfolio_weights(factor_model):
    return {
        "EqualWeights": pandas.Series(
            data=numpy.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            index=factor_model.universe,
        ),
        "ConcentratedPortfolio": pandas.Series(
            data=numpy.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            index=factor_model.universe,
        ),
        "LongShortPortfolio": pandas.Series(
            data=numpy.array([0.4, 0.4, 0.2, -0.6, -0.4]),
            index=factor_model.universe,
        ),
    }


def test_base_class():

    # Check abstract method raises exception in base class
    with pytest.raises(NotImplementedError):
        equity_risk_model.tearsheet.BaseTearsheet().create_portfolio_panel(
            None
        )


def test_concentration_tearsheet(portfolio_weights, concentration_tearsheet):

    tearsheet = concentration_tearsheet.create_tearsheet(portfolio_weights)

    # Check format of output
    assert isinstance(tearsheet, pandas.DataFrame)
    assert len(tearsheet.columns) == len(portfolio_weights.keys())


def test_factor_risk_summary_tearsheet(
    portfolio_weights, factor_risk_summary_tearsheet
):

    tearsheet = factor_risk_summary_tearsheet.create_tearsheet(
        portfolio_weights
    )

    # Check format of output
    assert isinstance(tearsheet, pandas.DataFrame)
    assert len(tearsheet.columns) == len(portfolio_weights.keys())


def test_factor_group_risk_tearsheet(
    portfolio_weights, factor_group_risk_summary_tearsheet
):

    tearsheet = factor_group_risk_summary_tearsheet.create_tearsheet(
        portfolio_weights
    )

    # Check format of output
    assert isinstance(tearsheet, pandas.DataFrame)
    assert len(tearsheet.columns) == len(portfolio_weights.keys())


def test_factor_risk_tearsheet(portfolio_weights, factor_risk_tearsheet):

    tearsheet = factor_risk_tearsheet.create_tearsheet(portfolio_weights)

    # Check format of output
    assert isinstance(tearsheet, pandas.DataFrame)
    assert len(tearsheet.columns) == len(portfolio_weights.keys())
