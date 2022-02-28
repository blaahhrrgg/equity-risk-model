import equity_risk_model
import numpy
import pandas
import pytest


@pytest.fixture(scope="module")
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

    return equity_risk_model.model.FactorRiskModel(
        universe,
        factors,
        factor_loadings,
        covariance_factor,
        covariance_specific,
    )


@pytest.fixture(scope="module")
def risk_calculator():

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

    return equity_risk_model.risk.RiskCalculator(factor_model)


@pytest.fixture(scope="module")
def concentration_calculator(risk_calculator):

    return equity_risk_model.concentration.ConcentrationCalculator(
        risk_calculator=risk_calculator
    )

