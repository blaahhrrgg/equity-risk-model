import equity_risk_model
import numpy
import pandas

from pytest_cases import fixture


@fixture(scope="module")
def factor_model():

    universe = numpy.array(["A", "B", "C", "D", "E"])
    factors = numpy.array(["foo", "bar", "baz"])

    factor_loadings = pandas.DataFrame(
        data=numpy.array(
            [
                [0.2, 0.3, -0.1, -0.2, 0.45],
                [0.01, -0.2, -0.23, -0.01, 0.4],
                [0.1, 0.05, 0.23, 0.15, -0.1],
            ]
        ),
        columns=universe,
        index=factors,
    )

    covariance_factor = pandas.DataFrame(
        data=numpy.array(
            [[0.3, 0.05, 0.01], [0.05, 0.15, -0.10], [0.01, -0.10, 0.2]]
        ),
        columns=factors,
        index=factors,
    )

    covariance_specific = pandas.DataFrame(
        data=numpy.diag([0.05, 0.04, 0.10, 0.02, 0.09]),
        index=universe,
        columns=universe,
    )

    return equity_risk_model.model.FactorRiskModel(
        universe,
        factors,
        factor_loadings,
        covariance_factor,
        covariance_specific,
    )


@fixture(scope="module")
def factor_model_with_groups():

    universe = numpy.array(["A", "B", "C", "D", "E"])
    factors = numpy.array(["foo", "bar", "baz"])

    factor_loadings = pandas.DataFrame(
        data=[
            [0.2, 0.3, -0.1, -0.2, 0.45],
            [0.01, -0.2, -0.23, -0.01, 0.4],
            [0.1, 0.05, 0.23, 0.15, -0.1],
        ],
        index=factors,
        columns=universe,
    )

    covariance_factor = pandas.DataFrame(
        data=[[0.3, 0.05, 0.01], [0.05, 0.15, -0.10], [0.01, -0.10, 0.2]],
        index=factors,
        columns=factors,
    )

    covariance_specific = pandas.DataFrame(
        data=numpy.diag([0.05, 0.04, 0.10, 0.02, 0.09]),
        index=universe,
        columns=universe,
    )

    factor_groups = {"Alpha": ["foo", "bar"], "Beta": ["baz"]}

    return equity_risk_model.model.FactorRiskModel(
        universe,
        factors,
        factor_loadings,
        covariance_factor,
        covariance_specific,
        factor_group_mapping=factor_groups,
    )


@fixture(scope="module")
def risk_calculator(factor_model):

    return equity_risk_model.risk.RiskCalculator(factor_model)


@fixture(scope="module")
def risk_calculator_with_factor_groups(factor_model_with_groups):

    return equity_risk_model.risk.RiskCalculator(factor_model_with_groups)


@fixture(scope="module")
def concentration_calculator(risk_calculator):

    return equity_risk_model.concentration.ConcentrationCalculator(
        risk_calculator=risk_calculator
    )
