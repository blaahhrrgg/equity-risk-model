import equity_risk_model
import pandas
import numpy
import pytest


def test_abstract_methods():
    with pytest.raises(NotImplementedError):
        equity_risk_model.optimiser.PortfolioOptimiser._objective_function(
            None
        )

    with pytest.raises(NotImplementedError):
        equity_risk_model.optimiser.PortfolioOptimiser._constraints(None)


def test_min_variance(factor_model):

    opt = equity_risk_model.optimiser.MinimumVariance(factor_model)
    opt.solve()

    numpy.testing.assert_almost_equal(
        opt.x.value,
        numpy.array([0.1502093, 0.1485704, 0.0598607, 0.5079278, 0.1334318]),
    )


def test_max_sharpe(factor_model):

    expected_returns = numpy.array([0.2, 0.1, 0.05, 0.1, 0.2])

    opt = equity_risk_model.optimiser.MaximumSharpe(
        factor_model, expected_returns
    )
    opt.solve()

    numpy.testing.assert_almost_equal(
        opt.x.value,
        numpy.array([0.82123741, 0, 0, 0, 0.17876259]),
    )


def test_proportional_factor_neutral(factor_model):

    expected_returns = numpy.array([0.2, 0.1, 0.05, 0.1, 0.2])

    opt = equity_risk_model.optimiser.ProportionalFactorNeutral(
        factor_model, expected_returns
    )
    opt.solve()

    # Check weights
    numpy.testing.assert_almost_equal(
        opt.x.value,
        numpy.array([0.0280274, 0.016464, -0.0464042, 0.0347927, -0.0182813]),
    )

    # Validate factor risk is zero
    w_opt = pandas.Series(data=opt.x.value, index=factor_model.universe)

    factor_risks = equity_risk_model.risk.RiskCalculator(
        factor_model
    ).factor_risks(w_opt)

    numpy.testing.assert_almost_equal(
        factor_risks, numpy.zeros((factor_model.n_factors))
    )


def test_internally_hedged_factor_neutral(factor_model):

    initial_weights = pandas.Series(
        data=[-0.2, -0.2, 0.2, -0.2, 0.2], index=factor_model.universe
    )

    opt = equity_risk_model.optimiser.InternallyHedgedFactorNeutral(
        factor_model, initial_weights
    )
    opt.solve()

    # Check weights
    numpy.testing.assert_almost_equal(
        opt.x.value,
        numpy.array([0.0144357, 0.1267867, 0.0577181, 0.0275523, -0.0880908]),
    )

    # Check sign of weights has not changed
    opt_weights = pandas.Series(opt.x.value, index=factor_model.universe)

    numpy.testing.assert_array_equal(
        numpy.sign(initial_weights + opt_weights), numpy.sign(initial_weights)
    )

    # Validate factor risk is zero
    factor_risks = equity_risk_model.risk.RiskCalculator(
        factor_model
    ).factor_risks(opt.x.value + initial_weights)

    numpy.testing.assert_almost_equal(
        factor_risks, numpy.zeros((factor_model.n_factors))
    )


def test_internally_hedged_factor_tolerant(factor_model):

    initial_weights = pandas.Series(
        data=[0.2, 0.2, 0.2, 0.2, 0.2], index=factor_model.universe
    )

    factor_risk_upper_bounds = numpy.array([0.01, 0.01, 0.01])

    opt = equity_risk_model.optimiser.InternallyHedgedFactorTolerant(
        factor_model, initial_weights, factor_risk_upper_bounds
    )
    opt.solve()

    # Check weights
    numpy.testing.assert_almost_equal(
        opt.x.value,
        numpy.array(
            [-0.2030572, -0.2315047, -0.0851033, -0.1368858, -0.0834827]
        ),
    )

    # Validate factor risks are less than specified upper bound
    factor_risks = equity_risk_model.risk.RiskCalculator(
        factor_model
    ).factor_risks(opt.x.value + initial_weights)

    numpy.testing.assert_array_less(
        factor_risks - factor_risk_upper_bounds,
        numpy.ones(factor_model.n_factors) * 1e-9,
    )
