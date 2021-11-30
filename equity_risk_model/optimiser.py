from dataclasses import dataclass
from typing import List, Protocol, Union

import cvxpy
import numpy
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Maximize, Minimize


@dataclass
class FactorRiskModel(Protocol):
    """Factor Risk Model"""

    universe: numpy.array
    factors: numpy.array
    loadings: numpy.array
    covariance_factor: numpy.array
    covariance_specific: numpy.array
    covariance_total: numpy.array


class PortfolioOptimiser(cvxpy.Problem):
    """Wrapper for cvxpy.Problem class to facilitate problem formulation.

    The problem is specified in terms of a quadratic objective function with
    affine equality and inequality constraints.

    The standard form is the following:

    .. math::

       \begin{array}{ll}
       \mbox{minimize}   & (1/2)x^TPx + q^Tx\\
       \mbox{subject to} & Gx \leq h \\
                         & Ax = b.
       \end{array}

    The matrices P, G, and A as well as vectors q, h, and b are to be specified
    whereas the variable x is the optimisation variable.
    
    The package `cvxpy` is used to solve the above problem.

    See Also
    --------
    .. https://www.cvxpy.org/index.html
    """

    def __init__(self, factor_model: FactorRiskModel):
        self.factor_model = factor_model
        self.x = cvxpy.Variable(self.factor_model.n_assets)
        super().__init__(
            objective=self._objective_function(),
            constraints=self._constraints(),
        )

    def _objective_function(self) -> Union[Minimize, Maximize]:
        """Define objective function of the optimisation problem"""
        raise NotImplementedError

    def _constraints(self) -> List[Constraint]:
        """Define constraints of the optimisation problem"""
        raise NotImplementedError


class MinimumVariance(PortfolioOptimiser):
    """Minimum Variance Long-Only Portfolio

    Find the portfolio weights which minimise the portfolio's variance subject
    to positive weights that sum to unity.
    """

    def __init__(self, factor_model: FactorRiskModel):
        super().__init__(factor_model)

    def _objective_function(self) -> Minimize:
        return cvxpy.Minimize(
            # Total Variance
            0.5
            * cvxpy.QuadForm(self.x, self.factor_model.covariance_total)
        )

    def _constraints(self) -> List[Constraint]:
        return [
            # Sum of weights equals unity
            numpy.ones((self.factor_model.n_assets)).T @ self.x == 1,
            # All weights are positive (long only positions)
            self.x >= 0,
        ]


class MaximumSharpe(PortfolioOptimiser):
    """Maximum Sharpe Portfolio

    Find the portfolio weights which maximise the portfolio's Sharpe ratio
    subject to positive weights that sum to unity.
    """

    def __init__(
        self,
        factor_model: FactorRiskModel,
        expected_returns: numpy.ndarray,
        gamma: float = 1.0,
    ):
        self.gamma = gamma
        self.expected_returns = expected_returns
        super().__init__(factor_model)

    def _objective_function(self) -> Minimize:
        return cvxpy.Minimize(
            # Total Variance
            0.5 * cvxpy.QuadForm(self.x, self.factor_model.covariance_total)
            - self.gamma * self.expected_returns @ self.x
        )

    def _constraints(self) -> List[Constraint]:
        return [
            # Sum of weights equals unity
            numpy.ones((self.factor_model.n_assets)).T @ self.x == 1,
            # All weights are positive (long only positions)
            self.x >= 0,
        ]


class ProportionalFactorNeutral(PortfolioOptimiser):
    """Proportional Factor Neutral Portfolio

    Find the portfolio weights proportional to the expected returns of the
    assets subject to the portfolio being factor neutral.
    """

    def __init__(
        self, factor_model: FactorRiskModel, expected_returns: numpy.ndarray
    ):
        self.expected_returns = expected_returns
        super().__init__(factor_model)

    def _objective_function(self) -> Minimize:
        return cvxpy.Minimize(
            # Distance between expected returns and portfolio weights
            cvxpy.sum_squares((self.x - self.expected_returns))
        )

    def _constraints(self) -> List[Constraint]:
        return [
            # Factor loadings of the portfolio are zero
            self.factor_model.loadings @ self.x
            == numpy.zeros((self.factor_model.n_factors))
        ]


class InternallyHedgedFactorNeutral(PortfolioOptimiser):
    """Internally Hedged Factor Neutral Portfolio

    Finds the (internal) hedge portfolio that results in a factor neutral
    portfolio without changing the sign of any weight.
    """

    def __init__(
        self, factor_model: FactorRiskModel, initial_weights: numpy.ndarray
    ):
        self.initial_weights = initial_weights
        super().__init__(factor_model)

    def _objective_function(self):
        return cvxpy.Minimize(
            # Specific Variance
            0.5
            * cvxpy.QuadForm(self.x, self.factor_model.covariance_specific)
        )

    def _constraints(self) -> List[Constraint]:
        return [
            # Factor loadings of the portfolio are zero
            self.factor_model.loadings @ (self.x + self.initial_weights)
            == numpy.zeros((self.factor_model.n_factors))
            # Sign of weights does not change
            # TODO!
        ]


class InternallyHedgedFactorTolerant(PortfolioOptimiser):
    """Internally Hedged Factor Tolerant Portfolio

    Finds the internal hedge portfolio that results in a factor tolerant
    portfolio (factor risk within specified bounds).
    """

    def __init__(
        self,
        factor_model: FactorRiskModel,
        initial_weights: numpy.ndarray,
        factor_risk_upper_bounds: numpy.ndarray,
    ):
        self.initial_weights = initial_weights
        self.factor_risk_upper_bounds = factor_risk_upper_bounds
        super().__init__(factor_model)

    def _objective_function(self) -> Minimize:

        cov_factor = (
            self.factor_model.loadings.T
            @ self.factor_model.covariance_factor
            @ self.factor_model.loadings
        )

        return cvxpy.Minimize(
            # Specific variance of the hedge portfolio
            cvxpy.QuadForm(self.x, self.factor_model.covariance_specific)
            # Factor variance of the end portfolio
            + cvxpy.QuadForm(self.x + self.initial_weights, cov_factor)
        )

    def _constraints(self) -> List[Constraint]:

        A = numpy.multiply(
            self.factor_model.loadings.T,
            numpy.sqrt(numpy.diag(self.factor_model.covariance_factor)),
        ).T

        return [
            # Final portfolio factor risk is less than or equal to upper bound
            A @ (self.x + self.initial_weights)
            <= self.factor_risk_upper_bounds,
            A @ -(self.x + self.initial_weights)
            <= self.factor_risk_upper_bounds,
        ]
