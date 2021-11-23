import abc

import cvxpy
import numpy


class BaseOptimiser(abc.ABC):
    """Abstract base class from which all optimiser classes will inherit.

    Optimisation is performed using the package cvxpy and specifying a convex
    optimisation problem in the form,

    .. math::

       \begin{array}{ll}
       \mbox{minimize}   & (1/2)x^TPx + q^Tx\\
       \mbox{subject to} & Gx \leq h \\
                         & Ax = b.
       \end{array}

    See Also
    --------
    .. https://www.cvxpy.org/index.html
    """

    def __init__(self, factor_model, verbose=False):
        self.factor_model = factor_model
        self.x = cvxpy.Variable(self.factor_model.n_assets)
        self.verbose = verbose
        self.problem = None

    @abc.abstractmethod
    def objective_function(self):
        """Objective function of the optimisation problem"""
        raise NotImplementedError

    @abc.abstractmethod
    def constraints(self):
        """Constraints of the optimisation problem"""
        raise NotImplementedError

    def setup_problem(self):
        """Create the convex optimisation problem"""
        self.problem = cvxpy.Problem(
            self.objective_function(), self.constraints()
        )

    def solve(self):
        """Solve the convex optimisation problem"""
        if not self.problem:
            self.setup_problem()
        self.problem.solve(verbose=self.verbose)
        return self


class MinimumVariance(BaseOptimiser):
    """Minimum Variance Long-Only Portfolio"""

    def __init__(self, factor_model, verbose=False):
        super().__init__(factor_model, verbose)

    def objective_function(self):
        return cvxpy.Minimize(
            # Total Variance
            0.5
            * cvxpy.QuadForm(self.x, self.factor_model.covariance_total)
        )

    def constraints(self):
        return [
            # Sum of weights equals one
            numpy.ones((self.factor_model.n_assets)).T @ self.x == 1,
            # All weights are positive (long only positions)
            self.x >= 0,
        ]


class MaximumSharpe(BaseOptimiser):
    """Maximum Sharpe Portfolio"""

    def __init__(
        self, factor_model, expected_returns, gamma=1.0, verbose=False
    ):
        super().__init__(factor_model, verbose)
        self.gamma = gamma
        self.expected_returns = expected_returns

    def objective_function(self):
        return cvxpy.Minimize(
            # Total Variance
            0.5 * cvxpy.QuadForm(self.x, self.factor_model.covariance_total)
            - self.gamma * self.expected_returns @ self.x
        )

    def constraints(self):
        return [
            # Sum of weights equals one
            numpy.ones((self.factor_model.n_assets)).T @ self.x == 1,
            # All weights are positive (long only positions)
            self.x >= 0,
        ]


class ProportionalFactorNeutral(BaseOptimiser):
    """Proportional Factor Neutral Portfolio"""

    def __init__(self, factor_model, expected_returns, verbose=False):
        super().__init__(factor_model, verbose)
        self.expected_returns = expected_returns

    def objective_function(self):
        return cvxpy.Minimize(
            # Distance between expected returns and portfolio weights
            cvxpy.sum_squares((self.x - self.expected_returns))
        )

    def constraints(self):
        return [
            # Factor loadings of the portfolio are zero
            self.factor_model.loadings @ self.x
            == numpy.zeros((self.factor_model.n_factors))
        ]


class InternallyHedgedFactorNeutral(BaseOptimiser):
    """Internally Hedged Factor Neutral Portfolio"""

    def __init__(self, factor_model, initial_weights, verbose=False):
        super().__init__(factor_model, verbose)
        self.initial_weights = initial_weights

    def objective_function(self):
        return cvxpy.Minimize(
            cvxpy.QuadForm(self.x, self.factor_model.covariance_specific)
        )

    def constraints(self):
        return [
            # Factor loadings of the portfolio are zero
            self.factor_model.loadings @ (self.x + self.initial_weights)
            == numpy.zeros((self.factor_model.n_factors))
        ]


class InternallyHedgedFactorTolerant(BaseOptimiser):
    """Internally Hedged Factor Tolerant Portfolio"""

    def __init__(
        self,
        factor_model,
        initial_weights,
        factor_risk_upper_bounds,
        verbose=False,
    ):
        super().__init__(factor_model, verbose)
        self.initial_weights = initial_weights
        self.factor_risk_upper_bounds = factor_risk_upper_bounds

    def objective_function(self):

        cov_factor = (
            self.factor_model.loadings.T
            @ self.factor_model.covariance_factor
            @ self.factor_model.loadings
        )

        return cvxpy.Minimize(
            # Specific risk of the hedge portfolio
            cvxpy.QuadForm(self.x, self.factor_model.covariance_specific)
            # Factor risk of the end portfolio
            + cvxpy.QuadForm(self.x + self.initial_weights, cov_factor)
        )

    def constraints(self):

        A = numpy.multiply(
            self.factor_model.loadings.T,
            numpy.sqrt(numpy.diag(self.factor_model.covariance_factor)),
        ).T

        return [
            A @ (self.x + self.initial_weights)
            <= self.factor_risk_upper_bounds,
            A @ -(self.x + self.initial_weights)
            <= self.factor_risk_upper_bounds,
        ]
