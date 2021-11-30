import abc

import cvxpy
import numpy


class BaseOptimiser(abc.ABC):
    """Abstract base class from which all optimiser classes will inherit.

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

    def __init__(self, factor_model):
        self.factor_model = factor_model
        self.x = cvxpy.Variable(self.factor_model.n_assets)
        self.problem = None

    @abc.abstractmethod
    def objective_function(self):
        """Define objective function of the optimisation problem"""
        raise NotImplementedError

    @abc.abstractmethod
    def constraints(self):
        """Define constraints of the optimisation problem"""
        raise NotImplementedError

    def setup_problem(self):
        """Create the convex optimisation problem"""
        self.problem = cvxpy.Problem(
            self.objective_function(), self.constraints()
        )

    def solve(self, *args, **kwargs):
        """Solve the convex optimisation problem"""
        if not self.problem:
            self.setup_problem()
        self.problem.solve(*args, **kwargs)
        return self


class MinimumVariance(BaseOptimiser):
    """Minimum Variance Long-Only Portfolio"""

    def __init__(self, factor_model):
        super().__init__(factor_model)

    def objective_function(self):
        return cvxpy.Minimize(
            # Total Variance
            0.5
            * cvxpy.QuadForm(self.x, self.factor_model.covariance_total)
        )

    def constraints(self):
        return [
            # Sum of weights equals unity
            numpy.ones((self.factor_model.n_assets)).T @ self.x == 1,
            # All weights are positive (long only positions)
            self.x >= 0,
        ]


class MaximumSharpe(BaseOptimiser):
    """Maximum Sharpe Portfolio"""

    def __init__(self, factor_model, expected_returns, gamma=1.0):
        super().__init__(factor_model)
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
            # Sum of weights equals unity
            numpy.ones((self.factor_model.n_assets)).T @ self.x == 1,
            # All weights are positive (long only positions)
            self.x >= 0,
        ]


class ProportionalFactorNeutral(BaseOptimiser):
    """Proportional Factor Neutral Portfolio"""

    def __init__(self, factor_model, expected_returns):
        super().__init__(factor_model)
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

    def __init__(self, factor_model, initial_weights):
        super().__init__(factor_model)
        self.initial_weights = initial_weights

    def objective_function(self):
        return cvxpy.Minimize(
            # Specific Variance
            0.5
            * cvxpy.QuadForm(self.x, self.factor_model.covariance_specific)
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
    ):
        super().__init__(factor_model)
        self.initial_weights = initial_weights
        self.factor_risk_upper_bounds = factor_risk_upper_bounds

    def objective_function(self):

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

    def constraints(self):

        A = numpy.multiply(
            self.factor_model.loadings.T,
            numpy.sqrt(numpy.diag(self.factor_model.covariance_factor)),
        ).T

        return [
            # End portfolio factor risk is less than upper bound
            A @ (self.x + self.initial_weights)
            <= self.factor_risk_upper_bounds,
            A @ -(self.x + self.initial_weights)
            <= self.factor_risk_upper_bounds,
        ]
