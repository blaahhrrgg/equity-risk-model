from dataclasses import dataclass
from typing import Protocol

import numpy

import equity_risk_model


@dataclass
class FactorRiskModel(Protocol):
    """Factor Risk Model"""

    universe: numpy.array
    factors: numpy.array
    loadings: numpy.array
    covariance_factor: numpy.array
    covariance_specific: numpy.array
    covariance_total: numpy.array


class RiskCalculator:
    """Risk Calculator

    Calculates a number of risk measures using an equity factor risk model.
    """

    def __init__(self, factor_model: FactorRiskModel):
        self.factor_model = factor_model

    def total_risk(self, weights: numpy.array) -> float:
        """The total risk of the portfolio

        Parameters
        ----------
        weights : numpy.array
            Asset holding weights of the portfolio

        Returns
        -------
        float
            The total risk of the portfolio
        """
        return numpy.sqrt(
            weights.T @ self.factor_model.covariance_total @ weights
        )

    def total_factor_risk(self, weights: numpy.array) -> float:
        """The total factor risk of the portfolio

        Parameters
        ----------
        weights : numpy.array
            Asset holding weights of the portfolio

        Returns
        -------
        float
            The total factor risk of the portfolio
        """

        x = weights @ self.factor_model.loadings.T
        return numpy.sqrt(x.T @ self.factor_model.covariance_factor @ x)

    def total_specific_risk(self, weights: numpy.array) -> float:
        """The total specific risk of the portfolio

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        float
            The total specific risk of the portfolio
        """

        return numpy.sqrt(
            weights.T @ self.factor_model.covariance_specific @ weights
        )

    def factor_risks(self, weights: numpy.array) -> numpy.array:
        """The risk associated with each factor in the equity factor model

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        numpy.array
            The risk associated with each factor in the equity factor model
        """
        return numpy.sqrt(
            numpy.multiply(
                (self.factor_model.loadings @ weights) ** 2,
                numpy.diag(self.factor_model.covariance_factor),
            )
        )

    def factor_risk_covariance(self, weights: numpy.array) -> float:
        """Risk associated with covariances between the factors

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        float
            The risk due to covariances between factors
        """
        return numpy.sqrt(
            self.total_factor_risk(weights) ** 2
            - numpy.sum(self.factor_risks(weights) ** 2)
        )

    def marginal_contribution_to_total_risk(
        self, weights: numpy.array
    ) -> numpy.array:
        """Marginal contribution to the total risk from each asset

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        numpy.array
            An array of marginal risk contributions to the total risk from
            each asset in the portfolio
        """
        return numpy.multiply(
            weights, self.factor_model.covariance_total @ weights
        ) / self.total_risk(weights)

    def marginal_contribution_to_total_factor_risk(
        self, weights: numpy.array
    ) -> numpy.array:
        """Marginal contribution to the total factor risk from each asset

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        numpy.array
            An array of marginal risk contributions to the total factor risk
            from each asset in the portfolio
        """
        cov = (
            self.factor_model.loadings.T
            @ self.factor_model.covariance_factor
            @ self.factor_model.loadings
        )

        return numpy.multiply(weights, cov @ weights) / self.total_factor_risk(
            weights
        )

    def marginal_contribution_to_total_specific_risk(
        self, weights: numpy.array
    ) -> numpy.array:
        """Marginal contribution to the total specific risk from each asset

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        numpy.array
            An array of marginal risk contributions to the total specific risk
            from each asset in the portfolio
        """
        return numpy.multiply(
            weights, self.factor_model.covariance_specific @ weights
        ) / self.total_specific_risk(weights)

    def marginal_contributions_to_factor_risks(
        self, weights: numpy.array
    ) -> numpy.array:
        """Marginal contribution to the risk of each factor from each asset

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        numpy.array
            A matrix of marginal risk contribution where element [i, j] is the
            contribution of asset j to factor i.
        """
        cov = numpy.diag(numpy.diag(self.factor_model.covariance_factor))

        return numpy.divide(
            numpy.multiply(
                numpy.multiply(self.factor_model.loadings, weights.T).T,
                (cov @ self.factor_model.loadings @ weights),
            ),
            self.factor_risks(weights),
        ).T

    def effective_number_of_correlated_bets(
        self, weights: numpy.array
    ) -> float:
        """Effective number of correlated bets in the portfolio

        The normalised marginal contribution to the total risk of the portfolio
        from each asset is used in the calculation of the effective number of
        constituents.

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        float
            The effective number of correlated bets in the portfolio

        See Also
        --------
        equity_risk_model.concentration.enc
        """
        q = self.marginal_contribution_to_total_risk(weights) / self.total_risk(
            weights
        )

        return equity_risk_model.concentration.enc(q)

    def effective_number_of_uncorrelated_bets(
        self, weights: numpy.array
    ) -> float:
        """Effective number of uncorrelated bets in the portfolio

        The normalised marginal contribution to the total specific risk of the
        portfolio from each asset is used in the calculation of the effective
        number of constituents.

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        float
            The effective number of uncorrelated bets in the portfolio

        See Also
        --------
        equity_risk_model.concentration.enc
        """
        q = self.marginal_contribution_to_total_specific_risk(
            weights
        ) / self.total_specific_risk(weights)

        return equity_risk_model.concentration.enc(q)
