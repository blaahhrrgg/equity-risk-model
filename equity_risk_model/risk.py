from ast import Call
import logging
import numpy
import pandas
from typing import Dict, Union, Callable

from .model import FactorRiskModel


PortfolioWeights = pandas.Series


class RiskCalculator:
    """Risk Calculator provides methods to calculate the risk of a portfolio
    using and factor risk model.

    References
    ----------
    .. Menchero, J., Orr, D.J., Wang, J., 2011. The Barra US Equity Model
    (USE4) Methodology Notes.
    """

    def __init__(self, factor_model: FactorRiskModel):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.factor_model = factor_model

    def _reindex_weights(func: Callable) -> Callable:
        """Define decorator to reindex weights to model universe"""

        def reindex(self, weights: PortfolioWeights) -> Callable:
            w = weights.reindex(self.factor_model.universe, axis=0).fillna(
                value=0
            )

            missing_tickers = set(weights.index).difference(
                set(self.factor_model.universe)
            )

            if missing_tickers:
                self.logger.warning(
                    "The following tickers are not in the model universe:\n"
                    f"{missing_tickers}"
                )

            return func(self, w)

        return reindex

    @_reindex_weights
    def total_risk(self, weights: PortfolioWeights) -> float:
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

    @_reindex_weights
    def total_factor_risk(self, weights: PortfolioWeights) -> float:
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

    @_reindex_weights
    def total_specific_risk(self, weights: PortfolioWeights) -> float:
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

    @_reindex_weights
    def factor_group_risks(
        self, weights: PortfolioWeights
    ) -> Dict[str, float]:
        """The risk associated with each factor group in the equity factor
        model

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        numpy.array
            The risk associated with each factor group in the equity factor
            model
        """
        out = {}

        for group, factors in self.factor_model.factor_group_mapping.items():

            loading = self.factor_model.loadings.loc[factors]
            cov_f = self.factor_model.covariance_factor.loc[factors, factors]

            sigma_f = loading.T @ cov_f @ loading

            out[group] = numpy.sqrt(weights.T @ sigma_f @ weights)

        return out

    @_reindex_weights
    def factor_group_covariance(self, weights: PortfolioWeights) -> float:
        """Risk associated with covariances between the factor groups

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        float
            The risk due to covariances between factors
        """
        difference = self.total_factor_risk(weights) ** 2 - numpy.sum(
            numpy.power(list(self.factor_group_risks(weights).values()), 2)
        )

        return numpy.sign(difference) * numpy.sqrt(abs(difference))

    @_reindex_weights
    def factor_risks(self, weights: PortfolioWeights) -> numpy.array:
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
        factor_risks = numpy.multiply(
            # Sign of loading to denote direction
            numpy.sign(self.factor_model.loadings @ weights),
            # Risk associated with each factor
            numpy.sqrt(
                numpy.multiply(
                    (self.factor_model.loadings @ weights) ** 2,
                    numpy.diag(self.factor_model.covariance_factor),
                )
            ),
        )

        factor_risks.index = self.factor_model.factor_index

        return factor_risks

    @_reindex_weights
    def factor_risk_covariance(self, weights: PortfolioWeights) -> float:
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
        difference = self.total_factor_risk(weights) ** 2 - numpy.sum(
            self.factor_risks(weights) ** 2
        )

        return numpy.sign(difference) * numpy.sqrt(abs(difference))

    @_reindex_weights
    def marginal_contribution_to_total_risk(
        self, weights: PortfolioWeights
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

    @_reindex_weights
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

    @_reindex_weights
    def marginal_contribution_to_total_specific_risk(
        self, weights: PortfolioWeights
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

    @_reindex_weights
    def marginal_contributions_to_factor_risks(
        self, weights: PortfolioWeights
    ) -> pandas.DataFrame:
        """Marginal contribution to the risk of each factor from each asset

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        pandas.DataFrame
            A matrix of marginal risk contribution where element [i, j] is the
            contribution of asset i to factor j.
        """
        # Only take diagonal elements
        cov = numpy.diag(numpy.diag(self.factor_model.covariance_factor))

        mcfr = numpy.divide(
            numpy.multiply(
                numpy.multiply(self.factor_model.loadings, weights.T).T,
                (cov @ self.factor_model.loadings @ weights),
            ),
            self.factor_risks(weights).values,
        )

        return pandas.DataFrame(
            data=mcfr,
            index=self.factor_model.universe,
            columns=self.factor_model.factor_index,
        )
