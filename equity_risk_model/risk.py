import numpy
import pandas
from typing import Dict, Union

from .model import FactorRiskModel


PortfolioWeights = Union[numpy.array, pandas.Series]


class RiskCalculator:
    """Risk Calculator

    Calculates a number of risk measures using an equity factor risk model.
    """

    def __init__(self, factor_model: FactorRiskModel):
        self.factor_model = factor_model

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
        return numpy.sqrt(
            self.total_factor_risk(weights) ** 2
            - numpy.sum(
                numpy.power(list(self.factor_group_risks(weights).values()), 2)
            )
        )

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
        factor_risks = numpy.sqrt(
            numpy.multiply(
                (self.factor_model.loadings @ weights) ** 2,
                numpy.diag(self.factor_model.covariance_factor),
            )
        )

        if isinstance(factor_risks, (numpy.ndarray, numpy.generic)):
            factor_risks = pandas.Series(
                data=factor_risks, index=self.factor_model.factor_index
            )

        elif isinstance(factor_risks, pandas.Series):
            factor_risks.index = self.factor_model.factor_index

        return factor_risks

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
        return numpy.sqrt(
            self.total_factor_risk(weights) ** 2
            - numpy.sum(self.factor_risks(weights) ** 2)
        )

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
        q = numpy.multiply(
            weights, self.factor_model.covariance_total @ weights
        ) / self.total_risk(weights)

        if isinstance(q, (numpy.ndarray, numpy.generic)):
            mctr = pandas.Series(data=q, index=self.factor_model.universe)

        return mctr

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

        q = numpy.multiply(weights, cov @ weights) / self.total_factor_risk(
            weights
        )

        if isinstance(q, (numpy.ndarray, numpy.generic)):
            mcfr = pandas.Series(data=q, index=self.factor_model.universe)

        return mcfr

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
        q = numpy.multiply(
            weights, self.factor_model.covariance_specific @ weights
        ) / self.total_specific_risk(weights)

        if isinstance(q, (numpy.ndarray, numpy.generic)):
            mcsr = pandas.Series(data=q, index=self.factor_model.universe)

        return mcsr

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
