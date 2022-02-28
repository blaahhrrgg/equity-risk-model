import numpy
from typing import Union, Dict


class ConcentrationCalculator:
    """ConcentrationCalculator provides methods to calculate standard"""

    def __init__(self, risk_calculator):
        self.risk_calculator = risk_calculator

    @staticmethod
    def entropy(weights: numpy.array) -> float:
        """Entropy of the distribution of portfolio weights

        The ENC measure converges to the entropy of the distribution of
        portfolio weights as alpha converges to one.

        Parameters
        ----------
        weights : numpy.array
            Percentage weights for a given portfolio

        Returns
        -------
        float
            The entropy of portfolio weights
        """
        return numpy.exp(-weights @ numpy.log(weights))

    @staticmethod
    def enc(weights: numpy.array, alpha: int = 2) -> float:
        """Effective number of constituents

        This function returns the effective number of constituents (ENC) in a
        portfolio. Portfolio diversification (respectively, concentration) is
        increasing (respectively, decreasing) in the ENC measure. The measure
        is directly proportional to the inverse of the variance of the
        portfolio weights.

        Taking alpha equal to 2 leads to a diversification measure defined as
        the inverse of the Herfindahl-Hirschman index of the percentage weights
        in a given portfolio.

        Parameters
        ----------
        weights : numpy.array
            Percentage weights for a given portfolio
        alpha : int, optional
            A free parameter of the measure, by default 2

        Returns
        -------
        float
            The effective number of constituents in the portfolio
        """
        return numpy.linalg.norm(weights, alpha) ** (alpha / (1 - alpha))

    def number_of_correlated_bets(self, weights: numpy.array) -> float:
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
        q = self.risk_calculator.marginal_contribution_to_total_risk(
            weights
        ) / self.risk_calculator.total_risk(weights)

        return self.enc(q)

    def number_of_uncorrelated_bets(self, weights: numpy.array) -> float:
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
        q = self.risk_calculator.marginal_contribution_to_total_specific_risk(
            weights
        ) / self.risk_calculator.total_specific_risk(weights)

        return self.enc(q)

    def min_assets_for_mcsr_threshold(
        self, weights: numpy.array, threshold: float = 0.5
    ) -> int:
        """The minimum number of assets required to reach a given specific risk
        contribution threshold

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio
        threshold : float, optional
            The target cumulative specific risk contribution, by default 0.5

        Returns
        -------
        int
            The minimum number of assets required to reach marginal specific
            risk contribution
        """

        q = self.risk_calculator.marginal_contribution_to_total_specific_risk(
            weights
        ) / self.risk_calculator.total_specific_risk(weights)

        return (
            q.sort_values(ascending=False)
            .cumsum()
            .loc[lambda x: x <= threshold]
            .shape[0]
        ) + 1

    def summarise_portfolio(
        self, weights: numpy.array
    ) -> Dict[str, Union[int, float]]:
        """Summarise concentration metrics in dictionary format

        Parameters
        ----------
        weights : numpy.array
            The holding weights of each asset of the portfolio

        Returns
        -------
        Dict[str, Union[int, float]]
            A dictionary summarising a portfolio in terms of concentration
            measures
        """

        return {
            "NAssets": len(weights),
            "NCorrelatedBets": self.number_of_correlated_bets(weights),
            "NUncorrelatedBets": self.number_of_uncorrelated_bets(weights),
            "NEffectiveConstituents": self.enc(weights),
            "NAssets>25%SpecificRisk": self.min_assets_for_mcsr_threshold(
                weights, 0.25
            ),
            "NAssets>50%SpecificRisk": self.min_assets_for_mcsr_threshold(
                weights, 0.5
            ),
        }
