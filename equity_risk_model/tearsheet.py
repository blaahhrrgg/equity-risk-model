import abc
import numpy
import pandas
from typing import Dict, Union

from equity_risk_model.risk import RiskCalculator
from equity_risk_model.concentration import ConcentrationCalculator


PortfolioWeights = Union[numpy.array, pandas.Series]


class BaseTearsheet:
    """Class to create a DataFrame summary of a portfolio"""

    @abc.abstractmethod
    def create_portfolio_panel(
        self, weights: PortfolioWeights
    ) -> pandas.Series:
        """Create a panel summarising a portfolio from portfolio weights

        Parameters
        ----------
        weights : PortfolioWeights
            The holding weights of each asset of the portfolio

        Returns
        -------
        pandas.Series
            A pandas series with summary information of the portfolio

        """
        raise NotImplementedError

    def create_tearsheet(
        self, portfolio_weights: Dict[str, PortfolioWeights]
    ) -> pandas.DataFrame:
        """Create a tearsheet from a dictionary of portfolio weights

        Parameters
        ----------
        portfolio_weights : Dict[str, PortfolioWeights]
            A dictionary of portfolio weights, the key is used as the portfolio
            identifier in the DataFrame

        Returns
        -------
        pandas.DataFrame
            A DataFrame providing a tabular summary of the portfolios
        """
        return pandas.DataFrame(
            {
                portfolio_name: self.create_portfolio_panel(weights)
                for portfolio_name, weights in portfolio_weights.items()
            }
        )


class ConcentrationTearsheet(BaseTearsheet):
    """Class to provide a summary of concentration metrics for portfolios"""

    def __init__(self, concentration_calculator: ConcentrationCalculator):
        self.concentration_calculator = concentration_calculator

    def create_portfolio_panel(
        self, weights: PortfolioWeights
    ) -> pandas.Series:
        return pandas.Series(
            self.concentration_calculator.summarise_portfolio(weights)
        )


class RiskTearsheet(BaseTearsheet):
    """Class the provides a summary of a portfolio's factor risk"""

    def __init__(self, risk_calculator: RiskCalculator):
        self.risk_calculator = risk_calculator


class FactorRiskSummaryTearsheet(RiskTearsheet):
    """Tearsheet summarising total, factor and specific risk"""

    def create_portfolio_panel(
        self, weights: PortfolioWeights
    ) -> pandas.Series:
        return pandas.Series(
            {
                "Total": self.risk_calculator.total_risk(weights),
                "Factor": self.risk_calculator.total_factor_risk(weights),
                "Specific": self.risk_calculator.total_specific_risk(weights),
            }
        )


class FactorGroupRiskTearsheet(RiskTearsheet):
    """Tearsheet summarising factor group risk"""

    def create_portfolio_panel(
        self, weights: PortfolioWeights
    ) -> pandas.Series:
        return pandas.Series(
            self.risk_calculator.factor_group_risks(weights)
            | {
                "Covariance": self.risk_calculator.factor_group_covariance(
                    weights
                )
            }
        )


class FactorRiskTearsheet(RiskTearsheet):
    """Tearsheet summarising individual factor risks"""

    def create_portfolio_panel(
        self, weights: PortfolioWeights
    ) -> pandas.Series:
        return self.risk_calculator.factor_risks(weights)
