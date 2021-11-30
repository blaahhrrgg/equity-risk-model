import dataclasses

import numpy


@dataclasses.dataclass
class FactorRiskModel:
    """Factor Risk Model

    A factor risk model performs factor risk decomposition and single stock
    risk attribution. For a universe of n assets and m factors, the risk model
    is given by a m by n matrix of factor loadings, a m by m matrix of factor
    covariances and a n by n (diagonal) matrix of specific asset returns.
    """

    universe: numpy.array
    factors: numpy.array
    loadings: numpy.array
    covariance_factor: numpy.array
    covariance_specific: numpy.array

    @property
    def n_assets(self) -> int:
        """Number of assets in the factor model

        Returns
        -------
        int
            The number of assets in the factor model
        """
        return len(self.universe)

    @property
    def n_factors(self) -> int:
        """Number of factors in the factor model

        Returns
        -------
        int
            The number of factors in the factor model
        """
        return len(self.factors)

    @property
    def covariance_total(self) -> numpy.array:
        """The covariance matrix for total returns

        Returns
        -------
        numpy.array
            Covariance matrix for total returns
        """
        return (
            self.loadings.T @ self.covariance_factor @ self.loadings
            + self.covariance_specific
        )
