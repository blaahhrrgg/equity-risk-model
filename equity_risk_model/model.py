import dataclasses
from typing import Dict, List, Union

import numpy
import pandas


@dataclasses.dataclass
class FactorRiskModel:
    """Factor Risk Model

    A factor risk model performs factor risk decomposition and single stock
    risk attribution. For a universe of n assets and m factors, the risk model
    is given by a m by n matrix of factor loadings, a m by m matrix of factor
    covariances and a n by n (diagonal) matrix of specific asset returns.

    Attributes
    ----------
    universe : numpy.array
        An array of security identifiers that define the model universe
    factors : numpy.array
        An array of factor names for factors in the model
    loadings : pandas.DataFrame
        A matrix of factor loadings for the assets in the universe, the index
        should correspond to `factors` whilst the columns should correspond to
        `universe`
    covariance_factor : pandas.DataFrame
        A matrix of factor covariances, both the index and columns should
        correspond to `factors`
    covariance_specific : pandas.DataFrame
        A diagonal matrix of specific variances, both the index and columns
        should correspond to `universe`.
    factor_group_mapping : Dict[str, str], optional
        A dictionary keyed by factor group names with values corresponding to
        a list of factor names in the corresponding factor group.
    """

    universe: numpy.array
    factors: numpy.array
    loadings: pandas.DataFrame
    covariance_factor: pandas.DataFrame
    covariance_specific: pandas.DataFrame
    factor_group_mapping: Dict[str, List[str]] = dataclasses.field(
        default_factory=lambda: {}
    )

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
    def factor_groups(self) -> List[str]:
        """Factor groups in the factor model

        Returns
        -------
        List[str]
            A list of factor groups in the factor model (if no factor groups
            have been specified then returns an empty list)
        """
        return list(self.factor_group_mapping.keys())

    @property
    def factor_index(self) -> Union[pandas.MultiIndex, pandas.Index]:
        """Returns index for factors

        Returns
        -------
        Union[pandas.MultiIndex, pandas.Index]
            An index that can be used for factor ouputs, if a factor group
            mapping is provided, then the index will be a multilevel index.
        """

        if self.factor_group_mapping:

            idx = []

            for factor in self.factors:

                group = None

                for factor_group, factors in self.factor_group_mapping.items():
                    if factor in factors:
                        group = factor_group

                idx.append((group, factor))

            return pandas.MultiIndex.from_tuples(
                idx, names=["FactorGroup", "Factor"]
            )

        return pandas.Index(self.factors, name="Factor")

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
