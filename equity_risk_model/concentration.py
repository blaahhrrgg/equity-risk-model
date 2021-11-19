import numpy


def enc(weights: numpy.array, alpha: int = 2) -> float:
    """Effective number of constituents

    This function returns the effective number of constituents (ENC) in a
    portfolio. Portfolio diversification (respectively, concentration) is
    increasing (respectively, decreasing) in the ENC measure. The measure is
    directly proportional to the inverse of the variance of the portfolio
    weights.

    Taking alpha equal to 2 leads to a diversification measure defined as the
    inverse of the Herfindahl-Hirschman index of the percentage weights in a
    given portfolio.

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


def entropy(weights: numpy.array) -> float:
    """Entropy of the distribution of portfolio weights

    The ENC measure converges to the entropy of the distribution of portfolio
    weights as alpha converges to one.

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
