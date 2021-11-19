import numpy


def is_positive_semidefinite(matrix: numpy.array) -> bool:
    """Check whether a matrix is positive semi-definite or not

    Attempt to compute the Cholesky decomposition of the matrix, if this fails
    then the matrix is not positive semidefinite.

    Parameters
    ----------
    matrix : numpy.array
        A matrix

    Returns
    -------
    bool
        True if the matrix is positive semidefinite, else False

    References
    ----------
    .. https://stackoverflow.com/questions/16266720
    """
    try:
        numpy.linalg.cholesky(matrix)
        return True
    except numpy.linalg.LinAlgError:
        return False
