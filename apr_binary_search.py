import numpy as np


def apr_binary_search(
    g,
    gmin,
    gmax,
    gamma,
    k,
    demand,
    tol=1e-8,
    max_iter=100
):
    """
    Reconstruct generator-contingency dispatch using APR.

    All power quantities must use consistent units
    (preferably per unit).

    Parameters
    ----------
    g : array
        Nominal generation dispatch.
    gmin, gmax : arrays
        Generator minimum and maximum limits.
    gamma : float or array
        APR response coefficients.
    k : int
        Failed generator index (zero-based).
    demand : float
        Total realized demand.
    tol : float
        Power-balance tolerance.

    Returns
    -------
    gk : array or None
        Reconstructed post-contingency dispatch.
    zk : float or None
        Primary-response signal.
    xk : array or None
        APR binary indicators (P1 convention).
    feasible : bool
        Whether APR power balance is satisfied.
    """

    g = np.asarray(g, dtype=float)
    gmin = np.asarray(gmin, dtype=float)
    gmax = np.asarray(gmax, dtype=float)

    gamma = np.broadcast_to(
        np.asarray(gamma, dtype=float),
        g.shape
    )

    if not (g.shape == gmin.shape == gmax.shape):
        raise ValueError("Generator arrays have inconsistent shapes")

    if not 0 <= k < len(g):
        raise ValueError("Invalid contingency index")

    if np.any(gamma < 0):
        raise ValueError("APR coefficients must be nonnegative")

    if np.any(g < gmin - tol) or np.any(g > gmax + tol):
        raise ValueError("Nominal generation violates bounds")

    capacity = gmax - gmin

    def dispatch(z):
        result = np.minimum(
            g + z * gamma * capacity,
            gmax
        )
        result[k] = 0.0
        return result

    # Check APR feasibility
    if (
        dispatch(0.0).sum() > demand + tol
        or dispatch(1.0).sum() < demand - tol
    ):
        return None, None, None, False

    low = 0.0
    high = 1.0

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        gk = dispatch(mid)

        mismatch = gk.sum() - demand

        if abs(mismatch) <= tol:
            break

        if mismatch < 0:
            low = mid
        else:
            high = mid

    zk = mid
    gk = dispatch(zk)

    # P1 convention:
    # x = 1: proportional APR response
    # x = 0: generator saturated at maximum output
    provisional = g + zk * gamma * capacity

    xk = (provisional < gmax - tol).astype(int)
    xk[k] = 0  # Not applicable for failed generator

    feasible = abs(gk.sum() - demand) <= tol

    return gk, zk, xk, feasible
