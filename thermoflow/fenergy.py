# Copyright 2021, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike  # numpy v1.20
from scipy import optimize
from scipy.special import expit  # Logistic sigmoid function: expit(x) = 1/(1+exp(-x))
from scipy.special import logsumexp

__all__ = (
    "fenergy_bar",
    "fenergy_bayesian",
    "fenergy_logmeanexp",
    "fenergy_logmeanexp_gaussian",
    "fenergy_symmetric_bar",
    "fenergy_symmetric_bidirectional",
    "fenergy_symmetric_nnznm",
)


def fenergy_logmeanexp(work_f: ArrayLike) -> float:
    work_f = np.asarray(work_f, dtype=np.float64)
    N_f = work_f.size
    delta_fenergy = -(logsumexp(-work_f) - np.log(N_f))

    return delta_fenergy


def fenergy_logmeanexp_gaussian(work_f: ArrayLike) -> float:
    work_f = np.asarray(work_f, dtype=np.float64)
    delta_fenergy = np.average(work_f) - 0.5 * np.var(work_f)
    return delta_fenergy


def fenergy_bar(
    work_f: ArrayLike,
    work_r: ArrayLike,
    weights_f: ArrayLike = None,
    weights_r: ArrayLike = None,
    uncertainty_method: str = "BAR",
) -> Tuple[float, float]:
    """

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.
        weights_f:  Optional weights for forward works
        weights_r:  Optional weights for reverse works
        uncertainty_method: Method to calculate errors ("BAR", "MBAR", or "Logistic")

    Returns:
        Estimated free energy difference, and the estimated error

    """

    W_f = np.asarray(work_f, dtype=np.float64)
    W_r = np.asarray(work_r, dtype=np.float64)

    if weights_f is None:
        weights_f = np.ones_like(W_f)
    weights_f = np.asarray(weights_f, dtype=np.float64)

    if weights_r is None:
        weights_r = np.ones_like(W_r)
    weights_r = np.asarray(weights_r, dtype=np.float64)

    N_f = sum(weights_f)
    N_r = sum(weights_r)
    M = np.log(N_f / N_r)

    lower = min(np.amin(W_f), np.amin(-W_r))
    upper = max(np.amax(W_f), np.amax(-W_r))

    def _bar(delta_fenergy: float) -> float:

        diss_f = W_f - delta_fenergy + M
        diss_r = W_r + delta_fenergy - M

        f = np.log(np.sum(weights_f * expit(-diss_f)))
        r = np.log(np.sum(weights_r * expit(-diss_r)))
        return f - r

    # Maximum likelihood free energy
    delta_fenergy = optimize.brentq(_bar, lower, upper)  # Find root

    # Error estimation
    diss_f = work_f - delta_fenergy + M
    diss_r = work_r + delta_fenergy - M

    slogF = np.sum(weights_f * expit(-diss_f))
    slogR = np.sum(weights_r * expit(-diss_r))

    slogF2 = np.sum(weights_f * expit(-diss_f) ** 2)
    slogR2 = np.sum(weights_r * expit(-diss_r) ** 2)

    nratio = (N_f + N_r) / (N_f * N_r)

    if uncertainty_method == "BAR":
        # BAR error estimate
        # (Underestimates error if posterior not Gaussian)
        err = np.sqrt((slogF2 / slogF ** 2) + (slogR2 / slogR ** 2) - nratio)
    elif uncertainty_method == "MBAR":
        # MBAR error estimate
        # (Massively overestimates error if posterior not Gaussian)
        err = np.sqrt(1.0 / (slogF - slogF2 + slogR - slogR2) - nratio)
    elif uncertainty_method == "Logistic":
        # MBAR error with a correction for non-overlapping work distributions
        mbar_err = np.sqrt(1.0 / (slogF - slogF2 + slogR - slogR2) - nratio)
        min_hysteresis = np.min(work_f) + np.min(work_r)
        logistic_err = np.sqrt((min_hysteresis ** 2 + 4 * np.pi ** 2) / 12)
        err = min(logistic_err, mbar_err)
    else:
        raise ValueError("Unknown uncertainty estimation method")

    return delta_fenergy, err


def fenergy_bayesian(work_f: ArrayLike, work_r: ArrayLike) -> Tuple[float, float]:
    """Bayesian free energy estimate

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.

    Returns:
        Posterior mean estimate of the free energy difference, and the estimated error
    """
    df, prob = fenergy_posterior(work_f, work_r)

    delta_fenergy = np.sum(df * prob)
    err = np.sqrt(np.sum(df * df * prob) - delta_fenergy ** 2)

    return delta_fenergy, err


def fenergy_posterior(
    work_f: ArrayLike, work_r: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """DOCME"""

    w_f = np.asarray(work_f, dtype=np.float64)
    w_r = np.asarray(work_r, dtype=np.float64)

    fe, err = fenergy_bar(work_f, work_r, uncertainty_method="Logistic")
    lower = fe - 4 * err
    upper = fe + 4 * err

    x = np.linspace(lower, upper, 100, dtype=np.float64)

    log_prob = np.zeros_like(x)

    N_f = w_f.size
    N_r = w_r.size
    M = np.log(N_f / N_r)

    for idx in range(x.size):
        fe = x[idx]
        diss_f = w_f - fe + M
        diss_r = w_r + fe - M
        log_prob[idx] = np.sum(_logexpit(diss_f)) + np.sum(_logexpit(diss_r))

    log_prob -= np.amax(log_prob)
    prob = np.exp(log_prob)
    prob /= np.sum(prob)

    return x, prob


def fenergy_symmetric_bar(
    work_ab: ArrayLike,
    work_bc: ArrayLike,
    uncertainty_method: str = "BAR",
) -> Tuple[float, float]:
    """BAR for symmetric periodic protocols.

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
        uncertainty_method: Method to calculate errors (BAR, MBAR, or Logistic)

    Returns:
        Estimated free energy difference to the middle point of the protocol, and
        an estimated error
    """
    work_ab = np.asarray(work_ab, dtype=np.float64)
    work_bc = np.asarray(work_bc, dtype=np.float64)

    weights_r = np.exp(-work_ab - fenergy_logmeanexp(work_ab))
    return fenergy_bar(work_ab, work_bc, None, weights_r, uncertainty_method)


def fenergy_symmetric_nnznm(work_ab: ArrayLike, work_bc: ArrayLike) -> float:
    """Free energy estimate for cyclic protocol.

    "Non equilibrium path-ensemble averages for symmetric protocols"
    Nguyen, Ngo, Zerba, Noskov, & Minh (2009), Eq 2

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
    Returns:
        Estimate of the free energy
    """
    work_ab = np.asarray(work_ab, dtype=np.float64)
    work_bc = np.asarray(work_bc, dtype=np.float64)

    delta_fenegy = (
        -np.log(2)
        + fenergy_logmeanexp(-work_ab)
        + np.log(1 + np.exp(-fenergy_logmeanexp(-work_ab - work_bc)))
    )

    return delta_fenegy


def fenergy_symmetric_bidirectional(work_ab: ArrayLike, work_bc: ArrayLike) -> float:
    """
    The bidirectional Minh-Chodera free energy estimate specialized to a symmetric
    protocol.

    Delta F = (2/N) sum (e^W_ab + e^-W_bc)^-1)

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
    Returns:
        Estimate of the free energy
    """

    work_ab = np.asarray(work_ab, dtype=np.float64)
    work_bc = np.asarray(work_bc, dtype=np.float64)

    N = work_ab.size

    return -(logsumexp(-work_ab + _logexpit(-work_ab - work_bc)) - np.log(N / 2))


def _logexpit(a: np.ndarray) -> np.ndarray:
    """
    log(expit(+x)) = log(1/(1+exp(-x)))
                   = x + log(1/(1+exp(+x)))
                   = x + log(expit(-x))

    """
    return np.piecewise(
        a,
        [a < 0, a >= 0],
        [lambda x: x + np.log(expit(-x)), lambda x: np.log(expit(x))],
    )
