# Copyright 2021-2024 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Estimates of free energy from work measurements.

All works and free energies are in multiples of kT. (i.e. nats)


Free Energy Estimates
---------------------
.. autofunction:: thermoflow.free_energy_bar

.. autofunction:: thermoflow.free_energy_bayesian

.. autofunction:: thermoflow.free_energy_logmeanexp

.. autofunction:: thermoflow.free_energy_logmeanexp_gaussian


Free Energy Estimates for symmetric protocols
--------------------------------------------
.. autofunction:: thermoflow.free_energy_symmetric_bar

.. autofunction:: thermoflow.free_energy_symmetric_bidirectional

.. autofunction:: thermoflow.free_energy_symmetric_nnznm"


"""

from typing import Tuple, Optional

import jax.numpy as jnp

from jax.scipy.special import (
    expit,
)  # Logistic sigmoid: expit(x) = 1/(1+exp(-x))
from jax.scipy.special import logsumexp
import jaxopt

from .utils import logexpit, Array, ArrayLike

__all__ = (
    "free_energy_bar",
    "free_energy_bayesian",
    "free_energy_logmeanexp",
    "free_energy_logmeanexp_gaussian",
    "free_energy_symmetric_bar",
    "free_energy_symmetric_bidirectional",
    "free_energy_symmetric_nnznm",
)


def free_energy_logmeanexp(work_f: ArrayLike) -> Array:
    """
    Calculate the free energy difference from a series of work measurements
    using the Jarzynski equality.

    (Note that we can't calculate an accurate error bounds for Jarzynski estimate
    using measurements from only a single protocol directions.)

    Args:
        work_f: Array of shape [N]
    Returns:
        Delta free energy
    Ref:
        TODO
    """
    work_f = jnp.asarray(work_f, dtype=jnp.float64)
    N_f = work_f.size
    delta_free_energy = -(logsumexp(-work_f) - jnp.log(N_f))

    return delta_free_energy


def free_energy_logmeanexp_gaussian(work_f: ArrayLike) -> Array:
    """
    Calculate the free energy difference from a series of work measurements
    under the assumption that the work distributions is Gaussian.

    Args:
        work_f: Array of shape [N]
    Returns:
        Delta free energy
    Ref:
        TODO
    """
    work_f = jnp.asarray(work_f, dtype=jnp.float64)
    delta_free_energy = jnp.average(work_f) - 0.5 * jnp.var(work_f)
    return delta_free_energy


def free_energy_bar(
    work_f: ArrayLike,
    work_r: ArrayLike,
    weights_f: Optional[ArrayLike] = None,
    weights_r: Optional[ArrayLike] = None,
    uncertainty_method: str = "BAR",
) -> Tuple[Array, Array]:
    """
    Estimate a free energy difference using the Bennett Acceptance Ratio method (BAR).

    Three methods for estimating the error are provided. The original 'BAR' method [1],
    which underestimates the error if the posterior is not Gaussian (this happens
    when there is little overlap between the forward and negative reverse work
    distributions and therefore the uncertainties are large); the 'MBAR' [3] approach which
    is optimal in the large sample limit, but overestimates the error when the
    posteriors are not Gaussian, and 'logistic', which is the MBAR error,
    but with a correction for non-overlapping work distributions.

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.
        weights_f:  Optional weights for forward works
        weights_r:  Optional weights for reverse works
        uncertainty_method: Method to calculate errors ("BAR" [1], "MBAR"[3], or "Logistic"[4])

    Returns:
        Estimated free energy difference, and the estimated error
    Refs:
        ???, ???, ???, ???

    """

    W_f = jnp.asarray(work_f, dtype=jnp.float64)
    W_r = jnp.asarray(work_r, dtype=jnp.float64)

    if weights_f is None:
        weights_f = jnp.ones_like(W_f)
    weights_f = jnp.asarray(weights_f, dtype=jnp.float64)

    if weights_r is None:
        weights_r = jnp.ones_like(W_r)
    weights_r = jnp.asarray(weights_r, dtype=jnp.float64)

    N_f = sum(weights_f)
    N_r = sum(weights_r)
    M = jnp.log(N_f / N_r)

    lower = jnp.min(jnp.asarray([jnp.amin(W_f), jnp.amin(-W_r)]))
    upper = jnp.max(jnp.asarray([jnp.amax(W_f), jnp.amax(-W_r)]))

    def _bar(delta_free_energy: float) -> Array:
        diss_f = W_f - delta_free_energy + M
        diss_r = W_r + delta_free_energy - M

        f = jnp.log(jnp.sum(weights_f * expit(-diss_f)))
        r = jnp.log(jnp.sum(weights_r * expit(-diss_r)))
        return f - r

    # Maximum likelihood free energy
    delta_free_energy = jaxopt.Bisection(_bar, lower, upper).run().params  # Find root

    # Error estimation
    diss_f = work_f - delta_free_energy + M
    diss_r = work_r + delta_free_energy - M

    slogF = jnp.sum(weights_f * expit(-diss_f))
    slogR = jnp.sum(weights_r * expit(-diss_r))

    slogF2 = jnp.sum(weights_f * expit(-diss_f) ** 2)
    slogR2 = jnp.sum(weights_r * expit(-diss_r) ** 2)

    nratio = (N_f + N_r) / (N_f * N_r)

    if uncertainty_method == "BAR":
        # BAR error estimate
        # (Underestimates error if posterior not Gaussian)
        err = jnp.sqrt((slogF2 / slogF**2) + (slogR2 / slogR**2) - nratio)
    elif uncertainty_method == "MBAR":
        # MBAR error estimate
        # (Massively overestimates error if posterior not Gaussian)
        err = jnp.sqrt(1.0 / (slogF - slogF2 + slogR - slogR2) - nratio)
    elif uncertainty_method == "Logistic":
        # MBAR error with a correction for non-overlapping work distributions
        mbar_err = jnp.sqrt(1.0 / (slogF - slogF2 + slogR - slogR2) - nratio)
        min_hysteresis = jnp.min(work_f) + jnp.min(work_r)
        logistic_err = jnp.sqrt((min_hysteresis**2 + 4 * jnp.pi**2) / 12)
        err = jnp.minimum(logistic_err, mbar_err)
    else:
        raise ValueError("Unknown uncertainty estimation method")

    return delta_free_energy, err


def free_energy_bayesian(work_f: ArrayLike, work_r: ArrayLike) -> Tuple[Array, Array]:
    """Bayesian free energy estimate

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.

    Returns:
        Posterior mean estimate of the free energy difference, and the estimated error
    """
    df, prob = free_energy_posterior(work_f, work_r)

    delta_free_energy = jnp.sum(df * prob)
    err = jnp.sqrt(jnp.sum(df * df * prob) - delta_free_energy**2)

    return delta_free_energy, err


def free_energy_posterior(work_f: ArrayLike, work_r: ArrayLike) -> Tuple[Array, Array]:
    """The Bayesian free energy posterior distribution.

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.
    Returns:
        energy and probability, pair of arrays of shapes [N]
    """

    w_f = jnp.asarray(work_f, dtype=jnp.float64)
    w_r = jnp.asarray(work_r, dtype=jnp.float64)

    fe, err = free_energy_bar(work_f, work_r, uncertainty_method="Logistic")
    lower = fe - 4 * err
    upper = fe + 4 * err

    x = jnp.linspace(lower, upper, 100, dtype=jnp.float64)

    res = [
        None,
    ] * x.size

    N_f = w_f.size
    N_r = w_r.size
    M = jnp.log(N_f / N_r)

    # FIXME
    for idx in range(x.size):
        fe = x[idx]
        diss_f = w_f - fe + M
        diss_r = w_r + fe - M
        res[idx] = jnp.sum(logexpit(diss_f)) + jnp.sum(logexpit(diss_r))  # type: ignore

    log_prob = jnp.asarray(res)

    log_prob -= jnp.amax(log_prob)
    prob = jnp.exp(log_prob)
    prob /= jnp.sum(prob)

    return x, prob


def free_energy_symmetric_bar(
    work_ab: ArrayLike,
    work_bc: ArrayLike,
    uncertainty_method: str = "BAR",
) -> Tuple[Array, Array]:
    """BAR for symmetric periodic protocols.

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
        uncertainty_method: Method to calculate errors (BAR, MBAR, or Logistic)

    Returns:
        Estimated free energy difference to the middle point of the protocol, and
        an estimated error
    """
    work_ab = jnp.asarray(work_ab, dtype=jnp.float64)
    work_bc = jnp.asarray(work_bc, dtype=jnp.float64)

    weights_r = jnp.exp(-work_ab - free_energy_logmeanexp(work_ab))
    return free_energy_bar(work_ab, work_bc, None, weights_r, uncertainty_method)


def free_energy_symmetric_nnznm(work_ab: ArrayLike, work_bc: ArrayLike) -> Array:
    """Free energy estimate for cyclic protocol.

    "Non equilibrium path-ensemble averages for symmetric protocols"
    Nguyen, Ngo, Zerba, Noskov, & Minh (2009), Eq 2

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
    Returns:
        Estimate of the free energy
    """
    work_ab = jnp.asarray(work_ab, dtype=jnp.float64)
    work_bc = jnp.asarray(work_bc, dtype=jnp.float64)

    delta_fenegy = (
        -jnp.log(2)
        + free_energy_logmeanexp(-work_ab)
        + jnp.log(1 + jnp.exp(-free_energy_logmeanexp(-work_ab - work_bc)))
    )

    return delta_fenegy


def free_energy_symmetric_bidirectional(
    work_ab: ArrayLike, work_bc: ArrayLike
) -> Array:
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

    work_ab = jnp.asarray(work_ab, dtype=jnp.float64)
    work_bc = jnp.asarray(work_bc, dtype=jnp.float64)

    N = work_ab.size

    return -(logsumexp(-work_ab + logexpit(-work_ab - work_bc)) - jnp.log(N / 2))
