# Copyright 2021-2024 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Random sampling of pairs of example work distributions that obey the fluctuation
theorem.

.. autofunction:: thermoflow.gaussian_work
.. autofunction:: thermoflow.gamma_work
.. autofunction:: thermoflow.exponential_work
.. autofunction:: thermoflow.two_point_work

"""

from typing import Tuple

import jax
import jax.numpy as jnp

from .utils import Array, Shape


__all__ = "gaussian_work", "gamma_work", "exponential_work", "two_point_work"


def gaussian_work(
    key: Array,
    mu: float,
    free_energy: float = 0.0,
    kT: float = 1.0,
    shape_f: Shape = (1,),
    shape_r: Shape = (1,),
) -> Tuple[Array, Array]:
    """Randomly sample Gaussian (normal) distributed work distributions with given mean
    dissipation.

    For Gaussian work distributions that obey the fluctuation theorems, the
    variance of dissipation is twice the mean, and the mean dissipation is the same
    forward and reverse.

    Args:
        key: JAX PRNG key
        mu: Mean dissipation (work minus free energy).
        free_energy: Shift work distributes to obtain this change in free energy.
        kT: Temperature in natural units
        shape_f: Array shape of forward work samples.
        shape_r: Array shape of reverse work samples.

    Returns:
        (W_f, W_r)
            Random samples from forward and reverse work distributions that
            obey the fluctuation theorem. Units of kT.

    Refs:
        [1] Nonequilibrium Equality for Free Energy Differences C. Jarzynski
        Phys. Rev. Lett. 78, 2690 1997
    """
    subkey0, subkey1 = jax.random.split(key)

    sd_work = jnp.sqrt(2 * mu)
    mean_work_f = mu
    mean_work_r = mu

    W_f = jax.random.normal(subkey0, shape_f) * sd_work + mean_work_f
    W_r = jax.random.normal(subkey1, shape_r) * sd_work + mean_work_r

    W_f = kT * W_f + free_energy
    W_r = kT * W_r - free_energy

    return W_f, W_r


def gamma_work(
    key: Array,
    a: float,
    scale: float,
    free_energy: float = 0.0,
    kT: float = 1.0,
    shape_f: Shape = (1,),
    shape_r: Shape = (1,),
) -> Tuple[Array, Array]:
    """Randomly sample gamma distributed work distributions with given mean
    dissipation.

    Gamma work distributions occur for the adiabatic compression or expansion of a
    dilute gas [1]. Parameters corresponding to figure 2 of [1]: (a=3, scale=0.587),
    (a=9, scale=0.587), (a=6, scale=5.34). For last case a Jarzynski average has
    large biases, and a bidirectional estimate of free energy is necessary.

    Args:
        key: JAX PRNG key
        a: Shape parameter of forward gamma work distribution.
        scale: Scale parameter of forward gamma work distribution.
        free_energy: Shift the work distributes to obtain this change in free energy.
        kT: Temperature in natural units
        shape_f: Array shape of forward work samples.
        shape_r: Array shape of reverse work samples.

    Returns:
        (W_f, W_r)
            Random samples from forward and reverse work distributions that
            obey the fluctuation theorem. Units of kT.

    Refs:
        [1] Work distribution for the adiabatic compression of a dilute and interacting
        classical gas, G.E. Crooks, C. Jarzynski. Phys. Rev. E 75 021116 (2007)
    """
    subkey0, subkey1 = jax.random.split(key)

    scale_f = scale
    scale_r = -scale_f / (1 + scale_f)

    W_f = jax.random.gamma(subkey0, a, shape_f) * scale_f
    W_r = jax.random.gamma(subkey1, a, shape_r) * scale_r

    delta = a * jnp.log(-scale_f / scale_r)  # Free energy before adjustment
    W_f = W_f - delta
    W_r = W_r + delta

    W_f = kT * W_f + free_energy
    W_r = kT * W_r - free_energy

    return W_f, W_r


def exponential_work(
    key: Array,
    scale: float,
    free_energy: float = 0.0,
    kT: float = 1.0,
    shape_f: Shape = (1,),
    shape_r: Shape = (1,),
) -> Tuple[Array, Array]:
    """Randomly sample exponentially distributed work distributions with given scale.
    Special case of gamma work distributions.

    Args:
        key: JAX PRNG key
        scale: Scale of exponential distribution of forward work.
        free_energy: Shift work distributes to obtain this change in free energy.
        kT: Temperature in natural units
        shape_f: Array shape of forward work samples.
        shape_r: Array shape of reverse work samples.

    Returns:
        (W_f, W_r)
            Random samples from forward and reverse work distributions that
            obey the fluctuation theorem. Units of kT.

    """
    return gamma_work(key, 1.0, scale, free_energy, kT, shape_f, shape_r)


def two_point_work(
    key: Array,
    p: float,
    free_energy: float = 0.0,
    kT: float = 1.0,
    shape_f: Shape = (1,),
    shape_r: Shape = (1,),
) -> Tuple[Array, Array]:
    """Sample work from a time symmetric two-point work distribution.

    The two-point distribution of work is the time symmetric distribution
    with the minimum variance for a given mean dissipation [1] [2] [3].

    Args:
        key: JAX PRNG key
        p: Probability of one point of the distribution.
        free_energy: Shift work distributes to obtain this change in free energy.
        kT: Temperature in natural units
        shape_f: Array shape of forward work samples.
        shape_r: Array shape of reverse work samples.

    Returns:
        (W_f, W_r)
            Random samples from forward and reverse work distributions that
            obey the fluctuation theorem. Units of kT.

    Ref:
        [1] https://arxiv.org/abs/1010.2319
        [2] https://arxiv.org/abs/1904.07574
        [3] https://arxiv.org/abs/2208.11206
    """
    assert 0.0 < p < 1.0  # FIXME

    subkey0, subkey1 = jax.random.split(key)

    w = jnp.log(p / (1 - p))

    W_f = (jax.random.bernoulli(subkey0, p, shape_f) - 0.5) * 2 * w
    W_r = (jax.random.bernoulli(subkey1, p, shape_r) - 0.5) * 2 * w

    W_f = kT * W_f + free_energy
    W_r = kT * W_r - free_energy

    return W_f, W_r
