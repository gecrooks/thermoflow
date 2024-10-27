# Copyright 2021-2024 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import jax
import jax.numpy as jnp
from thermoflow.free_energy import free_energy_bar, free_energy_logmeanexp
from thermoflow.work import exponential_work, gamma_work, gaussian_work, two_point_work


def test_gaussian_work() -> None:
    key = jax.random.key(42)
    shp = (1000, 4)
    W_f, W_r = gaussian_work(key, 4, free_energy=10.0, shape_f=shp, shape_r=shp)

    assert W_f.shape == shp
    assert W_r.shape == shp

    fe0 = free_energy_logmeanexp(W_f)
    fe1 = free_energy_logmeanexp(W_r)
    assert jnp.isclose(fe0, 10.0, rtol=0.1)
    assert jnp.isclose(fe1, -10.0, rtol=0.1)


def test_gamma_work() -> None:
    key = jax.random.key(42)
    shp = (1000,)

    key, subkey = jax.random.split(key)
    W_f, W_r = gamma_work(key, 3, 0.587, shape_f=shp, shape_r=(2000, 2))
    assert jnp.isclose(free_energy_logmeanexp(W_f), 0.0, atol=0.1)
    assert jnp.isclose(free_energy_logmeanexp(W_r), 0.0, atol=0.1)

    assert W_f.shape == shp
    assert W_r.shape == (2000, 2)

    key, subkey = jax.random.split(key)
    W_f, W_r = gamma_work(key, 9, 0.587, free_energy=0.0, shape_f=shp, shape_r=shp)
    assert jnp.isclose(free_energy_logmeanexp(W_f), 0.0, atol=0.1)
    assert jnp.isclose(free_energy_logmeanexp(W_r), 0.0, atol=0.4)

    # Jarzynski gives biased results for these parameters. But BAR still works.
    key, subkey = jax.random.split(key)
    W_f, W_r = gamma_work(key, 6, 5.34, free_energy=0.0, shape_f=shp, shape_r=shp)
    fe, err = free_energy_bar(W_f, W_r)
    assert jnp.isclose(fe, 0.0, atol=0.2)


def test_exponential_work() -> None:
    key = jax.random.key(42)
    shp = (1000,)

    key, subkey = jax.random.split(key)
    W_f, W_r = exponential_work(subkey, 0.5, free_energy=0.0, shape_f=shp, shape_r=shp)
    assert W_f.shape == shp
    assert W_r.shape == shp

    fe0 = free_energy_logmeanexp(W_f)
    fe1 = free_energy_logmeanexp(W_r)
    assert jnp.isclose(fe0, 0.0, atol=0.1)
    assert jnp.isclose(fe1, 0.0, atol=0.1)


def test_two_point_work() -> None:
    key = jax.random.key(42)
    shp = (10000,)

    key, subkey = jax.random.split(key)
    W_f, W_r = two_point_work(
        subkey, 0.9, free_energy=0.0, shape_f=shp, shape_r=(10001,)
    )
    assert W_f.shape == shp
    assert W_r.shape == (10001,)

    fe0 = free_energy_logmeanexp(W_f)
    fe1 = free_energy_logmeanexp(W_r)
    assert jnp.isclose(fe0, 0.0, atol=0.1)
    assert jnp.isclose(fe1, 0.0, atol=0.1)

    key, subkey = jax.random.split(key)
    kT = 2.5
    W_f, W_r = two_point_work(
        subkey, 0.9, free_energy=10.0, kT=kT, shape_f=shp, shape_r=shp * 2
    )
    fe0 = free_energy_logmeanexp(W_f / kT)
    fe1 = free_energy_logmeanexp(W_r / kT)
    assert jnp.isclose(fe0, 10.0 / kT, atol=0.1)
    assert jnp.isclose(fe1, -10.0 / kT, atol=0.1)
