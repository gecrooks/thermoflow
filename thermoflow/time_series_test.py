# Copyright 2023 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import jax
import jax.numpy as jnp
from pymbar import timeseries as pymbar_timeseries  # type:ignore
from thermoflow import time_series

# TODO: pymbar should be optional. Test should run anyways.


def test_correlated_time_series() -> None:
    key = jax.random.key(42)

    ts = time_series.correlated_time_series(key, 10.0, 1000)
    assert ts.shape == (1000,)


def test_autocorrelation_time() -> None:
    key = jax.random.key(42)

    ts = time_series.correlated_time_series(key, 10.0, 100000)
    tau0 = time_series.autocorrelation_time(ts, method="ips")
    assert 9 < tau0 < 11


def test_autocorrelation_time_batchmean() -> None:
    key = jax.random.key(42)
    ts = time_series.correlated_time_series(key, 10.0, 10000000)
    tau0 = time_series.autocorrelation_time(ts, method="batchmean")
    assert 9 < tau0 < 11


def test_crosscorrelation_times() -> None:
    key = jax.random.key(42)

    # Generate with same key we get correlated time series, tau=15
    ts0 = time_series.correlated_time_series(key, 10.0, 1000000)
    ts1 = time_series.correlated_time_series(key, 20.0, 1000000)
    mts = jnp.asarray([ts0, ts1])

    _ = time_series.crosscorrelation_times(mts, method="ips")
    _ = time_series.crosscorrelation_times(mts, method="batchmean")
    tau = time_series.crosscorrelation_times(mts)

    assert 9 < tau[0, 0] < 11
    assert 14 < tau[0, 1] < 16
    assert 19 < tau[1, 1] < 21
    assert jnp.isclose(tau[0, 1], tau[1, 0])

    check = (pymbar_timeseries.statistical_inefficiency(ts0, ts1) - 1) / 2
    assert jnp.isclose(
        tau[0, 1], check, rtol=0.01
    )  # TODO: Check. Similar but not identical answers

    key, subkey = jax.random.split(key)
    ts2 = time_series.correlated_time_series(subkey, 10.0, 1000000)
    mts = jnp.asarray([ts0, ts2])
    tau = time_series.crosscorrelation_times(mts)
    check = (pymbar_timeseries.statistical_inefficiency(ts0, ts2) - 1) / 2
    assert jnp.isclose(tau[0, 1], check, rtol=0.1)


def test_statistical_inefficiency() -> None:
    key = jax.random.key(42)
    ts = time_series.correlated_time_series(key, 10.0, 1000000)

    g = time_series.statistical_inefficiency(ts)
    check = pymbar_timeseries.statistical_inefficiency(ts)
    assert jnp.isclose(g, check, rtol=0.001)

    err = time_series.statistical_inefficiency_stderr(ts)
    assert err > 0.0

    ts0 = time_series.correlated_time_series(key, 10.0, 1000000)
    ts1 = time_series.correlated_time_series(key, 20.0, 1000000)
    mts = jnp.asarray([ts0, ts1])

    gg = time_series.cross_statistical_inefficiency(mts)
    assert gg.shape == (2, 2)
    errs = time_series.cross_statistical_inefficiency_stderr(mts)
    assert jnp.all(errs > 0)


def test_autocorrelation_function() -> None:
    key = jax.random.key(42)
    ts = time_series.correlated_time_series(key, 9.0, 1000000)
    _ = time_series.autocorrelation_function(ts)


def test_autocorrelation_time_stderr() -> None:
    key = jax.random.key(42)
    ts = time_series.correlated_time_series(key, 9.0, 1000000)
    _ = time_series.autocorrelation_time(ts)
    _ = time_series.autocorrelation_time_stderr(ts)


def test_crosscorrelation_times_stderr() -> None:
    key = jax.random.key(42)

    # Generate with same key we get correlated time series, tau=15
    ts0 = time_series.correlated_time_series(key, 10.0, 1000000)
    ts1 = time_series.correlated_time_series(key, 20.0, 1000000)
    mts = jnp.asarray([ts0, ts1])
    err = time_series.crosscorrelation_times_stderr(mts)
    assert err.shape == (2, 2)


def test_kirkwood_coefficient() -> None:
    key = jax.random.key(42)
    tau = 12.0
    ts = time_series.correlated_time_series(key, 12.0, 1000000)
    var = 1 / (1 - jnp.exp(-1 / tau) ** 2)
    expected = tau * var

    k = time_series.kirkwood_coefficient(ts)
    jnp.isclose(k, expected, rtol=0.1)

    err = time_series.kirkwood_coefficient_stderr(ts)
    assert err > 0.0


def test_kirkwood_tensor() -> None:
    key = jax.random.key(42)

    # Correlated time series
    key, subkey = jax.random.split(key)
    T = 1000000
    ts0 = time_series.correlated_time_series(subkey, 10.0, T)
    key, subkey = jax.random.split(key)
    ts1 = time_series.correlated_time_series(subkey, 15.0, T)

    ts2 = (ts1 + ts0) / 2
    ts3 = (ts1 - ts0) / 2

    mts = jnp.asarray([ts2, ts3])
    K = time_series.kirkwood_tensor(mts)
    assert K.shape == (2, 2)
    eig, _ = jnp.linalg.eigh(K)
    assert min(eig) >= 0.0

    err0 = time_series.kirkwood_tensor_stderr(mts)
    assert jnp.all(err0 >= 0.0)

    # Less data, more noise
    err1 = time_series.kirkwood_tensor_stderr(mts[:, 0 : T // 4])
    assert jnp.all(err1 > err0)

    # Uncorrelated time series, small cross correlation
    key, subkey = jax.random.split(key)
    ts0 = time_series.correlated_time_series(subkey, 10.0, 1000000)
    key, subkey = jax.random.split(key)
    ts1 = time_series.correlated_time_series(subkey, 10.0, 1000000)
    mts = jnp.asarray([ts0, ts1])
    K = time_series.kirkwood_tensor(mts)
    assert K[0, 1] < 1.0  # Should be small, but not zero due to numerical inaccuracy
    assert jnp.isclose(K[0, 1], K[1, 0])

    # Generate with same key we get correlated time series,
    # but not PSD tensor. Check that gets fixed up
    ts0 = time_series.correlated_time_series(key, 10.0, 1000000)
    ts1 = time_series.correlated_time_series(key, 20.0, 1000000)
    mts = jnp.asarray([ts0, ts1])
    K = time_series.kirkwood_tensor(mts, min_eigenvalue=1.01)
    eig, _ = jnp.linalg.eigh(K)
    assert min(eig) > 1.0


def test_detect_equilibration() -> None:
    key = jax.random.key(42)

    # Generate a time series with a long initial transient. Settles down to
    # equilibrium by about t=1300
    ts = time_series.correlated_time_series(key, 100.0, 10000, initial=0.)
    print(ts[0:10])
    
    t, g, Neff = pymbar_timeseries.detect_equilibration_binary_search(ts)
    print(t, g, Neff)
    t, g, Neff = time_series.detect_equilibration(ts)
    print(t,g, Neff)
    # Results checked against pymbar
    
    assert jnp.isclose(g, 122.29501307)
    assert t == 1324
    assert jnp.isclose(Neff, 70.95138045)

    # Constant sequence special case
    ts *= 0.0
    ts += 10.0
    t, g, Neff = time_series.detect_equilibration(ts)
    assert t == 0
    assert jnp.isclose(g, 1.0)
    assert jnp.isclose(Neff, 1.0)
