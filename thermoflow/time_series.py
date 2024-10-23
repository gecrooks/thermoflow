# Copyright 2023 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp

from .utils import Array

# TODO: Add docstrings
# TODO: Add sumsample

# crosscorrelation_functions
# autocorrelation_function

# cross_statistical_inefficiency
# cross_statistical_inefficiency_stderr
# statistical_inefficiency
# statistical_inefficiency_stderr

# autocorrelation_time
# autocorrelation_time_stderr
# crosscorrelation_times
# crosscorrelation_times_stderr

# kirkwood_tensor
# kirkwood_tensor_stderr
# kirkwood_coefficient
# kirkwood_coefficient_stderr

# detect_equilibration

# correlated_time_series


DEFAULT_CORRELATION_TIME_METHOD = "ips"


def crosscorrelation_functions(multiple_time_series: Array) -> Array:
    _, T = multiple_time_series.shape
    X = jnp.fft.fft(multiple_time_series, 2 * T - 1)
    cf = jnp.fft.ifft(X[:, None, :] * jnp.conj(X)[None, :, :])  # outer product
    cf = jnp.real(cf)
    return cf


def autocorrelation_function(time_series: Array) -> Array:
    return crosscorrelation_functions(time_series[None, :])[0, 0]


def crosscorrelation_times(
    multiple_time_series: Array, method: Optional[str] = None
) -> Array:
    method = DEFAULT_CORRELATION_TIME_METHOD if method is None else method

    if method == "ips":
        return _crosscorrelation_times_ips(multiple_time_series)
    elif method == "batchmean":
        return _crosscorrelation_times_batchmean(multiple_time_series)
    assert False


def crosscorrelation_times_stderr(
    multiple_time_series: Array, method: Optional[str] = None, subsamples: int = 16
) -> Array:
    kwargs = {"method": method}
    stderr = _subsample_stderr(
        crosscorrelation_times, multiple_time_series, subsamples, **kwargs
    )
    return stderr


def _crosscorrelation_times_batchmean(
    multiple_time_series: Array,  # [N, T]
) -> Array:  # [N, N]
    mts = multiple_time_series

    N, T = mts.shape
    batch_size = int(jnp.floor(T ** (2 / 3)))

    batch_num = T // batch_size
    batches = mts[:, 0 : batch_size * batch_num].reshape(N, batch_num, batch_size)

    batch_means = jnp.sum(batches, -1) / batch_size
    crosscorrelation_times = 0.5 * batch_size * jnp.cov(batch_means) / jnp.cov(mts)

    # if N==1 then last line creates a scalar. Promote back to 2d array for consistency
    crosscorrelation_times = jnp.atleast_2d(crosscorrelation_times)

    return crosscorrelation_times


@jax.jit
def _crosscorrelation_times_ips(multiple_time_series: Array) -> Array:
    N, T = multiple_time_series.shape

    C_t = crosscorrelation_functions(multiple_time_series)[:, :, 0:T]
    C_t /= C_t[:, :, 0:1]
    C_t *= (1.0 - jnp.arange(T) / T)[None, :]

    def mask_post_initial_positive_sequence(C_t: Array) -> Array:
        # For each series, set the first negative element
        # and all following elements to zero.
        def scan_mask(carried_mask: Array, instant: Array) -> Tuple[Array, Array]:
            carried_mask = jnp.where(instant < 0, False, carried_mask)
            result = jnp.where(carried_mask, instant, 0)
            return carried_mask, result

        init_mask = jnp.full((N, N), True)
        _, result = jax.lax.scan(scan_mask, init_mask, jnp.transpose(C_t))
        return jnp.transpose(result)

    C_t = mask_post_initial_positive_sequence(C_t)

    crosscorrelation_times = jnp.sum(C_t, axis=-1) - 1
    crosscorrelation_times = (crosscorrelation_times + crosscorrelation_times.T) / 2

    return crosscorrelation_times


def autocorrelation_time(time_series: Array, method: Optional[str] = None) -> float:
    return float(crosscorrelation_times(time_series[None, :], method)[0, 0])


def autocorrelation_time_stderr(
    time_series: Array, method: Optional[str] = None, subsamples: int = 16
) -> float:
    mts = time_series[None, :]
    return float(crosscorrelation_times_stderr(mts, method, subsamples)[0, 0])


def cross_statistical_inefficiency(
    multiple_time_series: Array, method: Optional[str] = None
) -> Array:
    tau = crosscorrelation_times(multiple_time_series, method)
    g = 1.0 + 2.0 * tau
    g = jnp.where(g < 1.0, 1.0, g)
    return g


def cross_statistical_inefficiency_stderr(
    multiple_time_series: Array, method: Optional[str] = None, subsamples: int = 16
) -> Array:
    kwargs = {"method": method}
    stderr = _subsample_stderr(
        cross_statistical_inefficiency, multiple_time_series, subsamples, **kwargs
    )
    return stderr


def statistical_inefficiency(time_series: Array, method: Optional[str] = None) -> float:
    g = cross_statistical_inefficiency(time_series[None, :], method)
    return float(g[0, 0])


def statistical_inefficiency_stderr(
    time_series: Array, method: Optional[str] = None, subsamples: int = 16
) -> float:
    stderr = cross_statistical_inefficiency_stderr(
        time_series[None, :], method, subsamples
    )
    return float(stderr[0, 0])


@jax.jit
def kirkwood_tensor(
    multiple_time_series: Array,  # [N, T]
    method: Optional[str] = None,
    min_eigenvalue: float = 0.0,
) -> Array:  # [N, N]
    tau = crosscorrelation_times(multiple_time_series, method)
    var = jnp.cov(multiple_time_series)
    M = var * tau

    # This tensor should be symmetric and positive semi-definite, but may not be due to
    # statistical errors. We return the nearest symmetric positive semi-definite
    # matrix in the Frobenius norm with eigenvalues at least min_eigenvalue
    # https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/

    M = (M + M.T) / 2  # make sure tensor is symmetric
    eig, Q = jnp.linalg.eigh(M)
    eig = jnp.where(eig > min_eigenvalue, eig, min_eigenvalue)
    M = Q @ jnp.diag(eig) @ Q.conj().T
    M = jnp.real(M)

    return M


def kirkwood_tensor_stderr(
    multiple_time_series: Array,  # [N, T]
    method: Optional[str] = None,
    min_eigenvalue: float = 0.0,
    subsamples: int = 16,
) -> Array:  # [N, N]
    kwargs = {"method": method, "min_eigenvalue": min_eigenvalue}
    return _subsample_stderr(
        kirkwood_tensor, multiple_time_series, subsamples, **kwargs
    )


def kirkwood_coefficient(time_series: Array, method: Optional[str] = None) -> float:
    tau = autocorrelation_time(time_series, method)
    var = jnp.var(time_series)
    return float(var * tau)


def kirkwood_coefficient_stderr(
    time_series: Array, method: Optional[str] = None
) -> float:
    err = kirkwood_tensor_stderr(time_series[None, :], method)
    return float(err[0, 0])


def detect_equilibration(A_t: Array, nodes: int = 16) -> Tuple[int, float, float]:
    """Detect initial transient region of an equilibrating time series
    using a heuristic that maximizes the number of effectively uncorrelated samples.

    We evaluate the statistical inefficiency on a sequence of exponentially spaced
    time points, and search for the time point that maximizes the effective number
    of uncorrelated samples after that time. We iterate with finer grids until a
    local maximum is located. Since the data is noisy we may not locate the global
    maximum.

    Args:
        A_t (Float[Array, "T"]): time series
        nodes (int): Number of search nodes at each iteration.

    Returns:
        t (int):  start index of equilibrated data
        g (float): statistical inefficiency of equilibrated data
        Neff (float): Effective number of uncorrelated samples after time t

    References:
        [1] J. D. Chodera, A Simple Method for Automated Equilibration Detection in
        Molecular Simulations, J. Chem. Theory Comput. 12:1799 (2016)
        http://dx.doi.org/10.1021/acs.jctc.5b00784

    Kudos:
        Adapted from pymbar/timeseries.py::detect_equilibration_binary_search
    """
    nodes = max(4, nodes)  # Number of nodes for binary search must be > 4
    T = A_t.size

    if jnp.isclose(A_t.std(), 0.0):  # Special case if time series is constant.
        return 0, 1.0, 1.0

    start = 0
    end = T - 2  # Must have at least one element for statistical_inefficiency

    while end - start >= 4:
        times = 2 ** jnp.linspace(jnp.log2(start + 1), jnp.log2(end + 1), nodes) - 1
        time_grid = jnp.unique(times.round().astype(int))

        g_t = jnp.asarray([statistical_inefficiency(A_t[t:]) for t in time_grid])
        Neff_t = (T - time_grid + 1) / g_t
        k = Neff_t.argmax()

        start = time_grid[max(0, k - 1)]
        end = time_grid[min(time_grid.size - 1, k + 1)]

    return int(time_grid[k]), float(g_t[k]), float(Neff_t[k])


def correlated_time_series(
    key: Array,
    tau: float,
    steps: int,
    initial: Optional[float] = None,
) -> Array:  # [T]
    """Generate time series data with given correlation time, drawn from an
    autoregressive model of order 1.

    Note if you generate multiple series with the same random noise (same key),
    then those series are correlated with a cross-correlation time equal to the
    mean of the correlation times of each series.

    Args:
        key: A jax PRNG key
        tau: Correlation time of the generated time series
        steps: length of the generated time series
        initial: Initial value for the auto-regression model. Provide the last value
            of a previously generated time series to extend the series.

    Ref:
        https://en.wikipedia.org/wiki/Autoregressive_model#Example:_An_AR(1)_process

    """

    phi = jnp.exp(-1 / tau)
    sd = jnp.sqrt(1 / (1 - phi**2))

    if initial is None:
        # Generate random initial points with correct variance
        key, subkey = jax.random.split(key)
        initial_carry = jax.random.normal(subkey) * sd
    else:
        initial_carry = jnp.asarray(initial)

    noise = jax.random.normal(key, (steps,))

    def step(carry: Array, sliver: Array) -> Tuple[Array, Array]:
        result = phi * carry + sliver
        carry = result
        return carry, result

    _, result_stack = jax.lax.scan(step, initial_carry, noise)

    return result_stack


# TESTME DOCME
def subsample_timeseries(time_series: Array, transient: bool = False) -> Array:
    if transient:
        t, g, _ = detect_equilibration(time_series)
    else:
        # t = 0
        g = statistical_inefficiency(time_series)

    indices = jnp.round(jnp.arange(0, time_series.size, g)).astype(int)

    return time_series[indices]


def _subsample_stderr(
    function: Callable, multiple_time_series: Array, subsamples: int = 16, **kwargs: Any
) -> Array:
    """Utility function to estimate standard errors by splitting time series data into
    subsamples and evaluating the given function on each sample
    """
    mts = multiple_time_series
    N, T = mts.shape
    subsample_length = T // subsamples
    mts = mts[:, 0 : subsample_length * subsamples].reshape(
        N, subsamples, subsample_length
    )
    mts = jnp.transpose(mts, axes=(1, 0, 2))
    results = jnp.stack([function(ss, **kwargs) for ss in mts])

    stderr = jnp.sqrt(jnp.var(results, axis=0))

    return stderr
