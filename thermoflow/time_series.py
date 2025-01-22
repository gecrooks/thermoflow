# Copyright 2023 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Functions for the analysis of correlated time series.

Note that if the inherent timescales of the system are long compared to duration of the time series being analyzed,
then results will be inaccurate and unreliable.

If time series have initial transients should detected (with 'detect_equilibration') and removed before further
analysis.

Refs:
    [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
    J. Chem. Phys. 129:124105, 2008
    http://dx.doi.org/10.1063/1.2978177

    [2] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
    histogram analysis method for the analysis of simulated and parallel tempering simulations.
    JCTC 3(1):26-41, 2007.

Kudos:
    Much of this module is a re-implementation (in jax) of the timeseries module in pymbar
    https://github.com/choderalab/pymbar/blob/master/pymbar/timeseries.py


Equilibration and decorrelation
-------------------------------
.. autofunction:: thermoflow.subsample_time_series
.. autofunction:: thermoflow.detect_equilibration


Correlation functions
---------------------

We provide two methods for calculating autocorrelation times, 'ips' (default)
and 'batchmean'

(TODO: Explain methods)

.. autofunction:: thermoflow.autocorrelation_time
.. autofunction:: thermoflow.autocorrelation_time_stderr
.. autofunction:: thermoflow.crosscorrelation_times
.. autofunction:: thermoflow.crosscorrelation_times_stderr

.. autofunction:: thermoflow.crosscorrelation_functions
.. autofunction:: thermoflow.autocorrelation_function


Statistical inefficiency
------------------------
The  statistical inefficiency of correlated time series is defined as g = 1 + 2 tau,
where tau is the correlation time (measured in unit steps). We enforce a minimum  g>=1.

Refs:
     [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.

.. autofunction:: thermoflow.statistical_inefficiency
.. autofunction:: thermoflow.statistical_inefficiency_stderr
.. autofunction:: thermoflow.cross_statistical_inefficiency
.. autofunction:: thermoflow.cross_statistical_inefficiency_stderr


Kirkwood coefficients
---------------------
.. autofunction:: thermoflow.kirkwood_coefficient
.. autofunction:: thermoflow.kirkwood_coefficient_stderr

.. autofunction:: thermoflow.kirkwood_tensor
.. autofunction:: thermoflow.kirkwood_tensor_stderr


Generation
----------
.. autofunction:: thermoflow.correlated_time_series

"""

from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp

from .utils import Array

# TODO: expand docstrings

__all__ = (
    "subsample_time_series",
    "crosscorrelation_functions",
    "autocorrelation_function",
    "cross_statistical_inefficiency",
    "cross_statistical_inefficiency_stderr",
    "statistical_inefficiency",
    "statistical_inefficiency_stderr",
    "autocorrelation_time",
    "autocorrelation_time_stderr",
    "crosscorrelation_times",
    "crosscorrelation_times_stderr",
    "kirkwood_tensor",
    "kirkwood_tensor_stderr",
    "kirkwood_coefficient",
    "kirkwood_coefficient_stderr",
    "detect_equilibration",
    "correlated_time_series",
)


DEFAULT_CORRELATION_TIME_METHOD = "ips"


def crosscorrelation_functions(multiple_time_series: Array) -> Array:
    """Compute the crosscorrleation functions for a sequence of corrleated time series,
    using the fast Fourier transform.

    Args:
        multiple_time_series: Array of shape [N, T]
    Returns
        Array of shape [N, T]

    """
    _, T = multiple_time_series.shape
    X = jnp.fft.fft(multiple_time_series, 2 * T - 1)
    cf = jnp.fft.ifft(X[:, None, :] * jnp.conj(X)[None, :, :])  # outer product
    cf = jnp.real(cf)
    return cf


def autocorrelation_function(time_series: Array) -> Array:
    """Compute the autocorrleation functions for a corrleated time series,
    using the fast Fourier transform.

    Args:
        time_series: Array of shape [T]
    Returns
        Array of shape [T]
    """
    return crosscorrelation_functions(time_series[None, :])[0, 0]


def crosscorrelation_times(
    multiple_time_series: Array, method: Optional[str] = None
) -> Array:
    """Compute the crosscorrelation_time of a series of correlated time series.

    Args:
        multiple_time_series: Array of shape [N, T]
        method: Either 'ips' (defualt) or 'batchmean'
    Returns:
        tau, array of shape [N, N]
    """
    method = DEFAULT_CORRELATION_TIME_METHOD if method is None else method

    if method == "ips":
        return _crosscorrelation_times_ips(multiple_time_series)
    elif method == "batchmean":
        return _crosscorrelation_times_batchmean(multiple_time_series)
    assert False


def crosscorrelation_times_stderr(
    multiple_time_series: Array, method: Optional[str] = None, subseries: int = 16
) -> Array:
    kwargs = {"method": method}
    stderr = _subsample_stderr(
        crosscorrelation_times, multiple_time_series, subseries, **kwargs
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
    """Compute the autocorrelation_time of a correlated time series.

    Args:
        time_series: Array of shape [T]
        method: Either 'ips' (defualt) or 'batchmean'
    Returns:
        tau, autocorrelation time
    """
    return float(crosscorrelation_times(time_series[None, :], method)[0, 0])


def autocorrelation_time_stderr(
    time_series: Array, method: Optional[str] = None, subseries: int = 16
) -> float:
    """Compute the standard error of the estimated autocorrelation time of a correlated time series.

    Args:
        time_series: Array of shape [T]
        method: Either 'ips' (defualt) or 'batchmean'
    Returns:
        stderr
    """
    mts = time_series[None, :]
    return float(crosscorrelation_times_stderr(mts, method, subseries)[0, 0])


def cross_statistical_inefficiency(
    multiple_time_series: Array, method: Optional[str] = None
) -> Array:
    """
    Compute the cross statistical inefficiency of a collection of correlated time series.


    Args:
        multiple_time_series: Array of shape [N, T]
        method: Either 'ips' (defualt) or 'batchmean'
    Returns:
        g, the estimated statistical inefficiency
    """
    tau = crosscorrelation_times(multiple_time_series, method)
    g = 1.0 + 2.0 * tau
    g = jnp.where(g < 1.0, 1.0, g)
    return g


def cross_statistical_inefficiency_stderr(
    multiple_time_series: Array, method: Optional[str] = None, subseries: int = 16
) -> Array:
    """Compute the standard error for the estimated statistical inefficiency of a correlated time series.

    Args:
        time_series: Array of shape [N, T]
        method: Either 'ips' (defualt) or 'batchmean'
    Returns:
        Standard error, array of shape [N]
    """
    kwargs = {"method": method}
    stderr = _subsample_stderr(
        cross_statistical_inefficiency, multiple_time_series, subseries, **kwargs
    )
    return stderr


def statistical_inefficiency(time_series: Array, method: Optional[str] = None) -> float:
    """Compute the statistical inefficiency of a correlated time series.

    Args:
        time_series: Array of shape [T]
        method: Either 'ips' (defualt) or 'batchmean'
    Returns:
        g, the estimated statistical inefficiency

    """

    g = cross_statistical_inefficiency(time_series[None, :], method)
    return float(g[0, 0])


def statistical_inefficiency_stderr(
    time_series: Array, method: Optional[str] = None, subseries: int = 16
) -> float:
    """Compute the standard error for the estimated statistical inefficiency of a correlated time series.

    Args:
        time_series: Array of shape [T]
        method: Either 'ips' (defualt) or 'batchmean'
    Returns:
        Standard error
    """
    stderr = cross_statistical_inefficiency_stderr(
        time_series[None, :], method, subseries
    )
    return float(stderr[0, 0])


@jax.jit
def kirkwood_tensor(
    multiple_time_series: Array,  # [N, T]
    method: Optional[str] = None,
    min_eigenvalue: float = 0.0,
) -> Array:  # [N, N]
    """Compute the Kirkwood tensor for a sequence of correlated time series.

    The elements of the Kirkwood tensor are the Kirkwood coefficients,
    (The integrated correlation functions, or the variance times the correlation times).
    Within the  thermodynamic geometry of linear response, the Kirkwood tensor is the friction,
    and acts as the metric tensor

    This tensor should be symmetric and positive semi-definite, but may not be due to
    statistical errors. We return the nearest symmetric positive semi-definite
    matrix in the Frobenius norm with eigenvalues at least min_eigenvalue
    https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/

    Args:
        multiple_time_series:
            An array of shape [N, T]
        method:
            Method for estimating correlation times, 'ips' (Defualt) or 'batchmean'
        min_eigenvalue:
            Minimum eignevalues of the Kirkwood tensor, default zero.
    Returns:
        An array of shape [N, N]
    Refs:
        TODO
    """
    tau = crosscorrelation_times(multiple_time_series, method)
    var = jnp.cov(multiple_time_series)
    M = var * tau

    # Nearest symmetric positive semi-definite matrix in the Frobenius norm
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
    subseries: int = 16,
) -> Array:  # [N, N]
    """Estimated standard errors for the coefficients in the the Kirkwood tensor..

    Args:
        multiple_time_series:
            An array of shape [N, T]
        method:
            Method for estimating correlation times, 'ips' (default) or 'batchmean'
        min_eigenvalue:
            Minimum eignevalues of the Kirkwood tensor, default zero.
        subseries: TODO
    Returns:
        An array of shape [N, N]
    """

    kwargs = {"method": method, "min_eigenvalue": min_eigenvalue}
    return _subsample_stderr(kirkwood_tensor, multiple_time_series, subseries, **kwargs)


def kirkwood_coefficient(time_series: Array, method: Optional[str] = None) -> float:
    """The Kirkwood coefficients for a correlated time series.

    The Kirkwood coefficient is the integrated correlation functions, or the variance times the correlation time.

    Args:
        time_series:
            An array of shape [T]
        method:
            Method for estimating correlation times, 'ips' (Defualt) or 'batchmean'
    Returns:
        Kirkwood coefficient
    """
    tau = autocorrelation_time(time_series, method)
    var = jnp.var(time_series)
    return float(var * tau)


def kirkwood_coefficient_stderr(
    time_series: Array, method: Optional[str] = None
) -> float:
    """Estimate of the error of a Kirkwood coefficients for a correlated time series.

    Args:
        time_series:
            An array of shape [T]
        method:
            Method for estimating correlation times, 'ips' (Defualt) or 'batchmean'
    Returns:
        stderr
    """
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
        t, g, Neff
            start index of equilibrated data,
            statistical inefficiency of equilibrated data,
            Effective number of uncorrelated samples after time t.

    Refs:
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


# TESTME
def subsample_time_series(time_series: Array, transient: bool = False) -> Array:
    """Extract uncorrelated samples from correlated timeseries data.

    Args:
        time_series: A jax array of shape [T]
        transient: If True initial transients will be detected and removed using the detect_equilibration function.
    Returns:
        An array of uncorrelated subsamples.
    """

    if transient:
        t, g, _ = detect_equilibration(time_series)
    else:
        # t = 0
        g = statistical_inefficiency(time_series)

    indices = jnp.round(jnp.arange(0, time_series.size, g)).astype(int)

    return time_series[indices]


def _subsample_stderr(
    function: Callable, multiple_time_series: Array, subseries: int = 16, **kwargs: Any
) -> Array:
    """Utility function to estimate standard errors by splitting time series data into
    subseries and evaluating the given function on each sample
    """
    mts = multiple_time_series
    N, T = mts.shape
    subsample_length = T // subseries
    mts = mts[:, 0 : subsample_length * subseries].reshape(
        N, subseries, subsample_length
    )
    mts = jnp.transpose(mts, axes=(1, 0, 2))
    results = jnp.stack([function(ss, **kwargs) for ss in mts])

    stderr = jnp.sqrt(jnp.var(results, axis=0))

    return stderr


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

    Returns:
        Correlated time series, size [steps]

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
        initial_carry = jnp.asarray(float(initial))
    print("initial carry", initial_carry)
    
    noise = jax.random.normal(key, (steps,))

    def step(carry: Array, sliver: Array) -> Tuple[Array, Array]:
        result = phi * carry + sliver
        carry = result
        return carry, result

    _, result_stack = jax.lax.scan(step, initial_carry, noise)

    return result_stack
