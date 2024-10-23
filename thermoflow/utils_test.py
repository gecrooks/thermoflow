# Copyright 2021-2024 Gavin E. Crooks
#
# This source code is licensed under the Apache-2.0 License
# found in the LICENSE file in the root directory of this source tree.

import os
import jax
import jax.numpy as jnp
from jax.scipy.special import expit
from thermoflow.utils import logexpit, random_key


def test_random_key() -> None:
    key1 = random_key()

    key2 = random_key(42)

    os.environ["SEED"] = "42"
    key3 = random_key(1)

    assert key1 != key2
    assert key2 == key3


def test_logexpit() -> None:
    key = jax.random.key(42)
    x = jax.random.normal(key)
    assert jnp.allclose(logexpit(x), jnp.log(expit(x)))

    assert jnp.isclose(logexpit(-1000.0), -1000.0)  # type: ignore
    assert jnp.isclose(logexpit(1000.0), 0.0)  # type: ignore
