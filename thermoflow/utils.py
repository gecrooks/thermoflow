# Copyright 2021-2024, Gavin E. Crooks
#
# This source code is licensed under the Apache-2.0 License
# found in the LICENSE file in the root directory of this source tree.


"""
Utilities


= JAX Type Hints
Jax type hints are a work in progress. See for example
https://github.com/google/jax/blob/main/jax/_src/typing.py

"""

import dataclasses
from dataclasses import dataclass
import time
import os
from typing import Any, Sequence, Tuple, Type, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import expit  # Logistic sigmoid function
from jax.tree_util import register_pytree_node


# == JAX Type Hints


type Array = jax.Array
"""JAX array type"""

type ArrayLike = jax.typing.ArrayLike

type DTypeLike = jax.typing.DTypeLike

type Shape = Sequence[int]
"""JAX array shape type"""

# == Jax Utils


def random_key(seed: Optional[int] = None) -> Array:
    """Return a random PRNG key array, seeded from the system clock.
    Can be override by providing an explicit seed, or by setting the SEED
    environment variable
    """
    seed = time.time_ns() if seed is None else seed
    seed = int(os.environ.get("SEED", seed))  #  environment variable override

    return jax.random.key(seed)


def dataclass_tuple(obj: Any) -> Tuple:
    """Convert a dataclass object to a tuple of data. Unlike `dataclasses.astuple` does
    not recurse."""
    return tuple(getattr(obj, field.name) for field in dataclasses.fields(obj))


def pytree_dataclass(cls: Type) -> Type:
    """Register a dataclass as a jax pytree node.

        @pytree_dataclass
        @dataclass(frozen=True)
        class SomeClass:
            x: int
            y: float

    Freezing the dataclass ensures we don't accidentally modify the data. We need to
    adopt a functional style and treat objects as immutable for jax.

    We don't try to build our own dataclass like decorator (as flax does). We don't
    seem to need all that extra complexity. And using custom dataclass decorators
    confuses the type checkers (which have a bunch of magic to deal with dataclass
    dynamical class creation)
    """
    # See
    # register_pytree_node_class
    #   https://github.com/google/jax/blob/main/jax/_src/tree_util.py
    # flax dataclass
    #   https://github.com/google/flax/blob/master/flax/struct.py
    # jax-md dataclass

    def tree_flatten(obj):  # type: ignore
        children = dataclass_tuple(obj)
        aux_data = None
        return (children, aux_data)

    def tree_unflatten(aux_data, children):  # type: ignore
        return cls(*children)

    register_pytree_node(cls, tree_flatten, tree_unflatten)

    return cls


# == Math


def logexpit(a: Array) -> Array:
    """
    Return the log of expit, the logistic sigmoid function.

        expit(x) = 1/(1+exp(-x))

    """
    # log(expit(+x)) = log(1/(1+exp(-x)))
    #            = x + log(1/(1+exp(+x)))
    #            = x + log(expit(-x))

    return jnp.piecewise(
        a,
        [a < 0, a >= 0],
        [lambda x: x + jnp.log(expit(-x)), lambda x: jnp.log(expit(x))],
    )


