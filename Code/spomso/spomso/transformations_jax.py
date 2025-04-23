# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np
import jax
from typing import Callable

array_like_type = jnp.ndarray | np.ndarray | list | tuple
scalar_like_type = float | int
function_like_type = Callable[[array_like_type, tuple], jnp.ndarray]


def move_sdf(function_: Callable[[array_like_type, tuple], array_like_type],
             move_vector: array_like_type) -> function_like_type:
    """
    Modifies the original SDF so that the origin is translated by the 'move_vector'.

    Args:
        function_: Original SDF.
        move_vector: A vector, with shape (3,), by which the origin is translated.

    Returns:
        Modified SDF.
    """
    @jax.jit
    def moved(co, *args):
        p = jnp.subtract(co.T - move_vector).T
        return function_(p, *args)

    return moved


def scale_sdf(function_: Callable[[array_like_type, tuple], array_like_type],
              scale_factor: scalar_like_type) -> function_like_type:
    """
    Modifies the original SDF by scaling it by the 'scale_factor'.

    Args:
        function_: Original SDF.
        scale_factor: A scalar factor by which the SDF is scaled.
    Returns:
        Modified SDF.
    """
    @jax.jit
    def scaled(co, *args):
        return scale_factor*function_(co/scale_factor, *args)

    return scaled


def rotate_sdf(function_: Callable[[array_like_type, tuple], array_like_type],
               rotation_matrix: array_like_type) -> function_like_type:
    """
    Modifies the original SDF by rotating it by the 'rotation_matrix'.

    Args:
        function_: Original SDF.
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def rotated(co, *args):
        rm = rotation_matrix.T
        co = rm.dot(co)
        return function_(co, *args)

    return rotated


def compound_euclidian_transform_sdf(function_: Callable[[array_like_type, tuple], array_like_type],
                                     rotation_matrix: array_like_type,
                                     move_vector: array_like_type,
                                     scale_factor: scalar_like_type) -> function_like_type:
    """
    Apply the euclidian transformations to the geometry (SDF).

    Args:
        function_: Original SDF
        rotation_matrix: 3x3 rotation matrix.
        move_vector: A vector, with shape (3,), by which the origin is translated.
        scale_factor: A scalar factor by which the SDF is scaled.

    Returns:
        Signed Distance field of shape (N,).
    """
    @jax.jit
    def transformed(co, *args):
        rm = rotation_matrix.T
        co = rm.dot(co)
        co = co / scale_factor
        co = jnp.subtract(co.T, rm.dot(move_vector)).T

        return scale_factor * function_(co, *args)

    return transformed
