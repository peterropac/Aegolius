# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np
from jax.lax import fori_loop
import jax


@jax.jit
def resolution_conversion(resolution: int) -> int:
    """
    Converts the given resolution so that there are odd number of points along each axis.

    Args:
        resolution: Given resolution along an axis.

    Returns:
        converted Resolution along an axis.
    """
    c = resolution % 2 == 1
    return resolution * c + (1 - c) * (resolution + 1)


def smarter_reshape(pattern: jnp.ndarray, resolution: tuple | list | jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """
    Converts the Signed Distance field point cloud into a grid.

    Args:
        pattern: Signed Distance field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Signed distance field on a rectilinear grid.
    """

    n_ele = pattern.shape[0]
    resolution = jnp.asarray(resolution)

    if resolution.size == 1:
        res = resolution_conversion(resolution)
        if n_ele//res == 1:
            return pattern
        elif n_ele//(res**2) == 1:
            return pattern.reshape(res, res)
        elif n_ele//(res**3) == 1:
            return pattern.reshape(res, res, res)
        else:
            raise ValueError(f"Cannot reshape the pattern with shape {pattern.shape}")

    if resolution.size == 2:
        res0 = resolution_conversion(resolution[0])
        res1 = resolution_conversion(resolution[1])

        div = n_ele//(res0*res1)
        if div == 1:
            return pattern.reshape(res0, res1)

        elif not div == 1:
            if not div%1 == 0:
                raise ValueError(f"Cannot reshape the pattern with shape {pattern.shape}")
            else:
                return pattern.reshape(res0, res1, int(div))

        else:
            raise ValueError(f"Cannot reshape the pattern with shape {pattern.shape}")

    if resolution.size == 3:
        res0 = resolution_conversion(resolution[0])
        res1 = resolution_conversion(resolution[1])
        res2 = resolution_conversion(resolution[2])

        div = n_ele // (res0 * res1 * res2)
        if div==1:
            return pattern.reshape(res0, res1, res2)
        else:
            raise ValueError(f"Cannot reshape the pattern with shape {pattern.shape}")


def vector_smarter_reshape(pattern: jnp.ndarray, resolution: tuple | list | jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """
    Converts a vector field point cloud into a grid.

    Args:
        pattern: Vector field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Vector field on a rectilinear grid of shape (3, resolution[0], resolution[1], resolution[2]).
    """

    x = smarter_reshape(pattern[0], resolution)
    y = smarter_reshape(pattern[1], resolution)
    z = smarter_reshape(pattern[2], resolution)

    return jnp.asarray([x, y, z])


def nd_vector_smarter_reshape(pattern: jnp.ndarray, resolution: tuple | list | np.ndarray | jnp.ndarray) -> jnp.ndarray:
    """
    Converts an n-dimensional vector field point cloud into a grid.

    Args:
        pattern: Vector field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Vector field on a rectilinear grid of shape (ND, resolution[0], resolution[1], resolution[2]).
    """

    c = smarter_reshape(pattern[0], resolution)
    out = jnp.zeros((pattern.shape[0], *c.shape))
    out = out.at[0].set(c)

    def body(i, a):
        a = a.at[i].set(smarter_reshape(pattern[i], resolution))
        return a
    out = fori_loop(1, pattern.shape[0], body, out)

    return out


