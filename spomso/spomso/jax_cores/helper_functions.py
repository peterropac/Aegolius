# Copyright (C) 2026 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

from math import prod as mprod

import jax
import jax.numpy as jnp
import numpy as np


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
    return (resolution * c + (1 - c) * (resolution + 1)).astype(int)


def _symmetric_axis(size: float | int, n: int) -> jnp.ndarray:
    """
    Axis of n (odd) points spanning [-size/2, size/2] and exactly symmetric about zero.
    See the NumPy implementation for why linspace is not used directly.
    """
    half = jnp.linspace(0.0, size / 2, n // 2 + 1)
    return jnp.concatenate((-half[:0:-1], half))


def grid_spacing(size: int | float | tuple | list | jnp.ndarray,
                 resolution: int | tuple | list | jnp.ndarray) -> tuple:
    """
    Physical distance between neighbouring points along each axis of a grid
    created by generate_grid(size, resolution).

    Args:
        size: Size of the grid along each dimension. A single value applies to every dimension.
        resolution: Number of grid points along each dimension (converted to odd, as in generate_grid).
            A single value applies to every dimension.

    Returns:
        Tuple with the grid spacing along each dimension.
    """

    size = np.asarray(size, dtype=float).ravel()
    resolution = np.asarray(resolution).ravel()

    ndim = max(size.size, resolution.size)
    if size.size == 1:
        size = np.repeat(size, ndim)
    if resolution.size == 1:
        resolution = np.repeat(resolution, ndim)
    if size.size != resolution.size:
        raise ValueError(f"size has {size.size} elements but resolution has {resolution.size}.")

    points = [int(r) if int(r) % 2 == 1 else int(r) + 1 for r in resolution]
    return tuple(float(s) / (n - 1) for s, n in zip(size, points))


def generate_grid(size: int | float | tuple | list | jnp.ndarray,
                  resolution: int | tuple | list | jnp.ndarray) -> (jnp.ndarray, tuple):
    """
    Generates a grid of points based on the provided size and resolution, centered at zero.
    The dimensionality of the grid is determined from the number of elements in the size input parameter.

    Args:
        size: size of the grid along each dimension.
        resolution: number of grid points along each dimension.

    Returns:
        Point cloud of points with shape (D, N),
                where D - dimensionality, N - total number of points in the grid.,
        Converted resolution of the grid, containing the number of points along each axis.
    """

    resolution = np.atleast_1d(np.squeeze(np.asarray(resolution)))
    if resolution.size not in (1, 2, 3):
        raise ValueError(
            f"Resolution must have 1, 2 or 3 elements, got {resolution.size}."
        )
    size = jnp.atleast_1d(jnp.squeeze(jnp.asarray(size)))
    if size.size not in (1, 2, 3):
        raise ValueError(f"Size must have 1, 2 or 3 elements, got {size.size}.")

    padded = list(resolution) + [resolution[0]] * (3 - resolution.size)
    co_res = tuple(resolution_conversion(int(r)).item() for r in padded)

    axes = [_symmetric_axis(size[i], co_res[i]) for i in range(size.size)]
    co = jnp.stack(jnp.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1)
    coor = jnp.concatenate((co, jnp.zeros((3 - len(axes), co.shape[1]))))

    return coor, co_res


def smarter_reshape(pattern: jnp.ndarray, resolution: tuple | list | jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """
    Converts the Signed Distance field point cloud into a grid.
    Can be used inside jax.jit as long as the resolution is static.

    Args:
        pattern: Signed Distance field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Signed distance field on a rectilinear grid.
    """

    n_ele = pattern.shape[0]
    res = tuple(
        int(r) if int(r) % 2 == 1 else int(r) + 1
        for r in np.asarray(resolution).ravel()
    )

    if len(res) == 1:
        candidates = [res * k for k in (1, 2, 3)]
    elif len(res) == 2:
        k, rem = divmod(n_ele, res[0] * res[1])  # optional trailing axis of length k
        candidates = [res + ((k,) if k > 1 else ())] if rem == 0 else []
    elif len(res) == 3:
        candidates = [res]
    else:
        raise ValueError(f"Resolution must have 1, 2 or 3 elements, got {len(res)}.")

    for shape in candidates:
        if mprod(shape) == n_ele:
            return pattern.reshape(shape)
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

    return nd_vector_smarter_reshape(pattern[:3], resolution)


def nd_vector_smarter_reshape(pattern: jnp.ndarray, resolution: tuple | list | np.ndarray | jnp.ndarray) -> jnp.ndarray:
    """
    Converts an n-dimensional vector field point cloud into a grid.

    Args:
        pattern: Vector field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Vector field on a rectilinear grid of shape (ND, resolution[0], resolution[1], resolution[2]).
    """

    grid_shape = smarter_reshape(pattern[0], resolution).shape
    return pattern.reshape(pattern.shape[0], *grid_shape)


