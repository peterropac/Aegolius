# Copyright (C) 2026 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np
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
    return (resolution * c + (1 - c) * (resolution + 1)).astype(int)

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

    resolution = jnp.squeeze(jnp.asarray(resolution))
    if resolution.size == 1:
        co_res_0 = resolution_conversion(int(resolution))
        co_res_1 = co_res_0
        co_res_2 = co_res_0

    if resolution.size == 2:
        co_res_0 = resolution_conversion(int(resolution[0]))
        co_res_1 = resolution_conversion(int(resolution[1]))
        co_res_2 = co_res_0

    if resolution.size == 3:
        co_res_0 = resolution_conversion(int(resolution[0]))
        co_res_1 = resolution_conversion(int(resolution[1]))
        co_res_2 = resolution_conversion(int(resolution[2]))

    size = jnp.squeeze(jnp.asarray(size))
    if size.size == 1:
        x = jnp.linspace(-size / 2, size / 2, co_res_0)
        coor = jnp.zeros((3, co_res_0))
        coor = coor.at[0].set(x)

    if size.size == 2:
        x = jnp.linspace(-size[0] / 2, size[0] / 2, co_res_0)
        y = jnp.linspace(-size[1] / 2, size[1] / 2, co_res_1)

        co = jnp.asarray(jnp.meshgrid(x, y, indexing="ij"))
        co = co.reshape(2, -1)
        coor = jnp.zeros((3, co.shape[1]))
        coor = coor.at[:2].set(co)

    if size.size == 3:
        x = jnp.linspace(-size[0] / 2, size[0] / 2, co_res_0)
        y = jnp.linspace(-size[1] / 2, size[1] / 2, co_res_1)
        z = jnp.linspace(-size[2] / 2, size[2] / 2, co_res_2)

        co = jnp.asarray(jnp.meshgrid(x, y, z, indexing="ij"))
        coor = co.reshape(3, -1)

    return coor, (co_res_0.item(), co_res_1.item(), co_res_2.item())


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

    for i in range(1, pattern.shape[0]):
      out = out.at[i].set(smarter_reshape(pattern[i], resolution))

    return out


