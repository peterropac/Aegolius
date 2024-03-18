# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def resolution_conversion(resolution: int) -> int:
    """
    Converts the given resolution so that there are odd number of points along each axis.

    Args:
        resolution: Given resolution along an axis.

    Returns:
        converted Resolution along an axis.
    """
    return int(resolution if resolution % 2 == 1 else resolution + 1)


def generate_grid(size: int | float | tuple | list | np.ndarray,
                  resolution: int | tuple | list | np.ndarray) -> (np.ndarray, tuple):
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

    resolution = np.asarray(resolution)
    if resolution.size == 1:
        co_res_0 = resolution_conversion(resolution)
        co_res_1 = co_res_0
        co_res_2 = co_res_0

    if resolution.size == 2:
        co_res_0 = resolution_conversion(resolution[0])
        co_res_1 = resolution_conversion(resolution[1])
        co_res_2 = co_res_0

    if resolution.size == 3:
        co_res_0 = resolution_conversion(resolution[0])
        co_res_1 = resolution_conversion(resolution[1])
        co_res_2 = resolution_conversion(resolution[2])

    size = np.asarray(size)
    if size.size == 1:
        x = np.linspace(-size[0] / 2,
                        size[0] / 2,
                        co_res_0)
        coor = np.zeros((3, co_res_0))
        coor[0] = x

    if size.size == 2:
        x = np.linspace(-size[0] / 2,
                        size[0] / 2,
                        co_res_0)

        y = np.linspace(-size[1] / 2,
                        size[1] / 2,
                        co_res_1)

        co = np.asarray(np.meshgrid(x, y, indexing="ij"))
        co = co.reshape(2, -1)
        coor = np.zeros((3, co.shape[1]))
        coor[:2] = co

    if size.size == 3:
        x = np.linspace(-size[0] / 2,
                        size[0] / 2,
                        co_res_0)

        y = np.linspace(-size[1] / 2,
                        size[1] / 2,
                        co_res_1)

        z = np.linspace(-size[2] / 2,
                        size[2] / 2,
                        co_res_2)

        co = np.asarray(np.meshgrid(x, y, z, indexing="ij"))
        coor = co.reshape(3, -1)

    return coor, (co_res_0, co_res_1, co_res_2)


def smarter_reshape(pattern: np.ndarray, resolution: tuple | list | np.ndarray) -> np.ndarray:
    """
    Converts the Signed Distance field point cloud into a grid.

    Args:
        pattern: Signed Distance field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Signed distance field in a grid.
    """

    n_ele = pattern.shape[0]
    resolution = np.asarray(resolution)

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


def vector_smarter_reshape(pattern: np.ndarray, resolution: tuple | list | np.ndarray) -> np.ndarray:
    """
    Converts a vector field point cloud into a grid.

    Args:
        pattern: Vector field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Vector field on a grid of shape (3, resolution[0], resolution[1], resolution[2]).
    """

    x = smarter_reshape(pattern[0], resolution)
    y = smarter_reshape(pattern[1], resolution)
    z = smarter_reshape(pattern[2], resolution)

    return np.asarray([x, y, z])


def nd_vector_smarter_reshape(pattern: np.ndarray, resolution: tuple | list | np.ndarray) -> np.ndarray:
    """
    Converts an n-dimensional vector field point cloud into a grid.

    Args:
        pattern: Vector field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Vector field on a grid of shape (ND, resolution[0], resolution[1], resolution[2]).
    """

    c = smarter_reshape(pattern[0], resolution)
    out = np.zeros((pattern.shape[0], *c.shape))
    out[0] = c
    for i in range(1, pattern.shape[0]):
        out[i] = smarter_reshape(pattern[i], resolution)

    return out


def binning(pattern: np.ndarray, bins: int, equal_width: bool = True) -> np.ndarray:
    """
    Discretize the Signed Distance field/pattern to based on the specified number of bins.

    Args:
        pattern: Signed Distance field or any field.
        bins: Number of bins - unique discrete values in the final pattern.
        equal_width: All the bins are of equal width. If False the first and the last bin have half the width.

    Returns:
        Modified field.
    """

    max_ = np.amax(pattern)
    min_ = np.amin(pattern)
    a = (max_ - min_)
    v = (pattern - min_)/a

    if equal_width:
        u = (v * bins).astype(int) / (bins - 1)
    else:
        u = np.round(v * (bins - 1), 0) / (bins - 1)

    out = u*a + min_
    return out












