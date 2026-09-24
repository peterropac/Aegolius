# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

from math import prod as mprod

import numpy as np


def as_point_array(points: np.ndarray | list | tuple,
                   dims: tuple = (2, 3),
                   pad_to: int | None = None,
                   min_points: int = 1) -> np.ndarray:
    """
    Converts a set of points to a new float array of shape (D, N).

    An array is read as (D, N) or (N, D), whichever gives an allowed coordinate space dimension D.
    When both readings are allowed, the final shape is determined based on `dims`.
    For example if dims=(2, 3), a (3, 2) array reads as three 2D points and dims=(3, 2) reads as two 3D points.
    Square arrays are read as (D, N).

    Args:
        points: Points with shape (D, N) or (N, D), a single point with shape (D,), or an empty sequence.
        dims: Allowed values of the spatial dimension D, in order of preference.
        pad_to: If given, rows of zeros are appended until D equals pad_to.
        min_points: Minimum number of points N.

    Returns:
        A new float array of shape (D, N).
    """
    dims = tuple(dims)
    arr = np.asarray(points, dtype=float)
    input_shape = arr.shape

    if arr.size == 0:
        arr = np.zeros((max(dims), 0))
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"Points must be a 2D array of shape (D, N), got shape {input_shape}.")
    rows_ok, cols_ok = arr.shape[0] in dims, arr.shape[1] in dims
    if not (rows_ok or cols_ok):
        raise ValueError(f"Points must have shape (D, N) or (N, D) with D in {dims}, got shape {input_shape}.")
    if not rows_ok or (cols_ok and dims.index(arr.shape[1]) < dims.index(arr.shape[0])):
        arr = arr.T
    if arr.shape[1] < min_points:
        raise ValueError(f"At least {min_points} point(s) required, got {arr.shape[1]}.")
    if pad_to is not None and arr.shape[0] < pad_to:
        arr = np.vstack((arr, np.zeros((pad_to - arr.shape[0], arr.shape[1]))))

    return arr


def resolution_conversion(resolution: int) -> int:
    """
    Converts the given resolution so that there are odd number of points along each axis.

    Args:
        resolution: Given resolution along an axis.

    Returns:
        converted Resolution along an axis.
    """
    return int(resolution if resolution % 2 == 1 else resolution + 1)


def _symmetric_axis(size: float | int, n: int) -> np.ndarray:
    """
    Axis of n (odd) points spanning [-size/2, size/2] and exactly symmetric about zero.

    np.linspace(-size/2, size/2, n) is not bitwise symmetric: mirrored points can differ
    by one ulp. A central difference of a field that is symmetric about the origin then
    returns rounding noise instead of exactly zero, which downstream normalization turns
    into a spurious unit vector. Building one half and mirroring it avoids that.
    """
    half = np.linspace(0.0, size / 2, n // 2 + 1)
    return np.concatenate((-half[:0:-1], half))


def grid_spacing(size: int | float | tuple | list | np.ndarray,
                 resolution: int | tuple | list | np.ndarray) -> tuple:
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

    resolution = np.atleast_1d(np.squeeze(resolution))
    if resolution.size not in (1, 2, 3):
        raise ValueError(
            f"Resolution must have 1, 2 or 3 elements, got {resolution.size}."
        )
    size = np.atleast_1d(np.squeeze(size))
    if size.size not in (1, 2, 3):
        raise ValueError(f"Size must have 1, 2 or 3 elements, got {size.size}.")

    padded = list(resolution) + [resolution[0]] * (3 - resolution.size)
    co_res = tuple(resolution_conversion(r) for r in padded)

    axes = [_symmetric_axis(s, n) for s, n in zip(size, co_res)]
    co = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1)
    coor = np.concatenate((co, np.zeros((3 - len(axes), co.shape[1]))))

    return coor, co_res


def smarter_reshape(pattern: np.ndarray, resolution: tuple | list | np.ndarray) -> np.ndarray:
    """
    Converts the Signed Distance field point cloud into a grid.

    Args:
        pattern: Signed Distance field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Signed distance field on a rectilinear grid.
    """

    n_ele = pattern.shape[0]
    res = tuple(int(r) if int(r) % 2 == 1 else int(r) + 1 for r in np.asarray(resolution).ravel())

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


def vector_smarter_reshape(pattern: np.ndarray, resolution: tuple | list | np.ndarray) -> np.ndarray:
    """
    Converts a vector field point cloud into a grid.

    Args:
        pattern: Vector field
        resolution: Resolution of the grid, determining the number of points along each axis.

    Returns:
        Vector field on a rectilinear grid of shape (3, resolution[0], resolution[1], resolution[2]).
    """

    return nd_vector_smarter_reshape(pattern[:3], resolution)


def nd_vector_smarter_reshape(pattern: np.ndarray, resolution: tuple | list | np.ndarray) -> np.ndarray:
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

    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}.")

    max_ = np.amax(pattern)
    min_ = np.amin(pattern)
    a = (max_ - min_)

    if a == 0:
        # constant field: every point already falls in a single bin
        return np.array(pattern, dtype=float, copy=True)

    v = (pattern - min_)/a

    if equal_width:
        # clip keeps v == 1.0 in the last bin instead of creating a bins+1-th one
        u = np.clip((v * bins).astype(int), 0, bins - 1) / (bins - 1)
    else:
        u = np.round(v * (bins - 1), 0) / (bins - 1)

    out = u*a + min_
    return out












