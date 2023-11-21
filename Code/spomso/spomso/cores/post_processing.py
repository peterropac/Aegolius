# Copyright (C) 2023 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.ndimage import convolve
from typing import Callable

# ----------------------------------------------------------------------------------------------------------------------
# SCALAR TRANSFORM FUNCTIONS


def sigmoid_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a sigmoid to the scalar (Signed Distance Function) field.
    :param u: Signed Distance field or any scalar field.
    :param amplitude: Maximum value of the transformed scalar field.
    :param width: Width of the sigmoid.
    :return: Transformed scalar field.
    """
    e = np.exp(4 * u / width)

    return amplitude*(1/(1 + e))


def positive_sigmoid_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a sigmoid, shifted to the positive velues by the value of the width parameter,
    to the scalar (Signed Distance Function) field.
    :param u: Signed Distance field or any scalar field.
    :param amplitude: Maximum value of the transformed scalar field.
    :param width: Width of the sigmoid.
    :return: Transformed scalar field.
    """
    e = np.exp(4 * (u - width) / width)

    return amplitude*(1/(1 + e))


def capped_exponential(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a decreasing exponential functon to the scalar (Signed Distance Function) field.
    :param u: Signed Distance field or any scalar field.
    :param amplitude: Maximum value of the transformed scalar field.
    :param width: Range at which the value of the transformed scalar field drops to almost zero.
    :return: Transformed scalar field.
    """
    e = np.exp(- 4 * u / width)

    return amplitude * np.minimum(e, 1)


def hard_binarization(u: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binarizes the Signed Distance field/pattern based on a threshold.
    Values below the threshold are 1 and values above are 0.
    :param u: Signed Distance field or any scalar field.
    :param threshold: Binarization threshold.
    :return: Binarized scalar field.
    """
    out = u <= threshold
    out = out.astype(float)
    return out


def linear_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a decreasing linear function to the scalar (Signed Distance Function) field.
    :param u: Signed Distance field or any scalar field.
    :param amplitude: Maximum value of the transformed scalar field.
    :param width: Range at which the value of the transformed scalar field drops to zero.
    :return: Transformed scalar field.
    """
    out = 1 - u/width

    return np.clip(out, 0, 1)*amplitude


def relu(u: np.ndarray, width: float | int = 1) -> np.ndarray:
    """
    Applies the ReLU function to the scalar (Signed Distance Function) field.
    :param u: Signed Distance field or any scalar field.
    :param width: Range at which the value of the transformed field reaches one.
    :return: Transformed scalar field.
    """
    return np.maximum(u/width, 0)


def gaussian_boundary(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies the Gaussian to the scalar (Signed Distance Function) field.
    :param u: Signed Distance field or any scalar field.
    :param amplitude: Maximum value of the transformed scalar field.
    :param width: Range at which the value of the transformed scalar field drops to almost zero.
    :return: Transformed scalar field.
    """
    out = np.exp(-4*(u/width)**2)

    return amplitude*out


def gaussian_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies the Gaussian to the positive values of the scalar (Signed Distance Function) field.
    :param u: Signed Distance field or any scalar field.
    :param amplitude: Maximum value of the transformed scalar field (and points at which the scalar field was < 0).
    :param width: Range at which the value of the transformed scalar field drops to almost zero.
    :return: Transformed scalar field.
    """
    u = np.maximum(u, 0)
    out = np.exp(-4*(u/width)**2)

    return amplitude*out


def conv_averaging(u: np.ndarray, kernel_size: int | tuple | list | np.ndarray, iterations: int) -> np.ndarray:
    """
    Averages the field using an averaging convolutional kernel of the specified size.
    :param u: Signed Distance field or any scalar field.
    :param kernel_size: Size of the averaging kernel. Must be an integer or a tuple/array of the
     same dimension as the scaler field.
    :param iterations: Number of times the convolutional averaging is applied to the input scalar field.
    :return: Transformed scalar field.
    """
    if iterations == 0:
        return u

    shape_u = u.shape

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)*len(shape_u)
    if isinstance(kernel_size, np.ndarray):
        kernel_size = tuple(kernel_size)
    if isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)

    if not len(kernel_size) == len(shape_u):
        raise ValueError("Dimension of the kernel and the field must match!")

    if len(kernel_size)==2:
        norm_ = kernel_size[0]*kernel_size[1]
    if len(kernel_size) == 3:
        norm_ = kernel_size[0] * kernel_size[1] * kernel_size[2]

    filter_ = np.ones(kernel_size) / norm_

    new = convolve(u, filter_)
    for i in range(iterations):
        new = convolve(new, filter_)

    return new


def conv_edge_detection(u: np.ndarray) -> np.ndarray:
    """
    Edge detection based ona 3x3 convolutional kernel.
    :param u: Signed Distance field or any scalar field.
    :return: Transformed scalar field.
    """

    if len(u.shape)==2:
        filter_ = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    if len(u.shape) == 3:
        f = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filter_ = np.zeros((3,3,1))
        filter_[:,:,:] = np.asarray([f]).T

    new = convolve(u, filter_)

    return new

