# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.lax import fori_loop
import jax
from functools import partial


array_like_type = np.ndarray | jnp.ndarray
scalar_like_type = float | int


@jax.jit
def sigmoid_falloff_jax(u: array_like_type, amplitude: scalar_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Applies a sigmoid to the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Width of the sigmoid.

    Returns:
        Transformed scalar field.
    """
    e = jnp.exp(4 * u / width)

    return amplitude * (1 / (1 + e))


@jax.jit
def positive_sigmoid_falloff_jax(u: array_like_type, amplitude: scalar_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Applies a sigmoid, shifted to the positive values by the value of the width parameter,
    to the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Width of the sigmoid.

    Returns:
        Transformed scalar field.
    """
    e = jnp.exp(4 * (u - width) / width)

    return amplitude * (1 / (1 + e))


@jax.jit
def capped_exponential_jax(u: array_like_type, amplitude: scalar_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Applies a decreasing exponential functon to the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to almost zero.

    Returns:
        Transformed scalar field.
    """
    e = jnp.exp(- 4 * u / width)

    return amplitude * jnp.minimum(e, 1)


@jax.jit
def hard_binarization_jax(u: array_like_type, threshold: scalar_like_type) -> jnp.ndarray:
    """
    Binarizes the Signed Distance field/pattern based on a threshold.
    Values below the threshold are 1 and values above are 0.

    Args:
        u: Signed Distance field or any scalar field.
        threshold: Binarization threshold.

    Returns:
        Binarized scalar field.
    """
    out = jnp.where(u <= threshold, 1, 0)
    return out


@jax.jit
def linear_falloff_jax(u: array_like_type, amplitude: scalar_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Applies a decreasing linear function to the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to zero.

    Returns:
        Transformed scalar field.
    """
    out = 1 - u / width

    return jnp.clip(out, 0, 1) * amplitude


@jax.jit
def relu_jax(u: array_like_type, width: scalar_like_type = 1) -> jnp.ndarray:
    """
    Applies the ReLU function to the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        width: Range at which the value of the transformed field reaches one.

    Returns:
        Transformed scalar field.
    """
    return jnp.maximum(u / width, 0)


@jax.jit
def smooth_relu_jax(u: array_like_type, smooth_width: scalar_like_type,
                    width: scalar_like_type = 1, threshold: scalar_like_type = 0.01) -> jnp.ndarray:
    """
    Applies the "squareplus" function to the scalar (Signed Distance Function) field.
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    Args:
        u: Signed Distance field or any scalar field.
        smooth_width: Distance from the origin at which the Smooth ReLU function
            is greater than ReLU for less than the value of the threshold parameter.
        width: Range at which the value of the transformed field reaches one.
        threshold: At smooth_width distance from the origin the value of the Smooth ReLU function is greater
            than ReLU for the value of the threshold parameter.
            at smooth_width distance from the origin.

    Returns:
        Transformed scalar field.
    """
    b = (smooth_width + threshold) * 4 * threshold
    v = u / width
    return (v + jnp.sqrt(v ** 2 + b)) / 2


@partial(jax.jit, static_argnames=['ground'])
def slowstart_jax(u: array_like_type,
                  smooth_width: scalar_like_type,
                  width: scalar_like_type = 1, threshold: scalar_like_type = 0.01,
                  ground: bool = True) -> jnp.ndarray:
    """
    Applies the SlowStart function to the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        smooth_width: Distance from the origin at which the SlowStart function
            is greater than ReLU for less than the value of the threshold parameter.
        width: Range at which the value of the transformed field reaches one.
        threshold: At smooth_width distance from the origin the value of the SlowStart function is greater
            than ReLU for the value of the threshold parameter.
        ground: if True the value of the function is zero at zero.

    Returns:
        Transformed scalar field.
    """
    b = (2 * smooth_width + threshold) * threshold
    return jnp.sqrt(jnp.maximum(u / width, 0) ** 2 + b / width) - jnp.sqrt(b / width) * ground


@jax.jit
def gaussian_boundary_jax(u: array_like_type, amplitude: scalar_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Applies the Gaussian to the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to almost zero.

    Returns:
        Transformed scalar field.
    """
    out = jnp.exp(-4 * (u / width) ** 2)

    return amplitude * out


@jax.jit
def gaussian_falloff_jax(u: array_like_type, amplitude: scalar_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Applies the Gaussian to the positive values of the scalar (Signed Distance Function) field.

    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field (and points at which the scalar field was < 0).
        width: Range at which the value of the transformed scalar field drops to almost zero.

    Returns:
        Transformed scalar field.
    """
    u = jnp.maximum(u, 0)
    out = jnp.exp(-4 * (u / width) ** 2)

    return amplitude * out


@jax.jit
def conv_multiple_jax(u: array_like_type,
                      filter_kernel: array_like_type,
                      iterations: int) -> jnp.ndarray:
    """
    Averages the field using an averaging convolutional kernel of the specified size.

    Args:
        u: Signed Distance field or any scalar field.
        filter_kernel: Convolutional kernel of appropriate dimensions.
        iterations: Number of times the convolutional averaging is applied to the input scalar field.

    Returns:
        Transformed scalar field.
    """

    new = convolve(u, filter_kernel, mode="same")

    def body(i, a):
        return convolve(a, filter_kernel, mode="same")
    new = fori_loop(0, iterations - 1, body, new)

    return new


@jax.jit
def conv_edge_detection_jax(u: array_like_type,) -> jnp.ndarray:
    """
    Edge detection with a 3x3 convolutional kernel.

    Args:
        u: Signed Distance field or any scalar field.

    Returns:
        Transformed scalar field.
    """
    if len(u.shape) == 2:
        filter_ = jnp.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    if len(u.shape) == 3:
        f = jnp.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filter_ = jnp.zeros((3, 3, 1))
        filter_.at[:, :, :].set(jnp.asarray([f]).T)

    new = convolve(u, filter_, mode="same")

    return new
