# Copyright (C) 2026 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np
import jax

from spomso.jax_cores.helper_functions import smarter_reshape

array_like_type = jnp.ndarray | np.ndarray | list | tuple
scalar_like_type = float | int

# ----------------------------------------------------------------------------------------------------------------------
# VECTOR MODIFICATION FUNCTIONS


@jax.jit
def batch_normalize(vec: array_like_type) -> jnp.ndarray:
    """
    Normalize a batch of vectors along axis 0. Zero-length vectors are left as zeros
    to avoid division by zero.

    Args:
        vec: Array of vectors with shape (D, N), where D is the vector dimension and
            N is the number of vectors.

    Returns:
        Array with shape (D, N) where each column has unit norm (or is zero).
    """
    m = jnp.linalg.norm(vec, axis=0)
    safe_m = jnp.where(m == 0, 1.0, m)
    return vec / safe_m


@jax.jit
def add_vectors(vec: array_like_type, add_vec: array_like_type) -> jnp.ndarray:
    """
    Add a vector (field) from vector field.

    Args:
        vec: Array of vectors with shape (D, N), where D is the vector dimension and
            N is the number of vectors.
        add_vec: Either a single vector with shape (D,) or an array with the same shape (D, N) as `vec`.

    Returns:
        Array with shape (D, N).
    """
    add_vec = jnp.asarray(add_vec)
    if add_vec.size == 3:
        return jnp.add(vec.T, add_vec).T
    else:
        return jnp.add(vec, add_vec)


@jax.jit
def subtract_vectors(vec: array_like_type, subtract_vec: array_like_type) -> jnp.ndarray:
    """
    Subtract a vector (field) from vector field.

    Args:
        vec: Array of vectors with shape (D, N), where D is the vector dimension and
            N is the number of vectors.
        subtract_vec: Either a single vector with shape (D,) or an array with the same shape (D, N) as `vec`.

    Returns:
        Array with shape (D, N).
    """
    subtract_vec = jnp.asarray(subtract_vec)
    if subtract_vec.size == 3:
        return jnp.subtract(vec.T, subtract_vec).T
    else:
        return jnp.subtract(vec, subtract_vec)


@jax.jit
def rescale_vectors(vec: array_like_type, scale: scalar_like_type | array_like_type) -> jnp.ndarray:
    """
    Multiply the vector field by a scalar, or scale each vector per-point by the scale array.

    Args:
        vec: Array of vectors with shape (D, N), where D is the vector dimension and
            N is the number of vectors.
        scale: Scalar multiplier applied to every vector, or an array of shape (N,)
            with a per-point scale factor.

    Returns:
        Array with shape (D, N).
    """
    return jnp.multiply(vec, scale)


@jax.jit
def rotate_vectors_phi(vec: array_like_type, phis: scalar_like_type | array_like_type) -> jnp.ndarray:
    """
    Rotate each vector by an angle `phi` about the z-axis.

    Args:
        vec: Array of vectors with shape (3, N), where N is the number of vectors.
        phis: Rotation angle in radians. Either a scalar or an array of shape (N,) with a per-point angle.

    Returns:
        Array with shape (3, N) containing the rotated vectors.
    """
    sa = jnp.sin(phis)
    ca = jnp.cos(phis)

    o = jnp.asarray([vec[0, :] * ca - vec[1, :] * sa,
                     vec[0, :] * sa + vec[1, :] * ca,
                     vec[2, :]])

    return o


@jax.jit
def rotate_vectors_theta(vec: array_like_type, thetas: scalar_like_type | array_like_type) -> jnp.ndarray:
    """
    Rotate each vector by an angle `theta` about the axis perpendicular to both the
    vector and the z-axis. This corresponds to tilting each vector out of its own
    xy-projection direction.

    Args:
        vec: Array of vectors with shape (3, N), where N is the number of vectors.
        thetas: Rotation angle in radians. Either a scalar (same rotation for every
            vector) or an array of shape (N,) with a per-point angle.

    Returns:
        Array with shape (3, N) containing the rotated vectors.
    """
    rvec = vec.at[2].set(0)
    rvec = batch_normalize(rvec)

    ca = jnp.cos(thetas)
    sa = jnp.sin(thetas)

    term2 = jnp.asarray([rvec[0] * vec[2],
                         rvec[1] * vec[2],
                         -rvec[0] * vec[0] - rvec[1] * vec[1]])
    out = vec * ca + term2 * sa
    return out


@jax.jit
def rotate_vectors_x_axis(vec: array_like_type, alpha: scalar_like_type | array_like_type) -> jnp.ndarray:
    """
    Rotate each vector by an angle `alpha` about the x-axis. The x-component is
    unchanged.

    Args:
        vec: Array of vectors with shape (3, N), where N is the number of vectors.
        alpha: Rotation angle in radians. Either a scalar (same rotation for every
            vector) or an array of shape (N,) with a per-point angle.

    Returns:
        Array with shape (3, N) containing the rotated vectors.
    """
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    o = jnp.asarray([vec[0, :],
                     vec[1, :] * ca - vec[2, :] * sa,
                     vec[1, :] * sa + vec[2, :] * ca,
                     ])

    return o


@jax.jit
def rotate_vectors_y_axis(vec: array_like_type, alpha: scalar_like_type | array_like_type) -> jnp.ndarray:
    """
    Rotate each vector by an angle `alpha` about the y-axis. The y-component is
    unchanged.

    Args:
        vec: Array of vectors with shape (3, N), where N is the number of vectors.
        alpha: Rotation angle in radians. Either a scalar (same rotation for every
            vector) or an array of shape (N,) with a per-point angle.

    Returns:
        Array with shape (3, N) containing the rotated vectors.
    """
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    o = jnp.asarray([vec[0, :] * ca - vec[2, :] * sa,
                     vec[1, :],
                     vec[0, :] * sa + vec[2, :] * ca,
                     ])

    return o


@jax.jit
def rotate_vectors_z_axis(vec: array_like_type, alpha: scalar_like_type | array_like_type) -> jnp.ndarray:
    """
    Rotate each vector by an angle `alpha` about the z-axis. The z-component is
    unchanged. Equivalent to `rotate_vectors_phi` but named for consistency with the
    x-axis and y-axis rotation functions.

    Args:
        vec: Array of vectors with shape (3, N), where N is the number of vectors.
        alpha: Rotation angle in radians. Either a scalar (same rotation for every
            vector) or an array of shape (N,) with a per-point angle.

    Returns:
        Array with shape (3, N) containing the rotated vectors.
    """
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    o = jnp.asarray([vec[0, :] * ca - vec[1, :] * sa,
                     vec[0, :] * sa + vec[1, :] * ca,
                     vec[2, :]])

    return o


@jax.jit
def rotate_vectors_axis(vec: array_like_type, axes: array_like_type,
                        alpha: scalar_like_type | array_like_type) -> jnp.ndarray:
    """
    Rotate each vector by an angle `alpha` about an arbitrary axis using the
    Rodrigues rotation formula.

    Args:
        vec: Array of vectors with shape (3, N), where N is the number of vectors.
        axes: Rotation axis. Either a single unit vector with shape (3,) — same axis
            for every rotation — or an array of shape (3, N) with a per-point axis.
        alpha: Rotation angle in radians. Either a scalar (same rotation for every
            vector) or an array of shape (N,) with a per-point angle.

    Returns:
        Array with shape (3, N) containing the rotated vectors.
    """
    axes = jnp.asarray(axes)
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    t1 = vec * ca
    t2 = sa * jnp.cross(axes.T, vec.T).T
    if axes.size == 3:
        t3 = (1 - ca) * jnp.outer(axes, (jnp.sum(jnp.multiply(vec.T, axes.T).T, axis=0)))
    else:
        t3 = (1 - ca) * axes * (jnp.sum(jnp.multiply(vec.T, axes.T).T, axis=0))
    o = t1 + t2 + t3

    return o


@jax.jit
def revolve_field_x(r: array_like_type, vec: array_like_type) -> jnp.ndarray:
    """
    Revolve a vector field about the x-axis. At each point `r`, the vector is
    rotated by the azimuthal angle `arctan2(z, y)` about the x-axis, so that a
    field defined on the xy-plane is extended radially around the x-axis.

    Args:
        r: Coordinates of points at which the vector field is evaluated. Shape must
            be (3, N), where N is the number of coordinate points.
        vec: Array of vectors with shape (3, N) to be revolved.

    Returns:
        Array with shape (3, N) containing the revolved vector field.
    """
    alpha = jnp.arctan2(r[2], r[1])
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    o = jnp.zeros(r.shape)
    o = o.at[0].set(vec[0, :])
    o = o.at[1].set(vec[1, :] * ca - vec[2, :] * sa)
    o = o.at[2].set(vec[1, :] * sa + vec[2, :] * ca)

    return o


@jax.jit
def revolve_field_y(r: array_like_type, vec: array_like_type) -> jnp.ndarray:
    """
    Revolve a vector field about the y-axis. At each point `r`, the vector is
    rotated by the azimuthal angle `arctan2(z, x)` about the y-axis, so that a
    field defined on the xy-plane is extended radially around the y-axis.

    Args:
        r: Coordinates of points at which the vector field is evaluated. Shape must
            be (3, N), where N is the number of coordinate points.
        vec: Array of vectors with shape (3, N) to be revolved.

    Returns:
        Array with shape (3, N) containing the revolved vector field.
    """
    alpha = jnp.arctan2(r[2], r[0])
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    o = jnp.zeros(r.shape)
    o = o.at[0].set(vec[0, :] * ca - vec[2, :] * sa)
    o = o.at[1].set(vec[1, :])
    o = o.at[2].set(vec[0, :] * sa + vec[2, :] * ca)

    return o


@jax.jit
def revolve_field_z(r: array_like_type, vec: array_like_type) -> jnp.ndarray:
    """
    Revolve a vector field about the z-axis. At each point `r`, the vector is
    rotated by the azimuthal angle `arctan2(y, x)` about the z-axis, so that a
    field defined on the xz-plane is extended radially around the z-axis.

    Args:
        r: Coordinates of points at which the vector field is evaluated. Shape must
            be (3, N), where N is the number of coordinate points.
        vec: Array of vectors with shape (3, N) to be revolved.

    Returns:
        Array with shape (3, N) containing the revolved vector field.
    """
    alpha = jnp.arctan2(r[1], r[0])
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)

    o = jnp.zeros(r.shape)
    o = o.at[0].set(vec[0, :] * ca - vec[1, :] * sa)
    o = o.at[1].set(vec[0, :] * sa + vec[1, :] * ca)
    o = o.at[2].set(vec[2, :])

    return o