# Copyright (C) 2026 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np
import jax

from spomso.jax_cores.helper_functions import smarter_reshape
from spomso.jax_cores.vector_modifications_jax import batch_normalize

array_like_type = jnp.ndarray | np.ndarray | list | tuple
scalar_like_type = float | int

# ----------------------------------------------------------------------------------------------------------------------
# VECTOR INITIALIZATION FUNCTIONS


@jax.jit
def cartesian_vector_field(p: array_like_type) -> jnp.ndarray:
    """
    Vector field defined by its components in the cartesian coordinate system (x, y, z).
    The components of the output vector field are cartesian (x, y, z).
    Args:
        p: Array of Cartesian components with shape (3, N), where N is the number of
            coordinate points. Rows are (u_x, u_y, u_z).

    Returns:
        Vector field with shape (3, N).
    """
    ux = p[0]
    uy = p[1]
    uz = p[2]
    return jnp.asarray((ux, uy, uz))


@jax.jit
def spherical_vector_field(p: array_like_type) -> jnp.ndarray:
    """
    Vector field defined by its components in the spherical coordinate system (r, phi, theta).
    The components of the output vector field are cartesian (x, y, z).

    Args:
        p: Array of spherical components with shape (3, N), where N is the number of
            coordinate points. Rows are (r, phi, theta) — radius, azimuthal angle,
            polar angle (in radians).

    Returns:
        Vector field with shape (3, N), in Cartesian coordinates.
    """
    r = p[0]
    phi = p[1]
    theta = p[2]

    u = r * jnp.cos(phi) * jnp.sin(theta)
    v = r * jnp.sin(phi) * jnp.sin(theta)
    w = r * jnp.cos(theta)

    return jnp.asarray((u, v, w))


@jax.jit
def cylindrical_vector_field(p: array_like_type) -> jnp.ndarray:
    """
    Vector field defined by its components in the cylindrical coordinate system (r, phi, z).
    The components of the output vector field are cartesian (x, y, z).

    Args:
        p: Array of cylindrical components with shape (3, N), where N is the number of
            coordinate points. Rows are (r, phi, z) — radius, azimuthal angle (in
            radians), and z-component.

    Returns:
        Vector field with shape (3, N), in Cartesian coordinates.
    """
    r = p[0]
    phi = p[1]
    z = p[2]

    u = r * jnp.cos(phi)
    v = r * jnp.sin(phi)

    return jnp.asarray((u, v, z))


@jax.jit
def radial_vector_field_spherical(r: array_like_type) -> jnp.ndarray:
    """
    Vector field where all the vectors are pointing radially outwards from the origin.
    Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
    The components of the output vector field are cartesian (x, y, z).

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.

    Returns:
        Vector field with shape (3, N), unit-normalized.
    """
    return batch_normalize(r)


@jax.jit
def radial_vector_field_cylindrical(r: array_like_type) -> jnp.ndarray:
    """
    Vector field where all the vectors are pointing radially outwards from the line x=0, y=0 (z-axis).
    Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
    The components of the output vector field are cartesian (x, y, z).

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.
        p: Placeholder for extra parameters (unused).

    Returns:
        Vector field with shape (3, N), unit-normalized.
    """
    r = r.at[2].set(0.0)
    return batch_normalize(r)


@jax.jit
def hyperbolic_vector_field_cylindrical(r: array_like_type) -> jnp.ndarray:
    """
    Hyperbolic vector field centered at the line x=0, y=0 (z-axis).
    Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
    The components of the output vector field are cartesian (x, y, z).

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.

    Returns:
        Vector field with shape (3, N).
    """
    z = jnp.zeros(r.shape[1])
    alpha = -jnp.arctan2(r[1], r[0])
    u = jnp.cos(alpha)
    v = jnp.sin(alpha)
    return jnp.asarray((u, v, z))


@jax.jit
def awn_vector_field_cylindrical(r: array_like_type, gamma: scalar_like_type) -> jnp.ndarray:
    """
    Azimuthal winding vector field on the cylinder: the in-plane direction is
    `gamma * arctan2(y, x)`. The z-component is zero.

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.
        gamma: Winding number multiplier.

    Returns:
        Vector field with shape (3, N).
    """
    z = jnp.zeros(r.shape[1])
    alpha = gamma * jnp.arctan2(r[1], r[0])
    u = jnp.cos(alpha)
    v = jnp.sin(alpha)
    return jnp.asarray((u, v, z))


@jax.jit
def vortex_vector_field_cylindrical(r: array_like_type) -> jnp.ndarray:
    """
    Cylindrical vortex vector field: at each point, the vector is tangent to a circle
    in the xy-plane centered on the z-axis. The z-component is zero.

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.

    Returns:
        Vector field with shape (3, N), unit-normalized.
    """
    r = r.at[2].set(0.0)
    v = batch_normalize(r)
    return jnp.asarray((-v[1], v[0], v[2]))


@jax.jit
def aar_vector_field_cylindrical(r: array_like_type, alpha: scalar_like_type) -> jnp.ndarray:
    """
    Axially aligned radial vector field on the cylinder: cylindrical radial vector
    field rotated by a fixed angle `alpha` about the z-axis.

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.
        alpha: Rotation angle in radians.

    Returns:
        Vector field with shape (3, N).
    """
    r = r.at[2].set(0.0)
    vec = batch_normalize(r)
    sa = jnp.squeeze(jnp.sin(alpha))
    ca = jnp.squeeze(jnp.cos(alpha))

    return jnp.asarray([
        vec[0, :] * ca - vec[1, :] * sa,
        vec[0, :] * sa + vec[1, :] * ca,
        vec[2, :],
    ])


@jax.jit
def aav_vector_field_cylindrical(r: array_like_type, alpha: scalar_like_type) -> jnp.ndarray:
    """
    Axially aligned vortex vector field on the cylinder: cylindrical vortex vector
    field rotated by a fixed angle `alpha` about the z-axis.

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.
        alpha: Rotation angle in radians.

    Returns:
        Vector field with shape (3, N).
    """
    r = r.at[2].set(0.0)
    vec = batch_normalize(r)
    sa = jnp.squeeze(jnp.sin(alpha))
    ca = jnp.squeeze(jnp.cos(alpha))

    return jnp.asarray([
        -vec[0, :] * sa - vec[1, :] * ca,
        vec[0, :] * ca - vec[1, :] * sa,
        vec[2, :],
    ])


@jax.jit
def x_vector_field(r: array_like_type) -> jnp.ndarray:
    """
    Vector field where only the X component (cartesian coordinates) is non-zero.
    Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
    The components of the output vector field are cartesian (x, y, z).

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.

    Returns:
        Vector field with shape (3, N), each column equal to (1, 0, 0).
    """
    vec = jnp.zeros(r.shape)
    return vec.at[0].set(1.0)


@jax.jit
def y_vector_field(r: array_like_type) -> jnp.ndarray:
    """
    Vector field where only the Y component (cartesian coordinates) is non-zero.
    Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
    The components of the output vector field are cartesian (x, y, z).

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.

    Returns:
        Vector field with shape (3, N), each column equal to (0, 1, 0).
    """
    vec = jnp.zeros(r.shape)
    return vec.at[1].set(1.0)


@jax.jit
def z_vector_field(r: array_like_type) -> jnp.ndarray:
    """
    Vector field where only the Z component (cartesian coordinates) is non-zero.
    Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
    The components of the output vector field are cartesian (x, y, z).

    Args:
        r: Coordinates of points on which the vector field is evaluated.
            Shape must be (D, N), where D is the dimension (D = 3) and N is the
            number of coordinate points.

    Returns:
        Vector field with shape (3, N), each column equal to (0, 0, 1).
    """
    vec = jnp.zeros(r.shape)
    return vec.at[2].set(1.0)


def from_sdf(sdf_: array_like_type,
             co_resolution: tuple | list | array_like_type) -> jnp.ndarray:
    """
    Vector field constructed from an SDF.
    Point cloud specifying the value of the SDF is taken as an input.
    The components of the output vector field are cartesian (x, y, z).

    Args:
        sdf_: Signed distance field evaluated on a rectilinear grid, flattened to
            shape (N,), where N is the total number of points.
        co_resolution: Number of points along each axis in the grid on which the SDF is evaluated.

    Returns:
        Vector field with shape (D, N), unit-normalized. D is the number of
        dimensions inferred from `co_resolution`.
    """
    dimensions = jnp.asarray(co_resolution).shape[0]
    gsdf = smarter_reshape(sdf_, co_resolution)

    vec = jnp.asarray(jnp.gradient(gsdf))
    vec = vec.reshape(dimensions, -1)

    return batch_normalize(vec)