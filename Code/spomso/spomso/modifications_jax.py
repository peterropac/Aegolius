# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable

from spomso.jax_cores.post_processing_jax import sigmoid_falloff_jax, positive_sigmoid_falloff_jax
from spomso.jax_cores.post_processing_jax import capped_exponential_jax
from spomso.jax_cores.post_processing_jax import linear_falloff_jax
from spomso.jax_cores.post_processing_jax import relu_jax, smooth_relu_jax, slowstart_jax
from spomso.jax_cores.post_processing_jax import hard_binarization_jax
from spomso.jax_cores.post_processing_jax import gaussian_boundary_jax, gaussian_falloff_jax
from spomso.jax_cores.post_processing_jax import conv_multiple_jax, conv_edge_detection_jax

from spomso.jax_cores.helper_functions import smarter_reshape


array_like_type = jnp.ndarray | np.ndarray | list | tuple
scalar_like_type = float | int
function_like_type = Callable[[array_like_type, tuple], jnp.ndarray]


def elongation(function_: function_like_type,
               elongate_vector: array_like_type) -> function_like_type:
    """
    Elongates the geometry along a certain vector by the length of the vector in each respective direction.

    Args:
        function_: Original SDF.
        elongate_vector: A vector, with shape (3,), by which the shape is elongated.

    Returns:
        Modified SDF.
    """
    ev = jnp.asarray(elongate_vector)/2

    @jax.jit
    def new_geo_object(co, *params):
        qo0 = co[0] - jnp.clip(co[0], -ev[0] / 2, ev[0] / 2)
        qo1 = co[1] - jnp.clip(co[1], -ev[1] / 2, ev[1] / 2)
        qo2 = co[2] - jnp.clip(co[2], -ev[2] / 2, ev[2] / 2)
        qo = jnp.asarray([qo0, qo1, qo2])
        return function_(qo, *params)

    return new_geo_object


def rounding(function_: function_like_type, rounding_radius: scalar_like_type) -> function_like_type:
    """
    Rounds off the geometry - effectively thickening it by the rounding radius.

    Args:
        function_: Original SDF.
        rounding_radius: Radius by which the edges are rounded and the object thickened.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return function_(co, *params) - rounding_radius

    return new_geo_object


def rounding_cs(function_: function_like_type,
                rounding_radius: scalar_like_type,
                bb_size: scalar_like_type) -> function_like_type:
    """
    Rounds off the geometry, but the geometry will be contained in its bounding box.
    Bounding box size must be specified.

    Args:
        function_: Original SDF.
        rounding_radius: Radius by which the edges are rounded.
        bb_size: Size of the bounding box. Typically, the largest dimension of the object.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        scale = 1 - 2*rounding_radius/bb_size + 1e-8
        scale = jnp.maximum(scale, 1e-8)
        return scale * function_(co / scale, *params) - rounding_radius

    return new_geo_object


def boundary(function_: function_like_type) -> function_like_type:
    """
    Get the boundary of the SDF.

    Args:
        function_: Original SDF.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return jnp.abs(function_(co, *params))

    return new_geo_object


def invert(function_: function_like_type) -> function_like_type:
    """
    Inverts the sign of the SDF.

    Args:
        function_: Original SDF.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return -function_(co, *params)

    return new_geo_object


def sign(function_: function_like_type) -> function_like_type:
    """
    Get the sign of the SDF.

    Args:
        function_: Original SDF.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return jnp.sign(function_(co, *params))

    return new_geo_object


def define_volume(function_: function_like_type,
                  interior: function_like_type,
                  interior_parameters: tuple) -> function_like_type:
    """
    Defines the interior of the SDF if with a function.

    Args:
        function_: Original SDF.
        interior: Function defining the interior (-1) and exterior (1) of an SDF.
        interior_parameters: Parameters of the function defining the interior of the SDF.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return function_(co, *params) * interior(co, *interior_parameters)

    return new_geo_object


def onion(function_: function_like_type, thickness: scalar_like_type) -> function_like_type:
    """
    Transforms the geometry into a surface with some thickness.

    Args:
        function_: Original SDF.
        thickness: Thickness of the resulting shape.

    Returns:
        Modified SDF.
    """
    @jax.jit
    def new_geo_object(co, *params):
        return jnp.abs(function_(co, *params)) - thickness

    return new_geo_object


def concentric(function_: function_like_type, width: scalar_like_type) -> function_like_type:
    """
    Transforms an isosurface into two concentric isosurfaces which are apart by the value of width parameter.
    Transforms a volume into an isosurface and rounds it by width/2.

    Args:
        function_: Original SDF.
        width: Thickness of the resulting shape.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return jnp.abs(function_(co, *params) - width / 2)

    return new_geo_object


def revolution(function_: function_like_type, radius: scalar_like_type) -> function_like_type:
    """
    Revolves a 2D shape around the y-axis to generate a 3D shape.
    First the 2D shape is translated along the x-axis by the radius of revolution,
    then it is revolved around the y-axis.

    Args:
        function_: Original SDF.
        radius: Radius of revolution.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        xz = jnp.asarray([co[0, :], co[2, :]])
        mag_xz = jnp.linalg.norm(xz, axis=0)
        qo = jnp.asarray([mag_xz - radius, co[1], jnp.zeros(co.shape[1])])

        return function_(qo, *params)

    return new_geo_object


def axis_revolution(function_: function_like_type, radius: scalar_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Revolves a 2D shape around an axis to generate a 3D shape.
    First the 2D shape is translated along the x-axis by the radius of revolution,
    and then revolved around the axis of revolution.
    The axis of revolution is angled by the specified angle with respect to the y-axis.

    Args:
        function_: Original SDF.
        radius: Radius of revolution.
        angle: Angle between the axis of revolution and the y-axis.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        rot = jnp.asarray([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        co_xy = rot.dot(co[:2])

        xz = jnp.asarray([co_xy[0, :], co[2, :]])
        mag_xz = jnp.linalg.norm(xz, axis=0)

        qo_xy = jnp.asarray([mag_xz, co_xy[1]])
        qo_xy = qo_xy.at[:2, :].set(rot.T.dot(qo_xy[:2]))

        qo = jnp.asarray([qo_xy[0] - radius, qo_xy[1], jnp.zeros(co.shape[1])])

        return function_(qo, *params)

    return new_geo_object


def extrusion(function_: function_like_type, distance: scalar_like_type) -> function_like_type:
    """
    Extrudes a 2D shape along the z-axis so that the height of the object is equal to the specified distance.

    Args:
        function_: Original SDF.
        distance: Final height of the object - distance of extrusion.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        qo = jnp.asarray(co.copy())
        qo = qo.at[2, :].set(0)
        d = function_(qo, *params)
        w = jnp.asarray((d, jnp.abs(co[2]) - distance / 2))

        first_term = jnp.minimum(jnp.maximum(w[0], w[1]), 0)
        second_term = jnp.linalg.norm(jnp.maximum(w, 0), axis=0)

        return first_term + second_term

    return new_geo_object


def twist(function_: function_like_type, pitch: scalar_like_type) -> function_like_type:
    """
    Twists the geometry around the z-axis based on the position along the z-axis and the pitch.

    Args:
        function_: Original SDF.
        pitch: rad/unit length.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        c = jnp.cos(pitch * co[2])
        s = jnp.sin(pitch * co[2])
        rot = jnp.asarray([[c, s], [-s, c]])

        qo_xy = rot[0, :, :] * co[0, :] + rot[1, :, :] * co[1, :]
        qo = jnp.concatenate([qo_xy, jnp.expand_dims(co[2], axis=0)], axis=0)

        return function_(qo, *params)

    return new_geo_object


def bend(function_: function_like_type, radius: scalar_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Bends the geometry around the z-axis, based on the specified bending radius and angle.
    The length of the bent section is radius*angle.

    Args:
        function_: Original SDF.
        radius: Bending radius.
        angle: Bending angle.

    Returns:
        Modified SDF.
    """
    @jax.jit
    def new_geo_object(co, *params):
        c = jnp.cos(angle / 2)
        s = jnp.sin(angle / 2)
        rot = jnp.asarray([[c, s], [-s, c]])

        qo = jnp.asarray(co.copy())
        qo = qo.at[1, :].add(-radius)
        phi = jnp.arctan2(qo[0], -qo[1])
        qo = qo.at[1, :].set(-radius + jnp.linalg.norm(qo[:2], axis=0))
        qo = qo.at[0, :].set(radius*phi)

        mask1 = radius * angle / 2 <= jnp.abs(qo[0])
        mask2 = co[0]*mask1 >= 0

        wo = jnp.asarray(co[:2].copy())

        sign = jnp.sign(co[0])

        wo = wo.at[0].add(-radius * s * sign)
        wo = wo.at[1].add(-radius * (1 - c))

        wo = jnp.where(mask2, rot.dot(wo[:2]), rot.T.dot(wo[:2]))

        wo = wo.at[0, :].add(radius * (angle / 2) * sign)

        wo = jnp.where(mask1, wo, qo[:2])

        qo = qo.at[:2].set(wo)

        return function_(qo, *params)

    return new_geo_object


def shear_xz(function_: function_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Displaces the points parallel to the x-axis of the coordinate system around the z-axis by the specified angle.

    Args:
        function_: Original SDF.
        angle: Shearing angle - the shear factor is given as tan(angle).

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        t = jnp.tan(angle)
        o = jnp.asarray([[1, 0, 0], [-t, 1, 0], [0, 0, 1]]).T
        qo = o.dot(jnp.asarray(co.copy()))

        return function_(qo, *params)

    return new_geo_object


def shear_yz(function_: function_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Displaces the points parallel to the y-axis of the coordinate system around the z-axis by the specified angle.

    Args:
        function_: Original SDF.
        angle: Shearing angle - the shear factor is given as tan(angle).

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        t = jnp.tan(angle)
        o = jnp.asarray([[1, 0, 0], [-t, 1, 0], [0, 0, 1]])
        qo = o.dot(jnp.asarray(co.copy()))

        return function_(qo, *params)

    return new_geo_object


def shear_xy(function_: function_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Displaces the points parallel to the x-axis of the coordinate system around the y-axis by the specified angle.

    Args:
        function_: Original SDF.
        angle: Shearing angle - the shear factor is given as tan(angle).

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        t = jnp.tan(angle)
        o = jnp.asarray([[1, 0, 0], [0, 1, 0], [-t, 0, 1]]).T
        qo = o.dot(jnp.asarray(co.copy()))

        return function_(qo, *params)

    return new_geo_object


def shear_zy(function_: function_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Displaces the points parallel to the z-axis of the coordinate system around the y-axis by the specified angle.

    Args:
        function_: Original SDF.
        angle: Shearing angle - the shear factor is given as tan(angle).

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        t = jnp.tan(angle)
        o = jnp.asarray([[1, 0, 0], [0, 1, 0], [-t, 0, 1]])
        qo = o.dot(jnp.asarray(co.copy()))

        return function_(qo, *params)

    return new_geo_object


def shear_yx(function_: function_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Displaces the points parallel to the y-axis of the coordinate system around the x-axis by the specified angle.

    Args:
        function_: Original SDF.
        angle: Shearing angle - the shear factor is given as tan(angle).

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        t = jnp.tan(angle)
        o = jnp.asarray([[1, 0, 0], [0, 1, 0], [0, -t, 1]]).T
        qo = o.dot(jnp.asarray(co.copy()))

        return function_(qo, *params)

    return new_geo_object


def shear_zx(function_: function_like_type, angle: scalar_like_type) -> function_like_type:
    """
    Displaces the points parallel to the z-axis of the coordinate system around the x-axis by the specified angle.

    Args:
        function_: Original SDF.
        angle: Shearing angle - the shear factor is given as tan(angle).

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        t = jnp.tan(angle)
        o = jnp.asarray([[1, 0, 0], [0, 1, 0], [0, -t, 1]])
        qo = o.dot(jnp.asarray(co.copy()))

        return function_(qo, *params)

    return new_geo_object


def displacement(function_: function_like_type,
                 displacement_function: function_like_type,
                 displacement_function_parameters: tuple) -> function_like_type:
    """
    Displaces the surface of an object based on the values of the displacement function. Should be applied last.

    Args:
        function_: Original SDF.
        displacement_function:
            Function which takes an array of coordinates with shape (D, N) and a tuple of parameters.
            D - number of dimensions (2 or 3);
            N - number of points in the point cloud.
        displacement_function_parameters: Parameters of the displacement function.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return function_(co, *params) + displacement_function(co, *displacement_function_parameters)

    return new_geo_object


def infinite_repetition(function_: function_like_type,
                        distances: array_like_type) -> function_like_type:
    """
    Infinitely repeats geometry in space on a cubic lattice.

    Args:
        function_: Original SDF.
        distances: A vector, with shape (3,), determining the distances between instances along each axis.

    Returns:
        Modified SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        d = jnp.asarray(distances)
        rd = d / 2
        qo = jnp.mod(jnp.add(co.T, rd), d) - rd

        return function_(qo.T, *params)

    return new_geo_object


def finite_repetition(function_: function_like_type,
                      size: array_like_type,
                      repetitions: array_like_type) -> function_like_type:
    """
    Repeats the geometry finite amount of times along each axis within a defined bounding box.

    Args:
        function_: Original SDF.
        size: Size of the bounding box along each axis in which the geometry is repeated.
        repetitions: Number of repetitions along each axis inside the bounding box.

    Returns:
        Modified SDF.
    """

    size = jnp.asarray(size)
    rep = jnp.asarray(repetitions)

    @jax.jit
    def new_geo_object(co, *params):
        c = size * (1 - 1 / rep) / 2
        d = size * (1 / 2 - 1 / rep)
        s = size / rep
        mx = (co[0, :] >= -d[0]) * (co[0, :] <= d[0])
        my = (co[1, :] >= -d[1]) * (co[1, :] <= d[1])
        mz = (co[2, :] >= -d[2]) * (co[2, :] <= d[2])

        # outer
        v = jnp.abs(co)
        v = jnp.subtract(v.T, c)
        v = jnp.asarray([jnp.where(co[0, :] < 0, -v[:, 0], v[:, 0]),
                         jnp.where(co[1, :] < 0, -v[:, 1], v[:, 1]),
                         jnp.where(co[2, :] < 0, -v[:, 2], v[:, 2])
                         ])
        v = v.T

        # inner
        u = jnp.subtract(co.T, d)
        u = jnp.mod(u, s) - s / 2
        v = jnp.asarray([jnp.where(mx, u[:, 0], v[:, 0]),
                         jnp.where(my, u[:, 1], v[:, 1]),
                         jnp.where(mz, u[:, 2], v[:, 2])
                         ])

        return function_(v, *params)

    return new_geo_object


def finite_repetition_rescaled(function_: function_like_type,
                               size: tuple | list | np.ndarray,
                               repetitions: tuple | list | np.ndarray,
                               instance_size: tuple | list | np.ndarray,
                               padding: tuple | list | np.ndarray) -> function_like_type:
    """
    Repeats the geometry finite amount of times along each axis within a defined bounding box.
    The geometry is rescaled based on the provided geometry bounding box size and padding along each axis.
    Args:
        function_: Original SDF.
        size: Size of the bounding box along each axis in which the geometry is repeated.
        repetitions: Number of repetitions along each axis inside the bounding box.
        instance_size: Size of the bounding box around one instance of geometry along each axis.
        padding: Padding around each instance along each axis.

    Returns:
        Modified SDF.
    """

    size = jnp.asarray(size)
    padding = jnp.asarray(padding)
    rep = jnp.asarray(repetitions)
    f = jnp.asarray(instance_size)

    @jax.jit
    def new_geo_object(co, *params):
        c = size * (1 - 1 / rep) / 2
        d = size * (1 / 2 - 1 / rep)
        s = size / rep
        mx = (co[0, :] >= -d[0]) * (co[0, :] <= d[0])
        my = (co[1, :] >= -d[1]) * (co[1, :] <= d[1])
        mz = (co[2, :] >= -d[2]) * (co[2, :] <= d[2])

        # outer
        v = jnp.abs(co)
        v = jnp.subtract(v.T, c)
        v = jnp.asarray([jnp.where(co[0, :] < 0, -v[:, 0], v[:, 0]),
                         jnp.where(co[1, :] < 0, -v[:, 1], v[:, 1]),
                         jnp.where(co[2, :] < 0, -v[:, 2], v[:, 2])
                         ])
        v = v.T

        # inner
        u = jnp.subtract(co.T, d)
        u = jnp.mod(u, s) - s / 2
        v = jnp.asarray([jnp.where(mx, u[:, 0], v[:, 0]),
                         jnp.where(my, u[:, 1], v[:, 1]),
                         jnp.where(mz, u[:, 2], v[:, 2])
                         ])

        sss = jnp.min(s / (f + padding))
        w = v / sss

        return sss * function_(w, *params)

    return new_geo_object


def symmetry(function_: function_like_type, axis: int) -> function_like_type:
    """
    Applies symmetry along an axis of the object.

    Args:
        function_: Original SDF.
        axis: Index of the axis along which the symmetry is applied.

    Returns:
        Modified SDF.
    """
    @jax.jit
    def new_geo_object(co, *params):
        if axis > co.shape[0]:
            return function_(co, *params)
        co = jnp.asarray(co)
        co = co.at[axis, :].apply(jnp.abs)
        return function_(co, *params)

    return new_geo_object


def mirror(function_: function_like_type,
           a: array_like_type,
           b: array_like_type) -> function_like_type:
    """
    Applies mirroring along an axis connecting the two specified points.

    Args:
        function_: Original SDF.
        a: Position of the object's mirror image.
        b: Position of the object.

    Returns:
        Modified SDF.
    """

    a = jnp.asarray(a)
    b = jnp.asarray(b)

    @jax.jit
    def new_geo_object(co, *params):
        w = b - a
        c = (b + a) / 2
        l = jnp.linalg.norm(w)

        x = w / l
        y = jnp.asarray([-x[1], x[0], 0])
        y = y / jnp.linalg.norm(y)
        z = jnp.cross(x, y)
        rot = jnp.asarray([x, y, z])
        co = jnp.subtract(co.T, c).T
        co = rot.dot(co)

        # outer
        v = jnp.asarray([jnp.abs(co[0, :]) - l / 2, co[1, :], co[2, :]])

        return function_(v, *params)

    return new_geo_object


def rotational_symmetry(function_: function_like_type,
                        n: int, radius: scalar_like_type, phase: scalar_like_type) -> function_like_type:
    """
    Applies n-fold rotational symmetry to the object.

    Args:
        function_: Original SDF.
        n: Order of the symmetry.
        radius: By how much the geometry is moved from the centre.
        phase: Angle by which the circular pattern is rotated by.

    Returns:
        Modified SDF.
    """

    angle = 2 * jnp.pi / n

    @jax.jit
    def new_geo_object(co, *params):
        rot = jnp.asarray([[jnp.cos(angle / 2 - phase), jnp.sin(angle / 2 - phase)],
                          [-jnp.sin(angle / 2 - phase), jnp.cos(angle / 2 - phase)]])
        co_xy = rot.dot(co[:2])

        phi = jnp.arctan2(co_xy[1], co_xy[0])
        phi = jnp.where(phi < 0, 2*jnp.pi + phi, phi)

        phi = jnp.mod(phi, angle) - angle / 2
        radii = jnp.linalg.norm(co[:2, :], axis=0)
        co = jnp.asarray([radii * jnp.cos(phi) - radius, radii * jnp.sin(phi), co[2]])

        return function_(co, *params)

    return new_geo_object


def linear_instancing(function_: function_like_type,
                      n: int,
                      a: array_like_type,
                      b: array_like_type) -> function_like_type:
    """
    Instances the geometry n-times along the length of a segment defined by two points.

    Args:
        function_: Original SDF.
        n: Number of instances along the segment.
        a: Starting point of the segment.
        b: Ending point of the segment.

    Returns:
        Modified SDF.
    """

    a = jnp.asarray(a)
    b = jnp.asarray(b)

    @jax.jit
    def new_geo_object(co, *params):
        w = b - a
        c = (b + a) / 2
        l = jnp.linalg.norm(w)

        x = w / l
        y = jnp.asarray([-x[1], x[0], 0])
        y = y / jnp.linalg.norm(y)
        z = jnp.cross(x, y)
        rot = jnp.asarray([x, y, z])
        co = jnp.subtract(co.T, c).T
        co = rot.dot(co)

        s = l / (n - 1)
        d = s / 2
        mx = (co[0, :] >= -l / 2 + d) * (co[0, :] <= l / 2 - d)

        # outer
        v0 = jnp.abs(co[0, :]) - l / 2
        v0 = jnp.where(co[0, :] < 0, -v0, v0)
        v = jnp.asarray([v0, co[1, :], co[2, :]])

        # inner
        if n > 2:
            u0 = jnp.mod(co[0] - (l/2 - d), s) - d
            v = jnp.asarray([jnp.where(mx, u0, v0), co[1, :], co[2, :]])

        return function_(v, *params)

    return new_geo_object


def custom_modification(function_: function_like_type,
                        modification: Callable[[function_like_type, array_like_type, tuple, tuple], jnp.ndarray],
                        modification_parameters: tuple) -> function_like_type:
    """
    Applies a custom user-specified modification to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        modification: A custom modification which takes the SDF as a first argument,
            point cloud of coordinates at which the SDF is evaluated as the second argument,
            parameters of the SDF as the third argument,
            and the custom modification parameters as the fourth argument.
        modification_parameters: Parameters of the custom modification.

    Returns:
        Modified Scalar Field.
    """

    def new_geo_object(co, *params):
        return modification(function_, co, params, modification_parameters)

    return new_geo_object


def sigmoid_falloff(function_: function_like_type,
                    amplitude: scalar_like_type, width: scalar_like_type) -> function_like_type:
    """
    Applies a sigmoid to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        amplitude: Maximum value of the transformed scalar field.
        width: Width of the sigmoid.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return sigmoid_falloff_jax(u, amplitude, width)

    return new_geo_object


def positive_sigmoid_falloff(function_: function_like_type,
                             amplitude: scalar_like_type,
                             width: scalar_like_type) -> function_like_type:
    """
    Applies a sigmoid to a function defining a scalar field (Signed Distance Function).
    The sigmoid is shifted towards the positive values by the value of the width parameter.

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        amplitude: Maximum value of the transformed scalar field.
        width: Width of the sigmoid.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return positive_sigmoid_falloff_jax(u, amplitude, width)

    return new_geo_object


def capped_exponential(function_: function_like_type,
                       amplitude: scalar_like_type,
                       width: scalar_like_type) -> function_like_type:
    """
    Applies the Capped Exponential function to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to almost zero.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return capped_exponential_jax(u, amplitude, width)

    return new_geo_object


def hard_binarization(function_: function_like_type, threshold: float) -> function_like_type:
    """
    Binarizes the output of a function defining a scalar field (Signed Distance Function).
    Values below the threshold are 1 and values above are 0.

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        threshold: Binarization threshold.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return hard_binarization_jax(u, threshold)

    return new_geo_object


def linear_falloff(function_: function_like_type, amplitude: scalar_like_type, width: scalar_like_type) -> function_like_type:
    """
    Applies a decreasing linear function to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to zero.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return linear_falloff_jax(u, amplitude, width)

    return new_geo_object


def relu(function_: function_like_type, width: scalar_like_type) -> function_like_type:
    """
    Applies the ReLU function to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        width: Range at which the value of the transformed field reaches one.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return relu_jax(u, width)

    return new_geo_object


def smooth_relu(function_: function_like_type, smooth_width: scalar_like_type,
                    width: scalar_like_type = 1., threshold: int | float = 0.01) -> function_like_type:
    """
    Applies the "squareplus" function to a function defining a scalar field (Signed Distance Function).
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        smooth_width: Distance from the origin at which the Smooth ReLU function
            is greater than ReLU for less than the value of the threshold parameter.
        width: Range at which the value of the transformed field reaches one.
        threshold: At smooth_width distance from the origin the value of the Smooth ReLU function is greater
            than ReLU for the value of the threshold parameter.
            at smooth_width distance from the origin.

    Returns:
        Transformed scalar field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return smooth_relu_jax(u, smooth_width, width, threshold)

    return new_geo_object


def slowstart(function_: function_like_type, smooth_width: scalar_like_type,
              width: scalar_like_type = 1,
              threshold: int | float = 0.01,
              ground: bool = True) -> function_like_type:
    """
    Applies the SlowStart function to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        smooth_width: Distance from the origin at which the SlowStart function
            is greater than ReLU for less than the value of the threshold parameter.
        width: Range at which the value of the transformed field reaches one.
        threshold: At smooth_width distance from the origin the value of the SlowStart function is greater
            than ReLU for the value of the threshold parameter.
        ground: if True the value of the function is zero at zero.

    Returns:
        Transformed scalar field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return slowstart_jax(u, smooth_width, width, threshold, ground=ground)

    return new_geo_object


def gaussian_boundary(function_: function_like_type,
                      amplitude: scalar_like_type, width: scalar_like_type) -> function_like_type:
    """
    Applies the Gaussian function to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to almost zero.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return gaussian_boundary_jax(u, amplitude, width)

    return new_geo_object


def gaussian_falloff(function_: function_like_type,
                     amplitude: scalar_like_type,
                     width: scalar_like_type) -> function_like_type:
    """
    Applies the Gaussian Falloff function to a function defining a scalar field (Signed Distance Function).

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        amplitude: Maximum value of the transformed scalar field (and points at which the scalar field was < 0).
        width: Range at which the value of the transformed scalar field drops to almost zero.

    Returns:
        Modified Scalar Field.
    """

    @jax.jit
    def new_geo_object(co, *params):
        u = function_(co, *params)
        return gaussian_falloff_jax(u, amplitude, width)

    return new_geo_object


def conv_averaging(function_: function_like_type,
                   kernel_size: array_like_type,
                   iterations: int,
                   co_resolution: tuple) -> function_like_type:
    """
    Averages the field using an averaging convolutional kernel of the specified size.

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        kernel_size: Size of the averaging kernel. Must be a tuple/array of the
            same dimension as the scaler field.
        iterations: Number of times the convolutional averaging is applied to the input scalar field.
        co_resolution: Resolution of the coordinate system.

    Returns:
        Modified Scalar Field.
    """

    def new_geo_object(co, *params):
        u = function_(co, *params)
        u = smarter_reshape(u, co_resolution)

        kernel_ = jnp.ones(kernel_size)
        kernel_ = kernel_/kernel_.sum()
        out = conv_multiple_jax(u, kernel_, iterations)
        return out.flatten()

    return new_geo_object


def conv_edge_detection(function_: function_like_type,
                        co_resolution: tuple) -> function_like_type:
    """
    Edge detection with a 3x3 convolutional kernel.

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        co_resolution: Resolution of the coordinate system.

    Returns:
        Modified Scalar Field.
    """

    def new_geo_object(co, *params):
        u = function_(co, *params)
        u = smarter_reshape(u, co_resolution)
        return conv_edge_detection_jax(u).flatten()

    return new_geo_object


def custom_post_process(function_: function_like_type,
                        function: function_like_type,
                        parameters: tuple) -> function_like_type:
    """
    Applies a custom user-specified post-processing function to a scalar (Signed Distance Function) field.

    Args:
        function_: A function defining a scalar field (Signed Distance Function).
        function: A custom post-processing function which takes the SDF as a first argument
            and the parameters of the function as the next arguments.
        parameters: Parameters of the custom post-processing function.

    Returns:
         Modified Scalar Field.
    """

    def new_geo_object(co, *params):
        u = function_(co, *params)
        return function(u, *parameters)

    return new_geo_object
