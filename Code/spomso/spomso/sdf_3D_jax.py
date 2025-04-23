# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np
import jax
from jax.lax import fori_loop

array_like_type = jnp.ndarray | np.ndarray | list | tuple
scalar_like_type = float | int


@jax.jit
def sdf_x(co: array_like_type, offset: scalar_like_type) -> jnp.ndarray:
    """
    Value of the X coordinate zeroed at some offset value.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        offset: Value at which the field is zero.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    return co[0] - offset


@jax.jit
def sdf_y(co: array_like_type, offset: scalar_like_type) -> jnp.ndarray:
    """
    Value of the Y coordinate zeroed at some offset value.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        offset: Value at which the field is zero.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    return co[1] - offset


@jax.jit
def sdf_z(co: array_like_type, offset: scalar_like_type) -> jnp.ndarray:
    """
    Value of the Z coordinate zeroed at some offset value.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        offset: Value at which the field is zero.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    return co[2] - offset


@jax.jit
def sdf_sphere(co: array_like_type, radius: scalar_like_type) -> jnp.ndarray:
    """
    Sphere defined by its radius.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        radius: Radius of the sphere.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    length = jnp.linalg.norm(co, axis=0)
    return length - radius


@jax.jit
def sdf_cylinder(co: array_like_type, radius: scalar_like_type, height: scalar_like_type) -> jnp.ndarray:
    """
    Cylinder defined by the radius and height.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        radius: Radius of the cylinder.
        height: Height of the cylinder.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    l1 = jnp.linalg.norm(co[:2, :], axis=0)

    d0 = l1 - radius
    d1 = jnp.abs(co[2, :]) - height / 2
    term1 = jnp.minimum(jnp.maximum(d0, d1), 0)
    term2 = jnp.linalg.norm(jnp.asarray([jnp.maximum(d0, 0), jnp.maximum(d1, 0)]), axis=0)
    return term1 + term2


@jax.jit
def sdf_box(co: array_like_type, size: array_like_type) -> jnp.ndarray:
    """
    Box defined by its side lengths.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        size: Side lengths (a, b, c).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    v = jnp.asarray(size)/2
    qo = jnp.abs(co).T - v

    t1 = jnp.linalg.norm(jnp.maximum(qo, 0.0), axis=1)
    t2 = jnp.minimum(jnp.maximum(qo[:, 0], jnp.maximum(qo[:, 1], qo[:, 2])), 0.0)
    return t1 + t2


@jax.jit
def sdf_torus(co: array_like_type, R: scalar_like_type, r: scalar_like_type) -> jnp.ndarray:
    """
    Torus defined by its primary and the secondary radius.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        R: Primary radius of the torus.
        r: Secondary radius of the torus.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    p = jnp.linalg.norm(co[:2], axis=0) - R
    q = jnp.linalg.norm(jnp.asarray([p, co[2]]), axis=0)
    return q - r


@jax.jit
def sdf_arc_3d(co: array_like_type,
               R: scalar_like_type, r: scalar_like_type,
               start_angle: scalar_like_type, end_angle: scalar_like_type) -> jnp.ndarray:
    """
    Arc defined by the radius, thickness, and the angles of both ends.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        R: Radius of the arc.
        r: Thickness of the arc.
        start_angle: Angle of one end with respect to the x-axis.
        end_angle: Angle of the other end with respect to the x-axis.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    c_angle = (start_angle + end_angle) / 2
    rot = jnp.asarray([[jnp.cos(c_angle), jnp.sin(c_angle)], [-jnp.sin(c_angle), jnp.cos(c_angle)]])

    co2 = rot.dot(co[:2])
    co2 = co2.at[1].set(jnp.abs(co2[1]))
    phi = jnp.arctan2(co2[1], co2[0])

    end_angle = jnp.abs(end_angle - c_angle)
    psi = jnp.clip(phi, 0, end_angle)
    oc = jnp.asarray((R * jnp.cos(psi),
                     R * jnp.sin(psi)))

    co2 = jnp.subtract(co2, oc)
    co = jnp.asarray((co2[0], co2[1], co[2]))
    length = jnp.linalg.norm(co, axis=0) - r
    return length


@jax.jit
def sdf_plane(co: array_like_type, normal: array_like_type, offset: scalar_like_type) -> jnp.ndarray:
    """
    Plane defined by its normal vector. SDF has a negative value for points below the plane.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        normal: Normal vector of the plane.
        offset: Offset of the origin along the normal vector.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    n = normal / jnp.linalg.norm(normal)
    d = jnp.dot(co.T, n).T
    return d - offset


@jax.jit
def sudf_plane(co: array_like_type, normal: array_like_type, thickness: scalar_like_type) -> jnp.ndarray:
    """
    Plane defined by its normal vector.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        normal: Normal vector of the plane.
        thickness: Thickness of the plane.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    n = normal / jnp.linalg.norm(normal)
    d = jnp.abs(jnp.dot(co.T, n).T)
    return d - thickness/2


@jax.jit
def sdf_segment_3d(co: array_like_type, a: array_like_type, b: array_like_type) -> jnp.ndarray:
    """
    Line defined by its starting and ending points.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        a: Vector defining the starting point.
        b: Vector defining the ending point.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    pa = jnp.subtract(co.T, a).T
    ba = b - a
    h = jnp.clip(jnp.sum(jnp.multiply(pa.T, ba).T, axis=0) / jnp.dot(ba, ba), 0, 1)
    length = jnp.linalg.norm(pa - jnp.outer(ba, h), axis=0)
    return length


@jax.jit
def sdf_cone(co: array_like_type, height: scalar_like_type, angle: scalar_like_type) -> jnp.ndarray:
    """
    Cone defined by its height and the angle of the slope.
    The base of the cone is moved down by: height - height_offset.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        height: Height of the cone.
        angle: Angle of the slope.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    # https://iquilezles.org/articles/distfunctions/
    q = jnp.asarray((jnp.tan(angle), -1)) * height
    co_z = co[2] - height * (0.5 ** (1 / 3))

    w = jnp.asarray((jnp.linalg.norm(co[:2], axis=0), co_z))
    ones = jnp.ones(w.shape[1])
    a = w - jnp.outer(q, jnp.clip(jnp.dot(w.T, q).T / jnp.dot(q, q), 0.0, 1.0))
    b = w.T - q * jnp.asarray((jnp.clip(w[0] / q[0], 0.0, 1.0), ones)).T

    aa = jnp.sum(a**2, axis=0)
    bb = jnp.sum(b**2, axis=1)
    d = jnp.minimum(aa, bb)
    s = jnp.maximum(-(w[0] * q[1] - w[1] * q[0]), -(w[1] - q[1]))
    return jnp.sqrt(d) * jnp.sign(s)


@jax.jit
def sdf_oriented_infinite_cone(co: array_like_type, angle: scalar_like_type) -> jnp.ndarray:
    """
    Cone with infinite height defined by the angle of its slope. The tip of the cone is at the origin.
    Values of the SDF below the cone are negative.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        angle: Angle of the slope.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    # https://iquilezles.org/articles/distfunctions/
    v = jnp.asarray([jnp.sin(angle), jnp.cos(angle)])
    q = jnp.asarray((jnp.linalg.norm(co[:2], axis=0), -co[2]))
    qv = jnp.outer(v, jnp.maximum(jnp.dot(q.T, v).T, 0.0))
    d = jnp.linalg.norm(q - qv, axis=0)

    f = jnp.sign(q[0] * v[1] - q[1] * v[0])

    return d * f


@jax.jit
def sdf_infinite_cone(co: array_like_type, angle: scalar_like_type) -> jnp.ndarray:
    """
    Cone with infinite height defined by the angle of its slope. The tip of the cone is at the origin.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        angle: Angle of the slope.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    v = jnp.asarray([jnp.sin(angle), jnp.cos(angle)])
    q = jnp.asarray((jnp.linalg.norm(co[:2], axis=0), -co[2]))
    qv = jnp.outer(v, jnp.maximum(jnp.dot(q.T, v).T, 0.0))
    d = jnp.linalg.norm(q - qv, axis=0)

    return d


@jax.jit
def sdf_solid_angle(co: array_like_type,
                    radius: scalar_like_type,
                    angle_1: scalar_like_type, angle_2: scalar_like_type) -> jnp.ndarray:
    """
    Solid angle defined by the radius of the globe and two angles.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        radius: Radius of the globe.
        angle_1: First angle defining the sector.
        angle_2: Second angle defining the sector.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    angle_diff = jnp.abs(angle_2 - angle_1) / 2
    c_angle = (angle_2 + angle_1) / 2
    rot = jnp.asarray([[jnp.cos(c_angle), jnp.sin(c_angle)], [-jnp.sin(c_angle), jnp.cos(c_angle)]])

    co_xy = rot.dot(co[:2])
    co_yz = jnp.asarray((co_xy[1], co[2]))
    co_y = jnp.linalg.norm(co_yz, axis=0)
    co2 = jnp.asarray((co_xy[0], co_y))

    phi = jnp.arctan2(co2[1], co2[0])
    psi = jnp.clip(phi, 0, angle_diff)
    oc = jnp.asarray((radius * jnp.cos(psi),
                     radius * jnp.sin(psi)))
    poc = jnp.subtract(co2, oc)
    length = jnp.linalg.norm(poc, axis=0)

    c = jnp.asarray([jnp.cos(angle_diff), jnp.sin(angle_diff)])
    outer = jnp.outer(c, jnp.clip(jnp.dot(co2.T, c).T, 0, radius))
    m = jnp.linalg.norm(co2 - outer, axis=0)

    msk = (jnp.linalg.norm(co2, axis=0) <= radius) * (phi <= angle_diff)
    out = jnp.minimum(m, length)
    out = jnp.where(msk, -out, out)

    return out


@jax.jit
def sdf_triangle_3d(co: array_like_type, a: array_like_type, b: array_like_type, c:array_like_type) -> jnp.ndarray:
    """
    Triangle defined by its three vertices.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        a: Vector defining the position of the first vertex.
        b: Vector defining the position of the second vertex.
        c: Vector defining the position of the third vertex.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    # https://iquilezles.org/articles/distfunctions/
    s1 = (b - a) / 1
    s2 = (c - b) / 1
    s3 = (a - c) / 1

    coa = jnp.subtract(co.T, a).T
    cob = jnp.subtract(co.T, b).T
    coc = jnp.subtract(co.T, c).T

    normal = jnp.cross(s1, s3)

    mask = jnp.sign(jnp.dot(jnp.cross(s1, normal), coa)) + jnp.sign(jnp.dot(jnp.cross(s2, normal), cob))
    mask += jnp.sign(jnp.dot(jnp.cross(s3, normal), coc))
    mask = mask < 2.

    term1 = jnp.outer(s1, jnp.clip(jnp.dot(s1, coa) / jnp.dot(s1, s1), 0, 1)) - coa
    term2 = jnp.outer(s2, jnp.clip(jnp.dot(s2, cob) / jnp.dot(s2, s2), 0, 1)) - cob
    term3 = jnp.outer(s3, jnp.clip(jnp.dot(s3, coc) / jnp.dot(s3, s3), 0, 1)) - coc

    ex1 = jnp.minimum(jnp.minimum(jnp.sum(term1 * term1, axis=0), jnp.sum(term2 * term2, axis=0)),
                      jnp.sum(term3 * term3, axis=0))
    ex2 = (jnp.dot(normal, coa) ** 2) / jnp.dot(normal, normal)

    outer = jnp.where(mask, jnp.sqrt(ex1), jnp.sqrt(ex2))

    return outer


@jax.jit
def sdf_quad_3d(co: array_like_type,
                a: array_like_type, b: array_like_type, c: array_like_type, d: array_like_type) -> jnp.ndarray:
    """
     Quadrilateral defined by its four coplanar vertices.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        a: Vector defining the position of the first vertex.
        b: Vector defining the position of the second vertex.
        c: Vector defining the position of the third vertex.
        d: Vector defining the position of the fourth vertex.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    # https://iquilezles.org/articles/distfunctions/
    s1 = (b - a)
    s2 = (c - b)
    s3 = (d - c)
    s4 = (a - d)

    coa = jnp.subtract(co.T, a).T
    cob = jnp.subtract(co.T, b).T
    coc = jnp.subtract(co.T, c).T
    cod = jnp.subtract(co.T, d).T

    normal = jnp.cross(s1, s4)

    mask = jnp.sign(jnp.dot(jnp.cross(s1, normal), coa)) + jnp.sign(jnp.dot(jnp.cross(s2, normal), cob))
    mask += jnp.sign(jnp.dot(jnp.cross(s3, normal), coc)) + jnp.sign(jnp.dot(jnp.cross(s4, normal), cod))
    mask = mask < 3.

    term1 = jnp.outer(s1, jnp.clip(jnp.dot(s1, coa) / jnp.dot(s1, s1), 0, 1)) - coa
    term2 = jnp.outer(s2, jnp.clip(jnp.dot(s2, cob) / jnp.dot(s2, s2), 0, 1)) - cob
    term3 = jnp.outer(s3, jnp.clip(jnp.dot(s3, coc) / jnp.dot(s3, s3), 0, 1)) - coc
    term4 = jnp.outer(s4, jnp.clip(jnp.dot(s4, cod) / jnp.dot(s4, s4), 0, 1)) - cod
    s_ex1 = jnp.minimum(jnp.sum(term1 * term1, axis=0), jnp.sum(term2 * term2, axis=0))
    s_ex2 = jnp.minimum(jnp.sum(term4 * term4, axis=0), jnp.sum(term3 * term3, axis=0))

    ex1 = jnp.minimum(s_ex2, s_ex1)
    ex2 = (jnp.dot(normal, coa) ** 2) / jnp.dot(normal, normal)

    outer = jnp.where(mask, jnp.sqrt(ex1), jnp.sqrt(ex2))

    return outer


@jax.jit
def sdf_segmented_line_3d(co: array_like_type, points: array_like_type) -> jnp.ndarray:
    """
    Segmented line connecting the provided points.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        points: Points to connect (D > 1, M).
    """
    out = np.ones(co.shape[1]) * 1e16

    def body(i, a):
        b = sdf_segment_3d(co, points[:, i], points[:, i + 1])
        a = jnp.minimum(a, b)
        return a

    out = fori_loop(0, points.shape[1]-1, body, out)

    return out


@jax.jit
def sdf_closed_segmented_line_3d(co: array_like_type, points: array_like_type) -> jnp.ndarray:
    """
    Closed segmented line connecting the provided points.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D > 1),
            N is the number of coordinate points.
        points: Points to connect (D > 1, M).
    """
    out = jnp.ones(co.shape[1]) * 1e16
    points = jnp.asarray(points)
    points = jnp.concatenate((points, jnp.expand_dims(points[:, 0], axis=1)), axis=1)

    def body(i, a):
        b = sdf_segment_3d(co, points[:, i], points[:, i + 1])
        a = jnp.minimum(a, b)
        return a

    out = fori_loop(0, points.shape[1]-1, body, out)

    return out

