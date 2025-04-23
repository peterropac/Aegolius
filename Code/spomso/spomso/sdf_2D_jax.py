# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np
import jax
from jax.lax import fori_loop
from functools import partial

array_like_type = jnp.ndarray | np.ndarray | list | tuple
scalar_like_type = float | int


@jax.jit
def sdf_circle(co: array_like_type, radius: scalar_like_type) -> jnp.ndarray:
    """
    Circle defined by its radius.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        radius: Radius of the circle.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    length = jnp.linalg.norm(co[:2], axis=0)
    return length - radius


@jax.jit
def sdf_box_2d(co: array_like_type, size: array_like_type) -> jnp.ndarray:
    """
    Rectangle defined by its side lengths.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        size: side lengths of the rectangle (a, b).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    d = jnp.subtract(jnp.abs(co[:2, :]).T, size/2).T

    term1 = jnp.linalg.norm(jnp.maximum(d, 0), axis=0)
    term2 = jnp.minimum(jnp.maximum(d[0], d[1]), 0)
    return term1 + term2


@jax.jit
def sdf_segment_2d(co: array_like_type, a: array_like_type, b: array_like_type) -> jnp.ndarray:
    """
    Line segment defined by its end points.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        a: Vector defining the position of the start point (D = 3,).
        b: Vector defining the position of the end point (D = 3,).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    a = jnp.asarray(a[:2])
    b = jnp.asarray(b[:2])
    pa = jnp.subtract(co[:2, :].T, a).T
    ba = b - a
    h = jnp.clip(jnp.sum(jnp.multiply(pa.T, ba).T, axis=0)/jnp.dot(ba, ba), 0, 1)
    length = jnp.linalg.norm(pa - jnp.outer(ba, h), axis=0)
    return length


@jax.jit
def sdf_rounded_box_2d(co: array_like_type, size: array_like_type, rounding: array_like_type) -> jnp.ndarray:
    """
    Rectangle defined by its side lengths.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        size: side lengths of the rectangle (a, b).
        rounding: rounding radii for each of the corners (r1, r2, r3, r4).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    m1 = co[0] > 0
    m2 = co[1] > 0
    m3 = ~m1

    r = rounding[0]*jnp.ones(co.shape[1])
    r = jnp.where(m1, rounding[1], r)
    r = jnp.where(m2, rounding[2], r)
    r = jnp.where(m3*m2, rounding[3], r)

    d = jnp.subtract(jnp.abs(co[:2, :]).T, size/2).T + r
    o = jnp.linalg.norm(jnp.maximum(d, 0), axis=0)
    u = jnp.minimum(jnp.maximum(d[0], d[1]), 0) - r

    return o + u


@jax.jit
def sdf_triangle_2d(co: array_like_type, p0: array_like_type, p1: array_like_type, p2: array_like_type) -> jnp.ndarray:
    """
    Triangle defined by the three vertices.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        p0: Vector defining the position of the first vertex (D = 3,).
        p1: Vector defining the position of the second vertex (D = 3,).
        p2: Vector defining the position of the third vertex (D = 3,).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2

    v0 = jnp.subtract(co[:2,:].T, p0).T
    v1 = jnp.subtract(co[:2,:].T, p1).T
    v2 = jnp.subtract(co[:2,:].T, p2).T

    pq0 = v0 - jnp.outer(e0, jnp.clip(jnp.dot(v0.T, e0).T / jnp.dot(e0, e0), 0, 1))
    pq1 = v1 - jnp.outer(e1, jnp.clip(jnp.dot(v1.T, e1).T / jnp.dot(e1, e1), 0, 1))
    pq2 = v2 - jnp.outer(e2, jnp.clip(jnp.dot(v2.T, e2).T / jnp.dot(e2, e2), 0, 1))

    s = jnp.sign(e0[0]*e2[1] - e0[1]*e2[0])

    dv0 = jnp.asarray( ( jnp.sum(jnp.multiply(pq0, pq0), axis=0), s*(v0[0]*e0[1]-v0[1]*e0[0]) ) )
    dv1 = jnp.asarray((jnp.sum(jnp.multiply(pq1, pq1), axis=0), s * (v1[0] * e1[1] - v1[1] * e1[0])))
    dv2 = jnp.asarray((jnp.sum(jnp.multiply(pq2, pq2), axis=0), s * (v2[0] * e2[1] - v2[1] * e2[0])))

    d = jnp.amin(jnp.asarray([dv0, dv1, dv2]), axis=0)

    return -jnp.sqrt(d[0])*jnp.sign(d[1])


@jax.jit
def sdf_arc(co: array_like_type,
            radius: scalar_like_type, start_angle: scalar_like_type, end_angle: scalar_like_type) -> jnp.ndarray:
    """
    Arc defined by the radius and the angles of both ends.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        radius: Radius of the arc.
        start_angle: Angle of one end with respect to the x-axis.
        end_angle: Angle of the other end with respect to the x-axis.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    co = co[:2]
    c_angle = (start_angle + end_angle)/2
    rot = jnp.asarray([[jnp.cos(c_angle), jnp.sin(c_angle)], [-jnp.sin(c_angle), jnp.cos(c_angle)]])
    co = rot.dot(co)
    co = co.at[1].set(jnp.abs(co[1]))
    phi = jnp.arctan2(co[1], co[0])

    end_angle = jnp.abs(end_angle - c_angle)

    psi = jnp.clip(phi, 0, end_angle)
    oc = jnp.asarray((radius*jnp.cos(psi), radius*jnp.sin(psi)))

    poc = jnp.subtract(co, oc)

    length = jnp.linalg.norm(poc, axis=0)
    return length


@jax.jit
def sdf_sector(co: array_like_type,
               radius: scalar_like_type, angle_1: scalar_like_type, angle_2: scalar_like_type) -> jnp.ndarray:
    """
    Sector defined by the radius of the circle and two angles.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        radius: Radius of the circle.
        angle_1: First angle defining the sector.
        angle_2: Second angle defining the sector.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    angle_diff = jnp.abs(angle_2 - angle_1)/2
    c_angle = (angle_2 + angle_1)/2

    rot = jnp.asarray([[jnp.cos(c_angle), jnp.sin(c_angle)], [-jnp.sin(c_angle), jnp.cos(c_angle)]])
    co = rot.dot(co[:2])
    co = co.at[1].set(jnp.abs(co[1]))

    phi = jnp.arctan2(co[1], co[0])
    psi = jnp.clip(phi, 0, angle_diff)
    oc = jnp.asarray((radius * jnp.cos(psi),
                     radius * jnp.sin(psi)))
    poc = jnp.subtract(co, oc)
    length = jnp.linalg.norm(poc, axis=0)

    c = jnp.asarray([jnp.cos(angle_diff), jnp.sin(angle_diff)])
    outer = jnp.outer(c, jnp.clip(jnp.dot(co.T, c).T, 0, radius))
    m = jnp.linalg.norm(co - outer, axis=0)

    out = jnp.minimum(m, length)
    msk = (jnp.linalg.norm(co, axis=0) <= radius) * (phi <= angle_diff)
    out = jnp.where(msk, -1*out, out)

    return out


@jax.jit
def sdf_inf_sector(co: array_like_type, angle_1: scalar_like_type, angle_2: scalar_like_type) -> jnp.ndarray:
    """
    Sector of infinite radius defined by two angles.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        angle_1: First angle defining the sector.
        angle_2: Second angle defining the sector.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    angle_diff = jnp.abs(angle_2 - angle_1)/2
    c_angle = (angle_2 + angle_1)/2

    rot = jnp.asarray([[jnp.cos(c_angle), jnp.sin(c_angle)], [-jnp.sin(c_angle), jnp.cos(c_angle)]])
    co = rot.dot(co[:2])
    co = co.at[1].set(jnp.abs(co[1]))

    phi = jnp.arctan2(co[1], co[0])

    c = jnp.asarray([jnp.cos(angle_diff), jnp.sin(angle_diff)])
    outer = jnp.outer(c, jnp.clip( jnp.dot(co.T, c).T, 0, jnp.inf))
    m = jnp.linalg.norm(co - outer, axis=0)

    sign = jnp.sign(phi - angle_diff)
    out = sign*m

    return out


@jax.jit
def sdf_ngon(co: array_like_type, radius: scalar_like_type, n: int) -> jnp.ndarray:
    """
    N-sided regular polygon, defined by the outer radius and the number of sides.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        radius: Outer radius of the regular polygon.
        n: Number of sides of the regular polygon.

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    beta = jnp.pi*(0.5 - 1/n)
    alpha = 2*jnp.pi/n

    phi = jnp.arctan2(co[1], co[0])
    phi = jnp.where(phi < 0, 2*jnp.pi + phi, phi)

    phi = jnp.mod(phi, alpha)
    r_ = jnp.linalg.norm(co, axis=0)
    co = jnp.asarray([jnp.cos(phi), jnp.sin(phi)])*r_

    s = jnp.sin(beta)
    c = jnp.cos(beta)
    t = jnp.asarray([-c, s])
    no = jnp.asarray([s, c])
    l = 2*radius*jnp.sin(alpha/2)

    qo = jnp.subtract(co.T, jnp.asarray([radius, 0])).T
    h = jnp.clip(jnp.dot(qo.T, t), 0, l)
    length = jnp.linalg.norm(qo - jnp.outer(t, h), axis=0)
    sign = jnp.sign(jnp.dot(qo.T, no))
    out = length*sign

    return out


@jax.jit
def sdf_segmented_line_2d(co: array_like_type, points: array_like_type) -> jnp.ndarray:
    """
    Segmented line connecting the provided points.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        points: Points to connect (D = 3, M).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    out = jnp.ones(co.shape[1])*1e16

    def body(i, a):
        b = sdf_segment_2d(co, points[:, i], points[:, i + 1])
        a = jnp.minimum(a, b)
        return a
    out = fori_loop(0, points.shape[1]-1, body, out)

    return out


@jax.jit
def sdf_closed_segmented_line_2d(co: array_like_type, points: array_like_type) -> jnp.ndarray:
    """
    Closed segmented line connecting the provided points.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        points: Points to connect (D = 3, M).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    out = jnp.ones(co.shape[1])*1e16
    points = jnp.asarray(points)
    points = jnp.concatenate((points, jnp.expand_dims(points[:, 0], axis=1)), axis=1)

    def body(i, a):
        b = sdf_segment_2d(co, points[:, i], points[:, i + 1])
        a = jnp.minimum(a, b)
        return a
    out = fori_loop(0, points.shape[1]-1, body, out)

    return out


@jax.jit
def sdf_polygon_2d(co: array_like_type, points: array_like_type) -> jnp.ndarray:
    """
    Convex polygon with vertices at the provided coordinate points.

    Args:
        co: coordinates of points on which the SDF is evaluated.
            Shape must be (D, N), where D is the dimension of the coordinate system (D = 3),
            N is the number of coordinate points.
        points: Points to connect (D = 3, M).

    Returns:
        SDF, with shape (N,), evaluated at the provided coordinate points.
    """
    points = jnp.asarray(points)

    convexity = check_convex_all(points)

    condition = jnp.sum(jnp.sign(convexity)) == -1*points.shape[1]
    points = points.at[:, :].set(points[:, ::-1]) * condition + points * (1 - condition)

    sp = interior_convex(co, points)
    interior = jnp.where(sp <= 0, -1, 1)

    closed_points = jnp.concatenate((points, jnp.expand_dims(points[:, 0], axis=1)), axis=1)

    def body(i, a):
        b = sdf_segment_2d(co, closed_points[:, i], closed_points[:, i + 1])
        a = jnp.minimum(a, b)
        return a

    out = jnp.ones(co.shape[1]) * 1e16
    out = fori_loop(0, closed_points.shape[1] - 1, body, out)

    return out*interior


def check_convex_all(vs: jnp.ndarray) -> jnp.ndarray:
    """
    Checks if a polygon is convex.

    Args:
        vs: Coordinates of the vertices with shape (D >= 2, N), where N is the number of vertices.
    Returns:
        Convexity of each vertex.
    """

    v1s = vs[:2, 1:-1] - vs[:2, :-2]
    v2s = vs[:2, 2:] - vs[:2, 1:-1]
    c = jnp.cross(v1s.T, v2s.T)
    c_last = jnp.cross(vs[:2, 0] - vs[:2, -1], vs[:2, 1] - vs[:2, 0])
    c = jnp.concatenate((c, jnp.expand_dims(c_last, axis=0)), axis=0)
    return c


def interior_convex(co: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """
    Partitions the coordinate system into the interior (-1) and exterior (+1) of a convex polygon.

    Args:
        co: Point cloud of coordinates with shape (D >= 2, N);
            D - number of dimensions;
            N - number of points in the point cloud.
        points: Coordinates of the vertices with shape (D >= 2, N), where N is the number of vertices.
    Returns:
        Map of the interior and exterior of the polygon (N,).
    """

    def calculate_normals(i, a):
        k = (i + 1) % points.shape[1]
        ci = points[:2, i] / 2 + points[:2, k] / 2 - zero[:2]
        vi = points[:2, k] - points[:2, i]
        vin = vi / jnp.linalg.norm(vi)
        ni = jnp.asarray((-vin[1], vin[0]))
        ni = ni * (1 - 2 * (ni.dot(ci) < 0))
        a = a.at[:, i].set(ni)
        return a

    zero = jnp.average(points, axis=1)
    normals = points.copy()[:2, :]
    normals = fori_loop(0, points.shape[1], calculate_normals, normals)

    def update_interior(i, a):
        m = jnp.sign(jnp.dot(co[:2, :].T - points[:2, i], normals[:2, i]))
        a = a.at[:].set(jnp.maximum(a, m))
        return a

    sp = -jnp.ones(co.shape[1])
    sp = fori_loop(0, points.shape[1], update_interior, sp)

    return sp
