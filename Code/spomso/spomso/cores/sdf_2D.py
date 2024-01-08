# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


def sdf_circle(co, radius):
    length = np.linalg.norm(co[:2], axis=0)
    return length - radius


def sdf_box_2d(co, size):
    d = np.subtract(np.abs(co[:2, :]).T, size).T

    term1 = np.linalg.norm(np.maximum(d, 0), axis=0)
    term2 = np.minimum(np.maximum(d[0],d[1]), 0)
    return term1 + term2


def sdf_segment_2d(co, a, b):
    a = np.asarray(a[:2])
    b = np.asarray(b[:2])
    pa = np.subtract(co[:2, :].T, a).T
    ba = b - a
    h = np.clip(np.sum(np.multiply(pa.T, ba).T, axis=0)/np.dot(ba, ba), 0, 1)
    length = np.linalg.norm(pa - np.outer(ba,h), axis=0)
    return length


def sdf_rounded_box_2d(co, size, rounding):

    m1 = co[0] > 0
    m2 = co[1] > 0
    m3 = co[0] < 0

    r = rounding[0]*np.ones(co.shape[1])
    r[m1] = rounding[1]
    r[m2] = rounding[2]
    r[m3*m2] = rounding[3]

    d = np.subtract(np.abs(co[:2, :]).T, size).T + r
    o = np.linalg.norm(np.maximum(d, 0), axis=0)
    u = np.minimum(np.maximum(d[0], d[1]), 0) - r

    return o + u


def sdf_triangle_2d(co, p0, p1, p2):

    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2

    v0 = np.subtract(co[:2,:].T, p0).T
    v1 = np.subtract(co[:2,:].T, p1).T
    v2 = np.subtract(co[:2,:].T, p2).T

    pq0 = v0 - np.outer(e0, np.clip(np.dot(v0.T, e0).T / np.dot(e0, e0), 0, 1))
    pq1 = v1 - np.outer(e1, np.clip(np.dot(v1.T, e1).T / np.dot(e1, e1), 0, 1))
    pq2 = v2 - np.outer(e2, np.clip(np.dot(v2.T, e2).T / np.dot(e2, e2), 0, 1))

    s = np.sign(e0[0]*e2[1] - e0[1]*e2[0])

    dv0 = np.asarray( ( np.sum(np.multiply(pq0, pq0), axis=0), s*(v0[0]*e0[1]-v0[1]*e0[0]) ) )
    dv1 = np.asarray((np.sum(np.multiply(pq1, pq1), axis=0), s * (v1[0] * e1[1] - v1[1] * e1[0])))
    dv2 = np.asarray((np.sum(np.multiply(pq2, pq2), axis=0), s * (v2[0] * e2[1] - v2[1] * e2[0])))

    d = np.amin([dv0, dv1, dv2], axis=0)

    return -np.sqrt(d[0])*np.sign(d[1])


def sdf_arc_positive_only(co, radius, start_angle, end_angle):

    co = co[:2]

    phi = np.arctan2(co[1], co[0])
    phi[phi < 0] = 2 * np.pi + phi[phi < 0]

    ad = end_angle - start_angle
    h = np.clip((phi - start_angle)/ad, 0, 1)
    psi = ad*h + start_angle
    oc = np.asarray((radius*np.cos(psi),
                    radius*np.sin(psi) ))

    poc = np.subtract(co, oc)

    length = np.linalg.norm(poc, axis=0)
    start_length = np.linalg.norm(np.subtract(co.T, (radius*np.cos(start_angle), radius*np.sin(start_angle))).T, axis=0)
    return np.minimum(length, start_length)


def sdf_arc(co, radius, start_angle, end_angle):

    co = co[:2]
    c_angle = (start_angle + end_angle)/2
    rot = np.asarray([[np.cos(c_angle), np.sin(c_angle)], [-np.sin(c_angle), np.cos(c_angle)]])
    co = rot.dot(co)
    co[1] = np.abs(co[1])
    phi = np.arctan2(co[1], co[0])

    end_angle = np.abs(end_angle - c_angle)

    psi = np.clip(phi, 0, end_angle)
    oc = np.asarray((radius*np.cos(psi),
                    radius*np.sin(psi) ))

    poc = np.subtract(co, oc)

    length = np.linalg.norm(poc, axis=0)
    return length


def sdf_sector_old(co, radius, angle):
    # UNFINISHED
    c = np.asarray([np.sin(angle), np.cos(angle)])
    co = co[:2]
    co[0] = np.abs(co[0])

    l = np.linalg.norm(co, axis=0) - radius
    m = np.linalg.norm(co - np.outer(c, np.clip( np.dot(co.T, c), 0, radius)))

    return np.maximum(l, m*np.sign(c[1]*co[0] - c[0]*co[1]))


def sdf_sector(co, radius, angle_1, angle_2):

    angle_diff = np.abs(angle_2 - angle_1)/2
    c_angle = (angle_2 + angle_1)/2

    rot = np.asarray([[np.cos(c_angle), np.sin(c_angle)], [-np.sin(c_angle), np.cos(c_angle)]])
    co = rot.dot(co[:2])
    co[1] = np.abs(co[1])

    phi = np.arctan2(co[1], co[0])
    psi = np.clip(phi, 0, angle_diff)
    oc = np.asarray((radius * np.cos(psi),
                     radius * np.sin(psi)))
    poc = np.subtract(co, oc)
    length = np.linalg.norm(poc, axis=0)

    c = np.asarray([np.cos(angle_diff), np.sin(angle_diff)])
    outer = np.outer(c, np.clip( np.dot(co.T, c).T, 0, radius))
    m = np.linalg.norm(co - outer, axis=0)

    msk = (np.linalg.norm(co, axis=0) <= radius)*(phi <= angle_diff)
    out = np.minimum(m,length)
    out[msk] = -out[msk]

    return out


def sdf_inf_sector(co, angle_1, angle_2):

    angle_diff = np.abs(angle_2 - angle_1)/2
    c_angle = (angle_2 + angle_1)/2

    rot = np.asarray([[np.cos(c_angle), np.sin(c_angle)], [-np.sin(c_angle), np.cos(c_angle)]])
    co = rot.dot(co[:2])
    co[1] = np.abs(co[1])

    phi = np.arctan2(co[1], co[0])

    c = np.asarray([np.cos(angle_diff), np.sin(angle_diff)])
    outer = np.outer(c, np.clip( np.dot(co.T, c).T, 0, np.inf))
    m = np.linalg.norm(co - outer, axis=0)

    sign = (phi - angle_diff)
    out = sign*m

    return out


def sdf_ngon(co, radius, n):

    co = co[:2, :]
    beta = np.pi*(0.5 - 1/n)
    alpha = 2*np.pi/n

    phi = np.arctan2(co[1], co[0])
    phi[phi<0] = 2*np.pi + phi[phi<0]
    phi = np.mod(phi, alpha)
    r_ = np.linalg.norm(co, axis=0)
    co = np.asarray([np.cos(phi), np.sin(phi)])*r_

    qo = np.subtract(co.T, [radius, 0]).T
    s = np.sin(beta)
    c = np.cos(beta)
    t = np.asarray([-c, s])
    no = np.asarray([s, c])
    l = 2*radius*np.sin(alpha/2)

    h = np.clip(np.dot(qo.T, t), 0, l)
    length = np.linalg.norm(qo - np.outer(t, h), axis=0)
    sign = np.sign(np.dot(qo.T, no))
    out = length*sign

    return out


def sdf_segmented_curve_2d(co, points, t):

    v = np.floor(t).astype(int)
    u = t - v
    fval = points[:2, v+1]*u + points[:2, v]*(1-u)

    tree = KDTree(fval.T)
    mindist, minid = tree.query(co[:2, :].T)
    return mindist


def sdf_segmented_line_2d(co, points):

    out = np.ones(co.shape[1])*10000
    for i in range(points.shape[1]-1):
        part = sdf_segment_2d(co, points[:, i], points[:, i+1])
        out = np.minimum(out, part)

    return out


def sdf_parametric_curve_2d(co, f, f_parameters, t):
    fval = f(t, *f_parameters)
    tree = KDTree(fval.T)
    mindist, minid = tree.query(co[:2, :].T)
    return mindist


def sdf_point_cloud_2d(co, points):
    tree = KDTree(points[:2, :].T)
    mindist, minid = tree.query(co[:2, :].T)
    return mindist

