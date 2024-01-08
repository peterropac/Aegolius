# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


def sdf_x(co, offset):
    return co[0] - offset


def sdf_y(co, offset):
    return co[1] - offset


def sdf_z(co, offset):
    return co[2] - offset


def sdf_sphere(co, radius):
    length = np.linalg.norm(co, axis=0)
    return length - radius


def sdf_cylinder(co, radius, height):
    l1 = np.linalg.norm(co[:2,:], axis=0)

    d0 = l1 - radius
    d1 = np.abs(co[2,:]) - height/2
    term1 = np.minimum(np.maximum(d0, d1), 0)
    term2 = np.linalg.norm([np.maximum(d0, 0), np.maximum(d1, 0)], axis=0)
    return term1 + term2


def sdf_box(co, a, b, c):

    v = np.asarray((a,b,c))
    qo = np.abs(co).T - v

    t1 = np.linalg.norm(np.maximum(qo, 0.0), axis=1)
    t2 = np.minimum(np.maximum(qo[:, 0], np.maximum(qo[:, 1], qo[:, 2])), 0.0)
    return t1 + t2


def sdf_torus(co, R, r):
    p = np.linalg.norm(co[:2], axis=0) - R
    q = np.linalg.norm([p, co[2]], axis=0)
    return q - r


def sdf_chainlink(co, R, r, length):
    co[0] = co[0] - np.clip(co[0], -length/2, length/2)
    p = np.linalg.norm(co[:2], axis=0) - R
    q = np.linalg.norm([p, co[2]], axis=0)
    return q - r


def sdf_braid(co, length, R, r, pitch):
    c = np.cos(pitch * co[2])
    s = np.sin(pitch * co[2])
    rot = np.asarray([[c, s], [-s, c]])
    co[:2, :] = rot[0, :, :] * co[0, :] + rot[1, :, :] * co[1, :]

    co[2] -= np.clip(co[2], -length / 2, length / 2)

    p = np.linalg.norm(co[[0,2]], axis=0) - R
    q = np.linalg.norm([p, co[1]], axis=0)
    return q - r


def sdf_arc_3d(co, R, r, start_angle, end_angle):

    c_angle = (start_angle + end_angle) / 2
    rot = np.asarray([[np.cos(c_angle), np.sin(c_angle)], [-np.sin(c_angle), np.cos(c_angle)]])
    co[:2] = rot.dot(co[:2])
    co[1] = np.abs(co[1])
    phi = np.arctan2(co[1], co[0])

    end_angle = np.abs(end_angle - c_angle)

    psi = np.clip(phi, 0, end_angle)
    oc = np.asarray((R * np.cos(psi),
                     R * np.sin(psi)))

    co[:2] = np.subtract(co[:2], oc)

    length = np.linalg.norm(co, axis=0) - r
    return length


def sdf_plane(co, normal, offset):
    n = normal/np.linalg.norm(normal)
    d = np.dot(co.T, n).T
    return d - offset


def sudf_plane(co, normal, thickness):
    n = normal/np.linalg.norm(normal)
    d = np.abs(np.dot(co.T, n).T)
    return d - thickness


def sdf_segment_3d(co, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    pa = np.subtract(co.T, a).T
    ba = b - a
    h = np.clip(np.sum(np.multiply(pa.T, ba).T, axis=0)/np.dot(ba, ba), 0, 1)
    length = np.linalg.norm(pa - np.outer(ba,h), axis=0)
    return length


def sdf_cone(co, height, angle):
    # https://iquilezles.org/articles/distfunctions/
    q = np.asarray((np.tan(angle), -1))*height
    co[2] -= height*(0.5**(1/3))

    w = np.asarray((np.linalg.norm(co[:2], axis=0), co[2]))
    ones = np.ones(w.shape[1])
    a = w - np.outer(q, np.clip(np.dot(w.T, q).T / np.dot(q, q), 0.0, 1.0))
    b = w.T - q*np.asarray((np.clip(w[0] / q[0], 0.0, 1.0), ones)).T

    aa = np.sum(np.multiply(a,a), axis=0)
    bb = np.sum(np.multiply(b,b), axis=1)
    d = np.minimum(aa, bb)
    s = np.maximum(-(w[0] * q[1] - w[1] * q[0]), -(w[1] - q[1]))
    return np.sqrt(d) * np.sign(s)


def sdf_oriented_infinite_cone(co, angle):
    # https://iquilezles.org/articles/distfunctions/
    v = np.asarray([np.sin(angle), np.cos(angle)])
    q = np.asarray((np.linalg.norm(co[:2], axis=0), -co[2]))
    qv = np.outer(v, np.maximum(np.dot(q.T, v).T, 0.0))
    d = np.linalg.norm(q - qv)

    f = -2*(q[0] * v[1] - q[1] * v[0] < 0.0) + 1

    return d * f


def sdf_infinite_cone(co, angle):
    v = np.asarray([np.sin(angle), np.cos(angle)])
    q = np.asarray((np.linalg.norm(co[:2], axis=0), -co[2]))
    qv = np.outer(v, np.maximum(np.dot(q.T, v).T, 0.0))
    d = np.linalg.norm(q - qv)

    f = -2 * (q[0] * v[1] - q[1] * v[0] < 0.0) + 1

    return np.abs(d * f)


def sdf_solid_angle(co, radius, angle_1, angle_2):
    angle_diff = np.abs(angle_2 - angle_1)/2
    c_angle = (angle_2 + angle_1)/2
    rot = np.asarray([[np.cos(c_angle), np.sin(c_angle)], [-np.sin(c_angle), np.cos(c_angle)]])
    co[:2] = rot.dot(co[:2])
    co[1] = np.linalg.norm(co[1:,:], axis=0)

    phi = np.arctan2(co[1], co[0])
    psi = np.clip(phi, 0, angle_diff)
    oc = np.asarray((radius * np.cos(psi),
                     radius * np.sin(psi)))
    poc = np.subtract(co[:2], oc)
    length = np.linalg.norm(poc, axis=0)

    c = np.asarray([np.cos(angle_diff), np.sin(angle_diff)])
    outer = np.outer(c, np.clip( np.dot(co[:2].T, c).T, 0, radius))
    m = np.linalg.norm(co[:2] - outer, axis=0)

    msk = (np.linalg.norm(co[:2], axis=0) <= radius)*(phi <= angle_diff)
    out = np.minimum(m,length)
    out[msk] = -out[msk]

    return out


def sdf_triangle_3d(co, a, b, c):
    # https://iquilezles.org/articles/distfunctions/
    s1 = (b-a)/1
    s2 = (c-b)/1
    s3 = (a-c)/1
    
    coa = np.subtract(co.T, a).T
    cob = np.subtract(co.T, b).T
    coc = np.subtract(co.T, c).T

    normal = np.cross(s1, s3)

    mask = np.sign(np.dot(np.cross(s1, normal), coa)) + np.sign(np.dot(np.cross(s2, normal), cob))
    mask += np.sign(np.dot(np.cross(s3, normal), coc))
    mask = mask < 2.

    term1 = np.outer(s1, np.clip(np.dot(s1, coa[:, mask]) / np.dot(s1, s1), 0, 1) ) - coa[:, mask]
    term2 = np.outer(s2, np.clip(np.dot(s2, cob[:, mask]) / np.dot(s2, s2), 0, 1)) - cob[:, mask]
    term3 = np.outer(s3, np.clip(np.dot(s3, coc[:, mask]) / np.dot(s3, s3), 0, 1)) - coc[:, mask]
    ex1 = np.minimum( np.minimum(np.sum(term1*term1, axis=0), np.sum(term2*term2, axis=0)), np.sum(term3*term3, axis=0) )

    mask2 = np.invert(mask)
    ex2 = (np.dot(normal, coa[:, mask2])**2)/np.dot(normal, normal)

    outer = np.zeros(co.shape[1])
    outer[mask] = np.sqrt(ex1)
    outer[mask2] = np.sqrt(ex2)

    return outer


def sdf_quad_3d(co, a, b, c, d):
    # https://iquilezles.org/articles/distfunctions/
    s1 = (b - a)
    s2 = (c - b)
    s3 = (d - c)
    s4 = (a - d)

    coa = np.subtract(co.T, a).T
    cob = np.subtract(co.T, b).T
    coc = np.subtract(co.T, c).T
    cod = np.subtract(co.T, d).T

    normal = np.cross(s1, s4)

    mask = np.sign(np.dot(np.cross(s1, normal), coa)) + np.sign(np.dot(np.cross(s2, normal), cob))
    mask += np.sign(np.dot(np.cross(s3, normal), coc)) + np.sign(np.dot(np.cross(s4, normal), cod))
    mask = mask < 3.

    term1 = np.outer(s1, np.clip(np.dot(s1, coa[:, mask]) / np.dot(s1, s1), 0, 1)) - coa[:, mask]
    term2 = np.outer(s2, np.clip(np.dot(s2, cob[:, mask]) / np.dot(s2, s2), 0, 1)) - cob[:, mask]
    term3 = np.outer(s3, np.clip(np.dot(s3, coc[:, mask]) / np.dot(s3, s3), 0, 1)) - coc[:, mask]
    term4 = np.outer(s4, np.clip(np.dot(s4, cod[:, mask]) / np.dot(s4, s4), 0, 1)) - cod[:, mask]
    s_ex1 = np.minimum(np.sum(term1 * term1, axis=0), np.sum(term2 * term2, axis=0))
    s_ex2 = np.minimum(np.sum(term4 * term4, axis=0), np.sum(term3 * term3, axis=0))
    ex1 = np.minimum(s_ex2, s_ex1)

    mask2 = np.invert(mask)
    ex2 = (np.dot(normal, coa[:, mask2]) ** 2) / np.dot(normal, normal)

    outer = np.zeros(co.shape[1])
    outer[mask] = np.sqrt(ex1)
    outer[mask2] = np.sqrt(ex2)

    return outer


def sdf_segmented_curve_3d(co, points, t):

    v = np.floor(t).astype(int)
    u = t - v
    fval = points[:, v+1]*u + points[:, v]*(1-u)

    tree = KDTree(fval.T)
    mindist, minid = tree.query(co[:, :].T)
    return mindist


def sdf_segmented_line_3d(co, points):

    out = np.ones(co.shape[1])*10000
    for i in range(points.shape[1]-1):
        part = sdf_segment_3d(co, points[:, i], points[:, i+1])
        out = np.minimum(out, part)

    return out


def sdf_parametric_curve_3d(co, f, f_parameters, t):

    fval = f(t, *f_parameters)

    tree = KDTree(fval.T)
    mindist, minid = tree.query(co[:, :].T)
    return mindist


def sdf_point_cloud_3d(co, points):
    tree = KDTree(points.T)
    mindist, minid = tree.query(co[:3, :].T)
    return mindist

