# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from spomso.cores.helper_functions import smarter_reshape
from spomso.cores.vector_modification_functions import batch_normalize
from spomso.cores.vector_modification_functions import rotate_vectors_axis
from spomso.cores.post_processing import conv_averaging


def compute_crossings_2d(sdf_grid: np.ndarray, thr: float = 0.06) -> np.ndarray:
    """
    Calculates where boundaries in the SDF and separates the regions by assigning the a +1 or -1 value.
    :param sdf_grid: Signed Distance field or any scalar field.
    :param thr: Threshold by which the regions are separated.
    :return: Modified scalar field.
    """
    cross = np.zeros(sdf_grid.shape)

    s = sdf_grid[:, :]
    min_ = np.amin(s)
    c1 = np.ones(sdf_grid.shape)

    c1[:, 1:] = np.isclose(s[:, 1:], min_, atol=thr)*(~np.isclose(s[:, :-1], min_, atol=thr))

    for j in range(1, sdf_grid.shape[1]):
        cross[:, j] = 1*c1[:, j] + cross[:, j-1]

    o = np.mod(cross,2).T
    o = conv_averaging(o, 5, 2) >= 0.5
    o = 2 * o.T - 1

    return o


def lcwg1_2d_old(r, uu, p):

    w = p

    o = np.zeros((3, uu.shape[0]))

    l1 = o.copy()
    l1[0, :] = 1

    qq = np.clip(uu, 0, 1)

    l4 = o.copy()
    l4[0, :] = -np.cos(np.pi * qq)
    l4[1, :] = -np.sin(np.pi * qq)
    l4[:] = l4 / np.linalg.norm(l4, axis=0)

    wg_mask = np.abs(r[1]) <= w / 2

    o[:] = l1
    o[:, wg_mask] = l4[:, wg_mask]

    return o


def lcwg1_p1_old(r, uu, p):

    w = p[0]
    d = p[1]

    alpha = np.arctan2(2 * r[2] / (d ** 2), 8 * r[1] / (w ** 2))
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.zeros((3, uu.shape[0]))

    l1 = o.copy()
    l1[0, :] = 1

    qq = np.clip(uu, 0, 1)

    l2 = -np.sin(np.pi * qq)

    l4 = o.copy()
    l4[0, :] = -np.cos(np.pi * qq)
    l4[1, :] = l2 * ca
    l4[2, :] = l2 * sa
    l4[:] = l4 / np.linalg.norm(l4, axis=0)

    wg_mask = np.abs(r[1]) <= w / 2

    o[:] = l1
    o[:, wg_mask] = l4[:, wg_mask]

    return o


def lcwg1_m1_old(r, uu, p):

    w = p[0]
    d = p[1]

    alpha = -np.arctan2(2 * r[2] / (d ** 2), 8 * r[1] / (w ** 2))
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.zeros((3, uu.shape[0]))

    l1 = o.copy()
    l1[0, :] = 1

    qq = np.clip(uu, 0, 1)

    l2 = -np.sin(np.pi * qq)

    l4 = o.copy()
    l4[0, :] = -np.cos(np.pi * qq)
    l4[1, :] = l2 * ca
    l4[2, :] = l2 * sa
    l4[:] = l4 / np.linalg.norm(l4, axis=0)

    wg_mask = np.abs(r[1]) <= w / 2

    o[:] = l1
    o[:, wg_mask] = l4[:, wg_mask]

    return o


def lcwg1_2d(uu, p, co_resolution, sign):

    w = p

    dimensions = np.asarray(co_resolution).shape[0]
    gsdf = smarter_reshape(uu, co_resolution)

    vec = np.asarray(np.gradient(gsdf))
    vec = vec.reshape(dimensions, -1)

    # avec = np.abs(vec)
    # sign = np.sign(np.sum(np.multiply(vec, avec), axis=0))

    if sign is None or isinstance(sign, float):
        if isinstance(sign, float):
            s = compute_crossings_2d(gsdf[:, :, 0], thr=abs(sign))
        else:
            s = compute_crossings_2d(gsdf[:, :, 0])
        sign = np.repeat(s[:, :, np.newaxis], co_resolution[2], axis=2).flatten()

    qq = np.clip(2 * uu / w, 0, 1)
    phis = sign*(qq*np.pi + np.pi/2)
    sap = np.sin(phis)
    cap = np.cos(phis)

    l4 = vec.copy()
    l4 = np.asarray([l4[0, :] * cap - l4[1, :] * sap,
                     l4[0, :] * sap + l4[1, :] * cap,
                     l4[2, :]])

    o = np.zeros((3, uu.shape[0]))
    o[:dimensions, :] = l4
    o = batch_normalize(o)

    return o


def lcwg1_p1(uuww, p, co_resolution, sign):

    uu = uuww[0]
    ww = uuww[1]

    w = p[0]
    d = p[1]

    dimensions = np.asarray(co_resolution).shape[0]

    pp = np.linalg.norm([2 * uu / w, ww / d], axis=0)
    gsdf = smarter_reshape(pp, co_resolution)

    vec = np.asarray(np.gradient(gsdf))
    vec = vec.reshape(dimensions, -1)
    vec = batch_normalize(vec)

    e1 = vec.copy()
    e1[0] = -vec[1]
    e1[1] = vec[0]
    e1[2] = 0
    e1 = batch_normalize(e1)
    e2 = np.cross(vec.T, e1.T).T

    # avec = np.abs(vec)
    # sign = np.sign(np.sum(np.multiply(vec, avec), axis=0))

    if sign is None or isinstance(sign, float):
        if isinstance(sign, float):
            s = compute_crossings_2d(gsdf[:, :, 0], thr=abs(sign))
        else:
            s = compute_crossings_2d(gsdf[:, :, 0])
        sign = np.repeat(s[:, :, np.newaxis], gsdf.shape[2], axis=2).flatten()

    qq = np.clip(pp, 0, 1)
    phis = sign * (+qq * np.pi + np.pi / 2)
    vec2 = rotate_vectors_axis(vec, e2, phis)

    o = np.zeros((3, uu.shape[0]))
    alpha = np.arctan2(2 * ww / (d ** 2), 8 * uu / (w ** 2))
    o[:dimensions, :] = rotate_vectors_axis(vec2, e1, -2 * alpha)
    o = batch_normalize(o)

    return o


def lcwg1_m1(uuww, p, co_resolution, sign):

    uu = uuww[0]
    ww = uuww[1]

    w = p[0]
    d = p[1]

    dimensions = np.asarray(co_resolution).shape[0]

    pp = np.linalg.norm([2 * uu / w, ww / d], axis=0)
    gsdf = smarter_reshape(pp, co_resolution)

    vec = np.asarray(np.gradient(gsdf))
    vec = vec.reshape(dimensions, -1)
    vec = batch_normalize(vec)

    e1 = vec.copy()
    e1[0] = -vec[1]
    e1[1] = vec[0]
    e1[2] = 0
    e1 = batch_normalize(e1)
    e2 = np.cross(vec.T, e1.T).T

    if sign is None or isinstance(sign, float):
        if isinstance(sign, float):
            s = compute_crossings_2d(gsdf[:, :, 0], thr=abs(sign))
        else:
            s = compute_crossings_2d(gsdf[:, :, 0])
        sign = np.repeat(s[:, :, np.newaxis], gsdf.shape[2], axis=2).flatten()

    qq = np.clip(pp, 0, 1)
    phis = sign * (+qq * np.pi + np.pi / 2)
    vec2 = rotate_vectors_axis(vec, e2, phis)

    o = np.zeros((3, uu.shape[0]))
    alpha = -np.arctan2(2 * ww / (d ** 2), 8 * uu / (w ** 2))
    o[:dimensions, :] = rotate_vectors_axis(vec2, e1, -2*alpha)
    o = batch_normalize(o)

    return o

