# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from typing import Callable

# ----------------------------------------------------------------------------------------------------------------------
# VECTOR TRANSFORM FUNCTIONS


def batch_normalize(vec):

    m = np.linalg.norm(vec, axis=0)
    mask = ~(m == 0)
    vec[:, mask] = vec[:, mask]/m[mask]

    return vec


def add_vectors(vec, add_vec):
    add_vec = np.asarray(add_vec)
    if add_vec.size == 3:
        return np.add(vec.T, add_vec).T
    else:
        return np.add(vec, add_vec)


def subtract_vectors(vec, subtract_vec):
    subtract_vec = np.asarray(subtract_vec)
    if subtract_vec.size == 3:
        return np.add(vec.T, subtract_vec).T
    else:
        return np.add(vec, subtract_vec)


def rescale_vectors(vec, scale):

    return np.multiply(vec, scale)


def rotate_vectors_phi(vec, phis):
    sa = np.sin(phis)
    ca = np.cos(phis)

    o = np.asarray([vec[0, :] * ca - vec[1, :] * sa,
                    vec[0, :] * sa + vec[1, :] * ca,
                    vec[2, :]])

    return o


def rotate_vectors_theta(vec, thetas):

    rvec = vec.copy()
    rvec[2] = 0
    rvec = batch_normalize(rvec)

    ca = np.cos(thetas)
    sa = np.sin(thetas)

    term2 = np.asarray([rvec[0]*vec[2],
                        rvec[1]*vec[2],
                        -rvec[0]*vec[0] - rvec[1]*vec[1]])
    out = vec*ca + term2*sa
    return out


def rotate_vectors_x_axis(vec, alpha):
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.asarray([vec[0, :],
                    vec[1, :] * ca - vec[2, :] * sa,
                    vec[1, :] * sa + vec[2, :] * ca,
                    ])

    return o


def rotate_vectors_y_axis(vec, alpha):
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.asarray([vec[0, :] * ca - vec[2, :] * sa,
                    vec[1, :],
                    vec[0, :] * sa + vec[2, :] * ca,
                    ])

    return o


def rotate_vectors_z_axis(vec, alpha):
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.asarray([vec[0, :] * ca - vec[1, :] * sa,
                    vec[0, :] * sa + vec[1, :] * ca,
                    vec[2, :]])

    return o


def rotate_vectors_axis(vec, axes, alpha):
    axes = np.asarray(axes)
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    t1 = vec*ca
    t2 = sa*np.cross(axes.T, vec.T).T
    if axes.size==3:
        t3 = (1 - ca) * np.outer(axes, (np.sum(np.multiply(vec.T, axes.T).T, axis=0)))
    else:
        t3 = (1 - ca) * axes * (np.sum(np.multiply(vec.T, axes.T).T, axis=0))
    o = t1 + t2 + t3

    return o


def revolve_field_x(r, vec):

    alpha = np.arctan2(r[2], r[1])
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.zeros(r.shape)
    o[0, :] = vec[0, :]
    o[1, :] = vec[1, :] * ca - vec[2, :] * sa
    o[2, :] = vec[1, :] * sa + vec[2, :] * ca

    return o


def revolve_field_y(r, vec):
    alpha = np.arctan2(r[2], r[0])
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.zeros(r.shape)
    o[0, :] = vec[0, :] * ca - vec[2,:] * sa
    o[1, :] = vec[1, :]
    o[2, :] = vec[0, :] * sa + vec[2, :] * ca

    return o


def revolve_field_z(r, vec):
    alpha = np.arctan2(r[1], r[0])
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    o = np.zeros(r.shape)
    o[0, :] = vec[0, :]*ca - vec[1, :]*sa
    o[1, :] = vec[0, :] * sa + vec[1, :] * ca
    o[2, :] = vec[2, :]

    return o

