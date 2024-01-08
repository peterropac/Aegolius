# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from spomso.cores.vector_modification_functions import batch_normalize
from spomso.cores.helper_functions import smarter_reshape

# ----------------------------------------------------------------------------------------------------------------------
# VECTOR INITIALIZATION FUNCTIONS


def cartesian_define(p):
    ux = p[0]
    uy = p[1]
    uz = p[2]

    return np.asarray((ux, uy, uz))


def spherical_define(p):
    r = p[0]
    phi = p[1]
    theta = p[2]

    u = r*np.cos(phi)*np.sin(theta)
    v = r*np.sin(phi)*np.sin(theta)
    w = r*np.cos(theta)

    return np.asarray((u, v, w))


def cylindrical_define(p):
    r = p[0]
    phi = p[1]
    z = p[2]

    u = r*np.cos(phi)
    v = r*np.sin(phi)

    return np.asarray((u, v, z))


def radial_vector_field_spherical(r, *p):
    r = r.copy()
    return batch_normalize(r)


def radial_vector_field_cylindrical(r, *p):
    r = r.copy()
    r[2] = 0
    v = batch_normalize(r)
    return v


def hyperbolic_vector_field_cylindrical(r, *p):
    alpha = -np.arctan2(r[1], r[0])
    v = cylindrical_define(1, alpha, np.zeros(r.shape[1]))
    return v


def awn_vector_field_cylindrical(r, gamma):
    alpha = gamma*np.arctan2(r[1], r[0])
    print(alpha.shape, r.shape)
    v = cylindrical_define(1, np.squeeze(alpha), np.zeros(r.shape[1]))
    return v


def vortex_vector_field_cylindrical(r, *p):
    r = r.copy()
    r[2] = 0
    v = batch_normalize(r)

    w = v.copy()
    w[0] = -v[1]
    w[1] = v[0]
    return w


def aar_vector_field_cylindrical(r, alpha):
    r = r.copy()
    r[2] = 0
    vec = batch_normalize(r)
    print(alpha)
    sa = np.squeeze(np.sin(alpha))
    ca = np.squeeze(np.cos(alpha))

    o = np.asarray([vec[0, :] * ca - vec[1, :] * sa,
                    vec[0, :] * sa + vec[1, :] * ca,
                    vec[2, :]])

    return o


def aav_vector_field_cylindrical(r, alpha):
    r = r.copy()
    r[2] = 0
    vec = batch_normalize(r)
    print(alpha)
    sa = np.squeeze(np.sin(alpha))
    ca = np.squeeze(np.cos(alpha))

    o = np.asarray([-vec[0, :] * sa - vec[1, :] * ca,
                    vec[0, :] * ca - vec[1, :] * sa,
                    vec[2, :]])

    return o


def x_vector_field(r, *p):
    vec = np.zeros(r.shape)
    vec[0] = 1
    return vec


def y_vector_field(r, *p):
    vec = np.zeros(r.shape)
    vec[1] = 1
    return vec


def z_vector_field(r, *p):
    vec = np.zeros(r.shape)
    vec[2] = 1
    return vec


def from_sdf(sdf_, co_resolution):

    dimensions = np.asarray(co_resolution).shape[0]
    gsdf = smarter_reshape(sdf_, co_resolution)

    vec = np.asarray(np.gradient(gsdf))
    vec = vec.reshape(dimensions, -1)

    v = batch_normalize(vec)
    return v
