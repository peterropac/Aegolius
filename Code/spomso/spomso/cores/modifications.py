# Copyright (C) 2023 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from typing import Callable


class ModifyObject:

    def __init__(self, geo_object: Callable[[np.ndarray, tuple], np.ndarray]):
        """
        Class containing all the modifications, which can be applied to the geometry.
        :param geo_object: SDF of a geometry.
        """
        self._mod = []
        self.original_geo_object = geo_object
        self.geo_object = geo_object

    @property
    def modifications(self) -> list:
        """
        All the modifications which were applied to the geometry in chronological order.
        :return: List of modifications
        """
        return self._mod

    @property
    def modified_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        SDF of the modified geometry
        :return: SDF of the modified geometry
        """
        return self.geo_object

    @property
    def original_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        SDF of the unmodified geometry
        :return: SDF of the unmodified geometry
        """
        return self.original_geo_object

    def elongation(self, elon_vector: np.ndarray | tuple | list) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Elongates the geometry along a certain vector by the length of the vector in each respective direction.
        :param elon_vector: 3vector defining the direction and distance of the elongation.
        :return: Modified SDF.
        """
        self._mod.append("elongation")
        elon_vector = np.asarray(elon_vector)/2

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            qo0 = co[0] - np.clip(co[0], -elon_vector[0]/2, elon_vector[0]/2)
            qo1 = co[1] - np.clip(co[1], -elon_vector[1] / 2, elon_vector[1] / 2)
            qo2 = co[2] - np.clip(co[2], -elon_vector[2] / 2, elon_vector[2] / 2)
            qo = np.asarray([qo0, qo1, qo2])
            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def rounding(self, rounding_radius: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rounds off the geometry - effectively thickening it by the rounding radius.
        :param rounding_radius: Radius by which the edges are rounded and the object thickened.
        :return: Modified SDF.
        """
        self._mod.append("rounding")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return geo_object(co, *params) - rounding_radius

        self.geo_object = new_geo_object
        return new_geo_object

    def rounding_cs(self,
                    rounding_radius: float | int,
                    bb_size: int | float) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rounds off the geometry, but the geometry will be contained in its bounding box.
        Bounding box size must be specified.
        :param rounding_radius: Radius by which the edges are rounded.
        :param bb_size: Size of the bounding box. Typically, the largest dimension of the object.
        :return: Modified SDF.
        """
        self._mod.append("rounding_cs")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            scale = (1 - rounding_radius/bb_size)
            return scale*geo_object(co/scale, *params) - rounding_radius

        self.geo_object = new_geo_object
        return new_geo_object

    def boundary(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Get the boundary of a shape.
        :return: Modified SDF.
        """
        self._mod.append("boundary")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.abs(geo_object(co, *params))

        self.geo_object = new_geo_object
        return new_geo_object

    def sign(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Get the sign of the SDF.
        :return: Modified SDF.
        """
        self._mod.append("sign")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.sign(geo_object(co, *params))

        return new_geo_object

    def recover_volume(self,
                      interior: Callable[[np.ndarray, tuple], np.ndarray]) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Recovers the interior of the SDF if a function which outputs the correct value (-1 or 1) is provided.
        This function should take the same parameters as the SDF.
        Typically, this function is the output of self.sign().
        :param interior: Function defining the interior (negative values) of an SDF.
        :return: Modified SDF.
        """
        self._mod.append("recover_volume")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return geo_object(co, *params)*interior(co, *params)

        return new_geo_object

    def define_volume(self,
                      interior: Callable[[np.ndarray, tuple], np.ndarray],
                      interior_parameters: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Defines the interior of the SDF if with a function.
        :param interior: Function defining the interior (-1) and exterior (1) of an SDF.
        :param interior_parameters: Parameters of the function defining the interior of the SDF.
        :return: Modified SDF.
        """
        self._mod.append("define_volume")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return geo_object(co, *params)*interior(co, *interior_parameters)

        return new_geo_object

    def onion(self, thickness: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Transforms the geometry into a surface with some thickness.
        :param thickness: Thickness of the resulting shape.
        :return: Modified SDF.
        """
        self._mod.append("onion")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.abs(geo_object(co, *params)) - thickness

        self.geo_object = new_geo_object
        return new_geo_object

    def concentric(self, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Transforms an isosurface into two concentric isosurfaces which are apart by the value of width.
        Transforms a volume into an isosurface and rounds it by width/2.
        :param width: Thickness of the resulting shape.
        :return: Modified SDF.
        """
        self._mod.append("concentric")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.abs(geo_object(co, *params) - width/2)

        self.geo_object = new_geo_object
        return new_geo_object

    def revolution(self, radius: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Revolves a 2D shape around the y-axis to generate a 3D shape.
        First the 2D shape is translated along the x-axis by the radius of revolution,
        then it is revolved around the y-axis.
        :param radius: Radius of revolution.
        :return: Modified SDF.
        """
        self._mod.append("revolution")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            mag_xz = np.linalg.norm([co[0,:], co[2,:]], axis=0)
            qo = np.zeros(co.shape)
            qo[0] = mag_xz - radius
            qo[1] = co[1]

            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def axis_revolution(self, radius: float | int, angle: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Revolves a 2D shape around an axis to generate a 3D shape.
        First the 2D shape is translated along the x-axis by the radius of revolution,
        then it is revolved around the axis of revolution.
        The axis of revolution is angled by the specified angle with respect to the y-axis.
        :param radius: Radius of revolution.
        :param angle: Angle between the axis of revolution and the y-axis.
        :return: Modified SDF.
        """
        self._mod.append("revolution")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            rot = np.asarray([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
            co[:2, :] = rot.dot(co[:2])

            mag_xz = np.linalg.norm([co[0, :], co[2, :]], axis=0)

            qo = np.zeros(co.shape)
            qo[0] = mag_xz
            qo[1] = co[1]
            qo[:2] = rot.T.dot(qo[:2])
            qo[0] -= radius

            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def extrusion(self, distance: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Extrudes a 2D shape along the z-axis so that the height of the object is equal to the specified distance.
        :param distance: Final height of the object - distance of extrusion.
        :return: Modified SDF.
        """
        self._mod.append("extrusion")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            d = geo_object(co[:2], *params)
            w = np.asarray((d, np.abs(co[2]) - distance/2))

            first_term = np.minimum(np.maximum(w[0], w[1]), 0)
            second_term = np.linalg.norm(np.maximum(w, 0), axis=0)

            return first_term + second_term

        self.geo_object = new_geo_object
        return new_geo_object

    def twist(self, pitch: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Twists the geometry around the z-axis based on the position along the z-axis and the pitch.
        :param pitch: rad/unit length.
        :return: Modified SDF.
        """
        self._mod.append("twist")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = np.cos(pitch * co[2])
            s = np.sin(pitch * co[2])
            rot = np.asarray([[c,s],[-s,c]])
            qo = co.copy()
            qo[:2,:] = rot[0, :, :] * co[0, :] + rot[1, :, :] * co[1, :]

            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def bend2(self, radius, angle):
        # NOT WORKING!!!!!!
        self._mod.append("bend")

        geo_object = self.geo_object

        def new_geo_object(co, *params):


            f = np.clip(co[0]/radius, -angle/2, angle/2)
            c = np.cos(f)
            s = np.sin(f)

            qo = co.copy()
            qo[0] = s*radius - co[0] -s*co[1]
            qo[1] = radius*(1-c) + co[1]*(c-1)
            qo[2] = 0

            bo = co.copy()
            absco = np.abs(co[0])
            sco = np.sign(co[0])
            bo[2] = 0
            bo[0] = sco*(absco - radius*angle/2)*np.cos(angle/2)
            bo[1] = sco*bo[0] * np.tan(angle / 2)

            wo = qo + co + bo*(absco>radius*angle/2)

            # qo[:2, :] = rot[0, :, :] * co[0, :] + rot[1, :, :] * co[1, :]

            return geo_object(co, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def bend(self, radius: float | int, angle: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Bends the geometry around the z-axis, based on the specified bending radius and angle.
        The length of the bent section is radius*angle.
        :param radius: Bending radius.
        :param angle: Bending angle.
        :return: Modified SDF.
        """
        self._mod.append("bend")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = np.cos(angle/2)
            s = np.sin(angle/2)
            rot = np.asarray([[c,s],[-s,c]])

            qo = co.copy()
            qo[1] -= radius
            phi = np.arctan2(qo[0], -qo[1])
            qo[1] = -radius + np.linalg.norm(qo[:2], axis=0)
            qo[0] = radius * phi

            mask1 = radius * angle / 2 <= np.abs(qo[0])
            mask2 = co[0, mask1] >= 0

            wo = co[:2].copy()
            wo = wo[:, mask1]

            sign = np.sign(co[0, mask1])

            wo[0] -= radius*s*sign
            wo[1] -= radius*(1-c)

            wo[:2, mask2] = rot.dot(wo[:2, mask2])
            wo[:2, ~mask2] = rot.T.dot(wo[:2, ~mask2])

            wo[0] += radius * (angle/2) * sign

            qo[:2, mask1] = wo
            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def displacement(self,
                     displacement_function: Callable[[np.ndarray, tuple], np.ndarray],
                     displacement_function_parameters: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Displaces the surface of an object based on the values of the displacement function. Should be applied last.
        :param displacement_function: Function which takes an array of coordinates with shape (D, N) and a tuple of parameters.
        D - number of dimensions (2 or 3);
        N - number of points in the point cloud.
        :param displacement_function_parameters: Parameters of the displacement function.
        :return: Modified SDF.
        """
        self._mod.append("displacement")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            return geo_object(co, *params) + displacement_function(co, *displacement_function_parameters)

        self.geo_object = new_geo_object
        return new_geo_object

    def infinite_repetition(self, distances: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Infinitely repeats geometry in space on a cubic lattice.
        :param distances: 3vector determining the distances between instances along each axis.
        :return: Modified SDF.
        """
        self._mod.append("infinite_repetition")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            redistances = np.asarray(distances)/2
            qo = np.mod(np.add(co.T, redistances), distances) - redistances

            return geo_object(qo.T, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def finite_repetition(self,
                          size: tuple | list | np.ndarray,
                          repetitions: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Repeats the geometry finite amount of times along each axis within a defined bounding box.
        :param size: Size of the bounding box along each axis in which the geometry is repeated.
        :param repetitions: Number of repetitions along each axis inside the bounding box.
        :return: Modified SDF.
        """
        self._mod.append("finite_repetition")

        size = np.asarray(size)
        rep = np.asarray(repetitions)

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = size*(1 - 1/rep)/2
            d = size*(1/2 - 1/rep)
            s = size/rep
            mx = (co[0, :] >= -d[0])*(co[0, :] <= d[0])
            my = (co[1, :] >= -d[1])*(co[1, :] <= d[1])
            mz = (co[2, :] >= -d[2])*(co[2, :] <= d[2])

            # outer
            v = np.abs(co)
            v = np.subtract(v.T, c)
            v[:, 0] -= 2 * v[:, 0] * (co[0, :] < 0)
            v[:, 1] -= 2 * v[:, 1] * (co[1, :] < 0)
            v[:, 2] -= 2 * v[:, 2] * (co[2, :] < 0)

            # inner
            u = np.subtract(co.T, d)
            u[:] = np.mod(u[:], s) - s/2
            v[mx, 0] = u[mx,0]
            v[my, 1] = u[my, 1]
            v[mz, 2] = u[mz, 2]

            return geo_object(v.T, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def finite_repetition_rescaled(self,
                                   size: tuple | list | np.ndarray,
                                   repetitions: tuple | list | np.ndarray,
                                   instance_size: tuple | list | np.ndarray,
                                   padding: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Repeats the geometry finite amount of times along each axis within a defined bounding box.
        The geometry is rescaled based on the provided geometry bounding box size and padding along each axis.
        :param size: Size of the bounding box along each axis in which the geometry is repeated.
        :param repetitions: Number of repetitions along each axis inside the bounding box.
        :param instance_size: Size of the bounding box around one instance of geometry along each axis.
        :param padding: Padding around each instance along each axis.
        :return: Modified SDF.
        """
        self._mod.append("finite_repetition_rescaled")

        size = np.asarray(size)
        padding = np.asarray(padding)
        rep = np.asarray(repetitions)
        f = np.asarray(instance_size)

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = size*(1 - 1/rep)/2
            d = size*(1/2 - 1/rep)
            s = size/rep
            mx = (co[0, :] >= -d[0])*(co[0, :] <= d[0])
            my = (co[1, :] >= -d[1])*(co[1, :] <= d[1])
            mz = (co[2, :] >= -d[2])*(co[2, :] <= d[2])

            # outer
            v = np.abs(co)
            v = np.subtract(v.T, c)
            v[:, 0] -= 2 * v[:, 0] * (co[0, :] < 0)
            v[:, 1] -= 2 * v[:, 1] * (co[1, :] < 0)
            v[:, 2] -= 2 * v[:, 2] * (co[2, :] < 0)

            # inner
            u = np.subtract(co.T, d)
            u[:] = np.mod(u[:], s) - s/2
            v[mx, 0] = u[mx,0]
            v[my, 1] = u[my, 1]
            v[mz, 2] = u[mz, 2]

            sss = np.min(s/(f + padding))
            w = v/sss

            return sss*geo_object(w.T, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def symmetry(self, axis: int):
        """
        Applies symmetry along an axis of the object.
        :param axis: Index of the axis along which the symmetry is applied.
        :return: Modified SDF.
        """
        self._mod.append("symmetry")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            if axis > co.shape[0]:
                return geo_object(co, *params)

            co[axis, :] = np.abs(co[axis, :])
            return geo_object(co, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def mirror(self, a: tuple | list | np.ndarray, b: tuple | list | np.ndarray):
        """
        Applies mirroring along an axis connecting the two specified points.
        :param a: Position of the object's mirror image.
        :param b: Position of the object.
        :return: Modified SDF.
        """
        self._mod.append("mirror")

        a = np.asarray(a)
        b = np.asarray(b)
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            w = b - a
            c = (b + a) / 2
            l = np.linalg.norm(w)

            x = w / l
            y = np.asarray([-x[1], x[0], 0])
            y = y / np.linalg.norm(y)
            z = np.cross(x, y)
            rot = np.asarray([x, y, z])
            co = np.subtract(co.T, c).T
            co = rot.dot(co)

            # outer
            v = np.zeros(co.shape)
            v[1:, :] = co[1:]
            v[0, :] = np.abs(co[0, :]) - l / 2

            return geo_object(v, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def rotational_symmetry(self,
                            n: int,
                            radius: float | int,
                            phase: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies n-fold rotational symmetry to the object.
        :param n: Order of the symmetry.
        :param radius: By how much the geometry is moved from the centre.
        :param phase: Angle by which the circular pattern is rotated by.
        :return: Modified SDF.
        """
        self._mod.append("rotational_symmetry")

        angle = 2*np.pi/n
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            rot = np.asarray([[np.cos(angle/2 - phase), np.sin(angle/2 - phase)],
                              [-np.sin(angle/2 - phase), np.cos(angle/2 - phase)]])
            co[:2] = rot.dot(co[:2])
            phi = np.arctan2(co[1], co[0])
            phi[phi < 0] = 2 * np.pi + phi[phi < 0]
            phi = np.mod(phi, angle) - angle/2
            radii = np.linalg.norm(co[:2,:], axis=0)
            co[0] = radii*np.cos(phi) - radius
            co[1] = radii*np.sin(phi)

            return geo_object(co, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def linear_instancing(self,
                          n: int,
                          a: tuple | list | np.ndarray,
                          b: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a segment defined by two points.
        :param n: Number of instances along the segment.
        :param a: Starting point of the segment.
        :param b: Ending point of the segment.
        :return: Modified SDF.
        """
        self._mod.append("linear_instancing")

        a = np.asarray(a)
        b = np.asarray(b)
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            w = b-a
            c = (b + a) / 2
            l = np.linalg.norm(w)

            x = w/l
            y = np.asarray([-x[1], x[0], 0])
            y = y/np.linalg.norm(y)
            z = np.cross(x,y)
            rot = np.asarray([x,y,z])
            co = np.subtract(co.T, c).T
            co = rot.dot(co)

            s = l/(n-1)
            d = s/2
            mx = (co[0, :] >= -l/2 + d) * (co[0, :] <= l/2 - d)

            # outer
            v = np.zeros(co.shape)
            v[1:, :] = co[1:]
            v[0, :] = np.abs(co[0,:]) - l/2
            v[0, :] -= 2*v[0, :]*(co[0, :] < 0)

            # inner
            if n > 2:
                u = np.subtract(co.T, [l/2 - d, 0, 0]).T
                u[:] = np.mod(u[:], s) - d
                v[0, mx] = u[0, mx]

            return geo_object(v, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def curve_instancing(self,
                         f: Callable[[np.ndarray, tuple], np.ndarray],
                         f_parameters: tuple,
                         t_range: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a parametric curve defined by a function.
        :param f: Function defining the parametric curve.
        :param f_parameters: Parameters of the function defining the parametric curve.
        :param t_range: (t_start, t_end, n_instances):
        t_start - initial value of the parameter t, which is feed into the parametric curve.
        t_end - final value of the parameter t.
        n_instances -  number of instances of the geometry along the parametric curve.
        :return: Modified SDF.
        """
        self._mod.append("parametric_curve_instancing")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            ts = np.linspace(*t_range)
            fval = f(ts, *f_parameters)
            va = np.zeros((3, t_range[-1]))
            va[:fval.shape[0]] = fval
            ix = np.arange(0, t_range[-1])

            interp = NearestNDInterpolator(va.T, ix)
            values = interp(co[0], co[1], co[2])

            w = np.zeros(co.shape)
            for i in range(t_range[-1]):
                v = np.subtract(co.T, va[:, i]).T
                mask = values==i
                w[:, mask] = v[:, mask]

            return geo_object(w, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def aligned_curve_instancing(self,
                         f: Callable[[np.ndarray, tuple], np.ndarray],
                         f_parameters: tuple,
                         t_range: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a parametric curve defined by a function.
        The x-axis of the geometry is aligned with the tangent vector of the parametric curve.
        :param f: Function defining the parametric curve.
        :param f_parameters: Parameters of the function defining the parametric curve.
        :param t_range: (t_start, t_end, n_instances):
        t_start - initial value of the parameter t, which is feed into the parametric curve.
        t_end - final value of the parameter t.
        n_instances -  number of instances of the geometry along the parametric curve.
        :return: Modified SDF.
        """
        # approximation with the binormal (dy) and normal (dz) vector!!!
        self._mod.append("aligned_parametric_curve_instancing")

        geo_object = self.geo_object
        tol = 0.001

        def new_geo_object(co, *params):

            ts = np.linspace(*t_range)
            ts_min = ts-tol
            ts_max = ts+tol

            fval = f(ts, *f_parameters)
            va = np.zeros((3, t_range[-1]))
            va[:fval.shape[0]] = fval
            ix = np.arange(0, t_range[-1])

            fv_min = f(ts_min, *f_parameters)
            fv_max = f(ts_max, *f_parameters)
            der = (fv_max-fv_min)/(2*tol)
            der = der/np.linalg.norm(der, axis=0)

            dx = np.zeros((3, t_range[-1]))
            dx[:fval.shape[0]] = der
            dy = np.zeros((3, t_range[-1]))
            dy[0, :] = -dx[1]
            dy[1, :] = dx[0]
            dz = np.cross(dx.T, dy.T).T
            trot = np.asarray([dx, dy, dz])

            interp = NearestNDInterpolator(va.T, ix)
            values = interp(co[0], co[1], co[2])

            w = np.zeros(co.shape)
            for i in range(t_range[-1]):
                v = np.subtract(co.T, va[:, i]).T
                v = trot[:,:,i].dot(v)
                mask = values==i
                w[:, mask] = v[:, mask]

            return geo_object(w, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def fully_aligned_curve_instancing(self,
                         f: Callable[[np.ndarray, tuple], np.ndarray],
                         f_parameters: tuple,
                         t_range: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a parametric curve defined by a function.
        The x-axis of the geometry is aligned with the tangent vector of the parametric curve.
        The y-axis of the geometry is aligned with the normal vector of the parametric curve.
        The z-axis of the geometry is aligned with the binormal vector of the parametric curve.
        :param f: Function defining the parametric curve.
        :param f_parameters: Parameters of the function defining the parametric curve.
        :param t_range: (t_start, t_end, n_instances):
        t_start - initial value of the parameter t, which is feed into the parametric curve.
        t_end - final value of the parameter t.
        n_instances -  number of instances of the geometry along the parametric curve.
        :return: Modified SDF.
        """
        self._mod.append("fully_aligned_parametric_curve_instancing")

        geo_object = self.geo_object
        tol = 0.001

        def new_geo_object(co, *params):

            ts = np.linspace(*t_range)
            ts_min = ts-tol
            ts_max = ts+tol

            fval = f(ts, *f_parameters)
            va = np.zeros((3, t_range[-1]))
            va[:fval.shape[0]] = fval
            ix = np.arange(0, t_range[-1])

            fv_min = f(ts_min, *f_parameters)
            fv_max = f(ts_max, *f_parameters)
            der = (fv_max-fv_min)/(2*tol)
            dermag = np.linalg.norm(der, axis=0)
            der = der/dermag

            der2 = (fv_max - 2*fval + fv_min)/(tol**2)
            der2 = der2/dermag
            der2mag = np.linalg.norm(der2, axis=0)
            der2 = der2/der2mag

            dx = np.zeros((3, t_range[-1]))
            dx[:fval.shape[0]] = der
            dy = np.zeros((3, t_range[-1]))
            dy[:fval.shape[0]] = der2
            dz = np.cross(dx.T, dy.T).T
            trot = np.asarray([dx, dy, dz])

            interp = NearestNDInterpolator(va.T, ix)
            values = interp(co[0], co[1], co[2])

            w = np.zeros(co.shape)
            for i in range(t_range[-1]):
                v = np.subtract(co.T, va[:, i]).T
                v = trot[:,:,i].dot(v)
                mask = values==i
                w[:, mask] = v[:, mask]

            return geo_object(w, *params)

        self.geo_object = new_geo_object
        return new_geo_object
