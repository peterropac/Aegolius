# Copyright (C) 2023 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from typing import Callable
from spomso.cores.geom import GenericGeometry
from spomso.cores.transformations import EuclideanTransform
from spomso.cores.modifications import ModifyObject
from spomso.cores.sdf_2D import sdf_circle, sdf_box_2d, sdf_segment_2d, sdf_triangle_2d, sdf_rounded_box_2d
from spomso.cores.sdf_2D import sdf_sector, sdf_inf_sector, sdf_sector_old, sdf_ngon, sdf_arc
from spomso.cores.sdf_2D import sdf_parametric_curve_2d, sdf_segmented_curve_2d, sdf_segmented_line_2d
from spomso.cores.sdf_2D import sdf_point_cloud_2d


class GenericGeometry2D(EuclideanTransform, ModifyObject):

    def __init__(self, geo_sdf, *geo_parameters):
        EuclideanTransform.__init__(self)
        ModifyObject.__init__(self, geo_sdf)
        self.geo_parameters = geo_parameters
        self._sdf = geo_sdf

    def create(self, co):
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, self.geo_parameters)

    def propagate(self, co, *parameters_):
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, self.geo_parameters)


class Circle(GenericGeometry):

    def __init__(self, radius: float | int):
        """
        Circle defined by its radius.
        :param radius: Radius of the circle.
        """
        GenericGeometry.__init__(self, sdf_circle, radius)
        self._radius = radius

    @property
    def radius(self):
        return self._radius


class NGon(GenericGeometry):

    def __init__(self, radius: float | int, n_sides: int):
        """
        N-sided regular polygon, defined by the outer radius and the number of sides.
        :param radius: Outer radius of the regular polygon.
        :param n_sides: Number of sides of the regular polygon.
        """
        GenericGeometry.__init__(self, sdf_ngon, radius, n_sides)
        self._radius = radius
        self._n_sides = n_sides
    @property
    def radius(self):
        return self._radius

    @property
    def n_sides(self):
        return self._n_sides


class Rectangle(GenericGeometry):

    def __init__(self, a: float | int, b: float | int):
        """
        Rectangle defined by its side lengths.
        :param a: Side length along the x-axis.
        :param b: Side length along the y-axis.
        """
        GenericGeometry.__init__(self, sdf_box_2d, (a/2, b/2))
        self._a = a/2
        self._b = b/2

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def size(self):
        return np.asarray((self._a, self._b))


class RoundedRectangle(GenericGeometry):

    def __init__(self, a: float, b: float, rounding: tuple | list | np.ndarray):
        """
        Rounded rectangle defined by its side lengths and rounding radii for each corner.
        :param a: Side length along the x-axis.
        :param b: Side length along the y-axis.
        :param rounding: rounding radii for each corner.
        """
        GenericGeometry.__init__(self, sdf_rounded_box_2d, (a/2, b/2), rounding[:4])
        self._a = a/2
        self._b = b/2
        self._round_corners = rounding[:4]

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def size(self):
        return np.asarray((self._a, self._b))

    @property
    def round_corners(self):
        return np.asarray(self._round_corners)


class Segment(GenericGeometry):

    def __init__(self, a: tuple | list | np.ndarray, b: tuple | list | np.ndarray):
        """
        Line segment defined by its end points.
        :param a: Vector defining the position of the start point.
        :param b: Vector defining the position of the end point.
        """
        GenericGeometry.__init__(self, sdf_segment_2d, a, b)
        self._a = np.asarray(a)
        self._b = np.asarray(b)

    @property
    def point_a(self):
        return self._a

    @property
    def point_b(self):
        return self._b


class Triangle(GenericGeometry):

    def __init__(self, a: tuple | list | np.ndarray, b: tuple | list | np.ndarray, c: tuple | list | np.ndarray):
        """
        Triangle defined by the three vertices.
        :param a: Vector defining the position of the first vertex.
        :param b: Vector defining the position of the second vertex.
        :param c: Vector defining the position of the third vertex.
        """
        GenericGeometry.__init__(self, sdf_triangle_2d, np.asarray(a), np.asarray(b), np.asarray(c))
        self._a = np.asarray(a)
        self._b = np.asarray(b)
        self._c = np.asarray(c)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c


class SectorOld(GenericGeometry):

    def __init__(self, radius, angle):
        GenericGeometry.__init__(self, sdf_sector_old, radius, angle/2)
        self._radius = radius
        self._angle = angle/2

    @property
    def radius(self):
        return self._radius

    @property
    def angle(self):
        return self._angle


class Sector(GenericGeometry):

    def __init__(self, radius: float | int, angle_1: float | int, angle_2: float | int):
        """
        Sector defined by the radius of the circle and two angles.
        :param radius: Radius of the circle.
        :param angle_1: First angle defining the sector.
        :param angle_2: Second angle defining the sector.
        """
        GenericGeometry.__init__(self, sdf_sector, radius, angle_1, angle_2)
        self._radius = radius
        self._angle_1 = angle_1
        self._angle_2 = angle_2

    @property
    def radius(self):
        return self._radius

    @property
    def angle_1(self):
        return self._angle_1

    @property
    def angle_2(self):
        return self._angle_2


class InfiniteSector(GenericGeometry):

    def __init__(self, angle_1, angle_2):
        """
        Sector of infinite radius defined by two angles.
        :param angle_1: First angle defining the sector.
        :param angle_2: Second angle defining the sector.
        """
        GenericGeometry.__init__(self, sdf_inf_sector, angle_1, angle_2)
        self._angle_1 = angle_1
        self._angle_2 = angle_2

    @property
    def angle_1(self):
        return self._angle_1

    @property
    def angle_2(self):
        return self._angle_2


class Arc(GenericGeometry):

    def __init__(self, radius: float, start_angle: float, end_angle: float):
        """
        Arc defined by the radius and the angles of both ends.
        :param radius: Radius of the arc.
        :param start_angle: Angle of one end with respect to the x-axis.
        :param end_angle: Angle of the other end with respect to the x-axis.
        """
        GenericGeometry.__init__(self, sdf_arc, radius, start_angle, end_angle)
        self._radius = radius
        self._sangle = start_angle
        self._eangle = end_angle

    @property
    def radius(self):
        return self._radius

    @property
    def start_angle(self):
        return self._sangle

    @property
    def end_angle(self):
        return self._eangle


class ParametricCurve(GenericGeometry):

    def __init__(self,
                 parametric_curve: Callable[[np.ndarray, tuple], np.ndarray],
                 parametric_curve_parameters: tuple,
                 t_range: tuple,
                 closed: bool = False):
        """
        Curve connecting points on the user provided parametric curve.
        :param parametric_curve: Function defining the parametric curve with a single variable (t) and other parameters.
        :param parametric_curve_parameters: Parameters of the function defining the parametric curve.
        :param t_range: Range of the t parameter.
        t_range[0] -  start, t_range[1] - end, t_range[2] -  number of steps in between.
        :param closed: Is the curve closed on itself. False by default.
        """
        self._curve = parametric_curve
        self._c_params = parametric_curve_parameters
        self._t_range = t_range
        self._closed = closed

        GenericGeometry.__init__(self,
                                 self.sdf_closed_curve(),
                                 parametric_curve,
                                 parametric_curve_parameters,
                                 self.ts)

    def sdf_closed_curve(self):
        def new_geo_object(co, f, f_parameters, t):

            p0 = f(t[0], *f_parameters)
            p1 = f(t[-1], *f_parameters)

            f1 = sdf_parametric_curve_2d(co, f, f_parameters, t)
            f2 = sdf_segment_2d(co, p0, p1)

            return np.minimum(f1, f2)

        if self.closed:
            return new_geo_object
        else:
            return sdf_parametric_curve_2d

    @property
    def steps(self):
        return self._t_range[2]

    @property
    def t_start(self):
        return self._t_range[0]

    @property
    def t_end(self):
        return self._t_range[1]

    @property
    def ts(self):
        return np.linspace(*self._t_range)

    @property
    def closed(self):
        return self._closed

    def shape(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Converts the closed parametric curve into a shape.
        :return: Modified SDF.
        """
        if not self.closed:
            return self.geo_object

        self._mod.append("shape")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            d = geo_object(co, *params)
            if np.any(d < 0):
                return geo_object(co, *params)

            ts_ = np.zeros(self.steps+1)
            ts_[:self.steps] = self.ts

            points = self._curve(ts_, *self._c_params)

            interior = np.ones(co.shape[1])
            for i in range(points.shape[1] - 1):
                t = points[:, i + 1] - points[:, i]
                t = t / np.linalg.norm(t)
                n = np.asarray([-t[1], t[0], 0])
                n[1] = n[1] - 2 * (n[1] < 0) * n[1]

                lx = np.minimum(points[0, i], points[0, i + 1])
                ux = np.maximum(points[0, i], points[0, i + 1])
                mask = (co[0, :] >= lx) * (co[0, :] < ux)

                s = np.sign(np.dot(co[:2, mask].T - points[:2, i], n[:2]))
                interior[mask] *= s

            return geo_object(co, *params)*interior

        self.geo_object = new_geo_object
        return new_geo_object


class SegmentedParametricCurve(GenericGeometry):

    def __init__(self, points: list | tuple | np.ndarray, t_range: tuple, closed: bool = False):
        """
        Segmented line connecting the user provided points.
        :param points: Points to connect.
        :param t_range: Range of the t parameter.
        t_range[0] -  start, t_range[1] - end, t_range[2] -  number of steps in between.
        t_range[1]-t_range[0] >= number of points.
        :param closed: Is the segmented line closed on itself. False by default.
        """
        self._points = np.asarray(points)
        if self._points.shape[1] < self._points.shape[0]:
            self._points = self._points.T
        self._t_range = t_range
        self._closed = closed

        GenericGeometry.__init__(self,
                                 self.sdf_closed_curve(),
                                 self._points,
                                 self.ts)

    def sdf_closed_curve(self):
        def new_geo_object(co, points, t):

            f1 = sdf_segmented_curve_2d(co, points, t)
            f2 = sdf_segment_2d(co, points[:, 0], points[:, -1])

            return np.minimum(f1, f2)

        if self.closed:
            return new_geo_object
        else:
            return sdf_segmented_curve_2d

    @property
    def steps(self):
        return self._t_range[2]

    @property
    def t_start(self):
        return self._t_range[0]

    @property
    def t_end(self):
        return self._t_range[1]

    @property
    def ts(self):
        tt = np.linspace(self._t_range[0], self._t_range[1]-1, self._t_range[2])
        ts_ = np.clip(tt, 0, self._points.shape[1]-1.0001)
        return ts_

    @property
    def closed(self):
        return self._closed

    def polygon(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Converts the closed segmented line into a polygon.
        :return: Modified SDF.
        """
        if not self.closed:
            return self.geo_object

        self._mod.append("polygon")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            d = geo_object(co, *params)
            if np.any(d < 0):
                return geo_object(co, *params)

            points = np.zeros((self._points.shape[0], self._points.shape[1] + 1))
            points[:, :self._points.shape[1]] = self._points
            points[:, -1] = self._points[:, 0]

            interior = np.ones(co.shape[1])
            for i in range(points.shape[1] - 1):
                t = points[:, i + 1] - points[:, i]
                t = t / np.linalg.norm(t)
                n = np.asarray([-t[1], t[0], 0])
                n[1] = n[1] - 2 * (n[1] < 0) * n[1]

                lx = np.minimum(points[0, i], points[0, i + 1])
                ux = np.maximum(points[0, i], points[0, i + 1])
                mask = (co[0, :] >= lx) * (co[0, :] < ux)

                s = np.sign(np.dot(co[:, mask].T - points[:, i], n))
                interior[mask] *= s

            return geo_object(co, *params) * interior

        self.geo_object = new_geo_object
        return new_geo_object


class SegmentedLine(GenericGeometry):

    def __init__(self, points: list | tuple | np.ndarray, closed: bool = False):
        """
        Segmented line connecting the user provided points.
        :param points: Points to connect.
        :param closed: Is the segmented line closed on itself. False by default.
        """
        self._points = np.asarray(points)
        if self._points.shape[1] < self._points.shape[0]:
            self._points = self._points.T
        self._closed = closed

        GenericGeometry.__init__(self,
                                 self.sdf_closed_curve(),
                                 self._points)

    def sdf_closed_curve(self):
        def new_geo_object(co, points):

            f1 = sdf_segmented_line_2d(co, points)
            f2 = sdf_segment_2d(co, points[:, 0], points[:, -1])

            return np.minimum(f1, f2)

        if self.closed:
            return new_geo_object
        else:
            return sdf_segmented_line_2d

    @property
    def closed(self):
        return self._closed

    def polygon(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Converts the closed segmented line into a polygon.
        :return: Modified SDF.
        """
        if not self.closed:
            return self.geo_object

        self._mod.append("polygon")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            d = geo_object(co, *params)
            if np.any(d<0):
                return geo_object(co, *params)

            points = np.zeros((self._points.shape[0], self._points.shape[1] + 1))
            points[:, :self._points.shape[1]] = self._points
            points[:, -1] = self._points[:, 0]

            interior = np.ones(co.shape[1])
            for i in range(points.shape[1]-1):
                t = points[:, i+1] - points[:, i]
                t = t/np.linalg.norm(t)
                n = np.asarray([-t[1], t[0], 0])
                n[1] = n[1] - 2*(n[1] < 0)*n[1]

                lx = np.minimum(points[0, i], points[0, i+1])
                ux = np.maximum(points[0, i], points[0, i + 1])
                mask = (co[0, :] >= lx) * (co[0, :] < ux)

                s = np.sign(np.dot(co[:, mask].T - points[:, i], n))
                interior[mask] *= s

            return geo_object(co, *params)*interior

        self.geo_object = new_geo_object
        return new_geo_object


class PointCloud2D(GenericGeometry):

    def __init__(self, points: list | tuple | np.ndarray):
        """
        SDF of the point cloud.
        :param points: Positions of the points in an array of shape (2, N-points).
        """
        self._points = np.asarray(points)
        if self._points.shape[1] < self._points.shape[0]:
            self._points = self._points.T

        GenericGeometry.__init__(self,
                                 sdf_point_cloud_2d,
                                 self._points)

    @property
    def points(self):
        return self._points
