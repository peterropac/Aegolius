# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from typing import Callable
from spomso.cores.transformations import EuclideanTransform
from spomso.cores.modifications import ModifyObject
from spomso.cores.geom import GenericGeometry
from spomso.cores.sdf_2D import sdf_circle
from spomso.cores.sdf_3D import sdf_sphere, sdf_cylinder, sdf_box, sdf_torus, sdf_arc_3d, sdf_chainlink, sdf_braid
from spomso.cores.sdf_3D import sudf_plane, sdf_plane, sdf_segment_3d, sdf_cone
from spomso.cores.sdf_3D import sdf_infinite_cone, sdf_oriented_infinite_cone
from spomso.cores.sdf_3D import sdf_solid_angle, sdf_triangle_3d, sdf_quad_3d
from spomso.cores.sdf_3D import sdf_parametric_curve_3d, sdf_segmented_curve_3d, sdf_segmented_line_3d
from spomso.cores.sdf_3D import sdf_x, sdf_y, sdf_z
from spomso.cores.sdf_3D import sdf_point_cloud_3d


class GenericGeometry3D(EuclideanTransform, ModifyObject):

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


class X(GenericGeometry):
    """
    Value of the X coordinate zeroed at some offset value.

    Args:
        offset: Value at which the field is zero.
    """

    def __init__(self, offset: float | int):
        GenericGeometry.__init__(self, sdf_x, offset)
        self._offset = offset

    @property
    def offset(self) -> float | int:
        """Value at which the field is zero."""
        return self._offset


class Y(GenericGeometry):
    """
    Value of the Y coordinate zeroed at some offset value.

    Args:
        offset: Value at which the field is zero.
    """

    def __init__(self, offset: float | int):
        GenericGeometry.__init__(self, sdf_y, offset)
        self._offset = offset

    @property
    def offset(self) -> float | int:
        """Value at which the field is zero."""
        return self._offset


class Z(GenericGeometry):
    """
    Value of the Z coordinate zeroed at some offset value.

    Args:
        offset: Value at which the field is zero.
    """

    def __init__(self, offset: float | int):
        GenericGeometry.__init__(self, sdf_z, offset)
        self._offset = offset

    @property
    def offset(self) -> float | int:
        """Value at which the field is zero."""
        return self._offset


class InfiniteCylinder(GenericGeometry):
    """
    Cylinder of a given radius and infinite height.

    Args:
        radius: Radius of the cylinder.
    """

    def __init__(self, radius: float | int):
        GenericGeometry.__init__(self, sdf_circle, radius)
        self._radius = radius

    @property
    def radius(self) -> float | int:
        """Radius of the cylinder."""
        return self._radius


class Cylinder(GenericGeometry):
    """
    Cylinder defined by the radius and height.

    Args:
        radius: Radius of the cylinder.
        height: Height of the cylinder.
    """

    def __init__(self, radius: float | int, height: float | int):
        GenericGeometry.__init__(self, sdf_cylinder, radius, height)
        self._radius = radius
        self._height = height

    @property
    def radius(self) -> float | int:
        """Radius of the cylinder."""
        return self._radius

    @property
    def height(self) -> float | int:
        """Radius of the cylinder."""
        return self._height


class Sphere(GenericGeometry):
    """
    Sphere defined by the radius.

    Args:
        radius: Radius of the sphere.
    """

    def __init__(self, radius: float):
        GenericGeometry.__init__(self, sdf_sphere, radius)
        self._radius = radius

    @property
    def radius(self) -> float | int:
        """Radius of the sphere."""
        return self._radius


class Box(GenericGeometry):
    """
    Box defined by the side lengths a, b, c.

    Args:
        a: Side length along the x-axis.
        b: Side length along the y-axis.
        c: Side length along the z-axis.
    """
    
    def __init__(self, a: float | int, b: float | int, c: float | int):
        GenericGeometry.__init__(self, sdf_box, a/2, b/2, c/2)
        self._a = a/2
        self._b = b/2
        self._c = c/2

    @property
    def a(self) -> float | int:
        """Side length along the x-axis."""
        return self._a

    @property
    def b(self) -> float | int:
        """Side length along the y-axis."""
        return self._b

    @property
    def c(self) -> float | int:
        """Side length along the z-axis."""
        return self._c


class Plane(GenericGeometry):
    """
    Plane defined by the normal vector.
    
    Args:
        normal: Normal vector of the plane.
        thickness: Thickness of the plane.
    """
    
    def __init__(self, normal: tuple | list | np.ndarray, thickness: float | int):
        GenericGeometry.__init__(self, sudf_plane, np.asarray(normal), thickness)
        self._normal = np.asarray(normal)
        self._thickness = thickness

    @property
    def normal(self) -> np.ndarray:
        """Normal vector of the plane."""
        return self._normal

    @property
    def thickness(self) -> float | int:
        """Thickness of the plane."""
        return self._thickness


class OrientedPlane(GenericGeometry):
    """
    Plane defined by a normal vector. SDF has a negative value for points below the plane.
    Args:
        normal: Normal vector of the plane.
        offset: Offset of the origin along the normal vector.
    """
    
    def __init__(self, normal: tuple | list | np.ndarray, offset: int | float):
        GenericGeometry.__init__(self, sdf_plane, np.asarray(normal), offset)
        self._normal = np.asarray(normal)
        self._offset = offset

    @property
    def normal(self) -> np.ndarray:
        """Normal vector of the plane."""
        return self._normal

    @property
    def offset(self) -> float | int:
        """Offset of the origin along the normal vector."""
        return self._offset


class Line(GenericGeometry):
    """
    Line defined by the starting and ending point.
    
    Args:
        a: Vector defining the starting point.
        b: Vector defining the ending point.
    """

    def __init__(self, a: np.ndarray | list | tuple, b: np.ndarray | list | tuple):
        GenericGeometry.__init__(self, sdf_segment_3d, a, b)
        self._a = np.asarray(a)
        self._b = np.asarray(b)

    @property
    def point_a(self) -> np.ndarray:
        """Vector defining the starting point."""
        return self._a

    @property
    def point_b(self) -> np.ndarray:
        """Vector defining the ending point."""
        return self._b
    
    
class Triangle3D(GenericGeometry):
    """
    Triangle defined by the three vertices.

    Args:
        a: Vector defining the position of the first vertex.
        b: Vector defining the position of the second vertex.
        c: Vector defining the position of the third vertex.
    """

    def __init__(self, a: tuple | list | np.ndarray, b: tuple | list | np.ndarray, c: tuple | list | np.ndarray):
        GenericGeometry.__init__(self, sdf_triangle_3d, np.asarray(a), np.asarray(b), np.asarray(c))
        self._a = np.asarray(a)
        self._b = np.asarray(b)
        self._c = np.asarray(c)

    @property
    def a(self) -> np.ndarray:
        """Vector defining the position of the first vertex."""
        return self._a

    @property
    def b(self) -> np.ndarray:
        """Vector defining the position of the second vertex."""
        return self._b

    @property
    def c(self) -> np.ndarray:
        """Vector defining the position of the third vertex."""
        return self._c


class Quad(GenericGeometry):
    """
    Quadrilateral defined by the four vertices.
    
    Args:
        a: Vector defining the position of the first vertex.
        b: Vector defining the position of the second vertex.
        c: Vector defining the position of the third vertex.
        d: Vector defining the position of the fourth vertex.
    """
    def __init__(self,
                a: tuple | list | np.ndarray, b: tuple | list | np.ndarray,
                c: tuple | list | np.ndarray, d: tuple | list | np.ndarray):
        GenericGeometry.__init__(self, sdf_quad_3d, np.asarray(a), np.asarray(b), np.asarray(c), np.asarray(d))
        self._a = np.asarray(a)
        self._b = np.asarray(b)
        self._c = np.asarray(c)
        self._d = np.asarray(d)
        
    @property
    def a(self) -> np.ndarray:
        """Vector defining the position of the first vertex."""
        return self._a

    @property
    def b(self) -> np.ndarray:
        """Vector defining the position of the second vertex."""
        return self._b

    @property
    def c(self) -> np.ndarray:
        """Vector defining the position of the third vertex."""
        return self._c

    @property
    def d(self) -> np.ndarray:
        """Vector defining the position of the fourth vertex."""
        return self._d


class Torus(GenericGeometry):
    """
    Torus defined by the primary and the secondary radius.

    Args:
        primary_radius: Primary radius of the torus.
        secondary_radius: Secondary radius of the torus.
    """

    def __init__(self, primary_radius: float, secondary_radius: float):
        GenericGeometry.__init__(self, sdf_torus, primary_radius, secondary_radius)
        self._pradius = primary_radius
        self._sradius = secondary_radius

    @property
    def primary_radius(self) -> float | int:
        """Primary radius of the torus - distance between the center of the tube and the center of the torus."""
        return self._pradius

    @property
    def secondary_radius(self) -> float | int:
        """Secondary radius of the torus - radius of the tube."""
        return self._sradius


class ChainLink(GenericGeometry):
    """
    Chain Link defined by the primary radius, secondary radius and the length.

    Args:
        primary_radius: Width of the chain link.
        secondary_radius: Thickness of the wire.
        length: Length of the chain link.
    """

    def __init__(self, primary_radius: float | int, secondary_radius: float | int, length: float | int):
        GenericGeometry.__init__(self, sdf_chainlink, primary_radius, secondary_radius, length/2)
        self._pradius = primary_radius
        self._sradius = secondary_radius
        self._length = length/2

    @property
    def primary_radius(self) -> float | int:
        """Width of the chain link."""
        return self._pradius

    @property
    def secondary_radius(self) -> float | int:
        """Thickness of the wire."""
        return self._sradius

    @property
    def length(self) -> float | int:
        """Length of the chain link."""
        return self._length


class Braid(GenericGeometry):
    """
    Braid defined as a twisted chain link, where the total number of revolutions is defined via the pitch.

    Args:
        length: Length of the braid.
        primary_radius: Width of the braid.
        secondary_radius: Thickness of the braid.
        pitch: Pitch of the twist in units rad/unit length.
    """

    def __init__(self, length: float | int,
                primary_radius: float | int, secondary_radius: float | int,
                pitch: float | int):
        GenericGeometry.__init__(self, sdf_braid, length/2, primary_radius, secondary_radius, pitch)
        self._pradius = primary_radius
        self._sradius = secondary_radius
        self._length = length/2
        self._pitch = pitch

    @property
    def primary_radius(self) -> float | int:
        """Width of the braid."""
        return self._pradius

    @property
    def secondary_radius(self) -> float | int:
        """Thickness of the braid."""
        return self._sradius

    @property
    def length(self) -> float | int:
        """Length of the braid."""
        return self._length

    @property
    def pitch(self) -> float | int:
        """Pitch of the twist in units rad/unit length."""
        return self._pitch


class Arc3D(GenericGeometry):
    """
    Arc defined by the radius, thickness, and the angles of both ends.
        
    Args:    
        radius: Radius of the arc.
        thickness: Thickness of the arc.
        start_angle: Angle of one end with respect to the x-axis.
        end_angle: Angle of the other end with respect to the x-axis.
    """

    def __init__(self,
                radius: float | int, thickness: float | int,
                start_angle: float | int, end_angle: float | int):
        GenericGeometry.__init__(self, sdf_arc_3d,
                                radius, thickness,
                                start_angle, end_angle)
        self._pradius = radius
        self._sradius = thickness
        self._sangle = start_angle
        self._eangle = end_angle

    @property
    def start_angle(self) -> float | int:
        """Angle of one end with respect to the x-axis."""
        return self._sangle

    @property
    def end_angle(self) -> float | int:
        """Angle of the other end with respect to the x-axis."""
        return self._eangle

    @property
    def primary_radius(self) -> float | int:
        """Radius of the arc."""
        return self._pradius

    @property
    def secondary_radius(self) -> float | int:
        """Thickness of the arc."""
        return self._sradius


class Cone(GenericGeometry):

    def __init__(self, height: float | int, angle: float | int):
        """
        Cone defined by its height and the angle of the slope.
        The base of the cone is moved down by: height - height_offset.

        Args:
            height: Height of the cone.
            angle: Angle of the slope.
        """
        GenericGeometry.__init__(self, sdf_cone, height, angle)
        self._height = height
        self._angle = angle

    @property
    def height(self) -> float | int:
        """Height of the cone."""
        return self._height

    @property
    def height_offset(self) -> float | int:
        """Center of mass height."""
        return self._height*(0.5**(1/3))

    @property
    def angle(self) -> float | int:
        """Angle of the slope."""
        return self._angle

    @property
    def base_radius(self) -> float | int:
        """Radius of the base."""
        return self._height*np.tan(self._angle)


class InfiniteCone(GenericGeometry):
    """
    Cone with infinite height defined by the angle of its slope. The tip of the cone is in the origin.

    Args:
        angle: Angle of the slope.
    """

    def __init__(self, angle: float | int):
        GenericGeometry.__init__(self, sdf_infinite_cone, angle)
        self._angle = angle

    @property
    def angle(self) -> float | int:
        """Angle of the slope."""
        return self._angle


class OrientedInfiniteCone(GenericGeometry):
    """
    Cone with infinite height defined by the angle of its slope. The tip of the cone is in the origin.
    Values of the SDF below the cone are negative.

    Args:
        angle: Angle of the slope.
    """
    def __init__(self, angle: float | int):
        GenericGeometry.__init__(self, sdf_oriented_infinite_cone, angle)
        self._angle = angle

    @property
    def angle(self) -> float | int:
        """Angle of the slope."""
        return self._angle


class SolidAngle(GenericGeometry):
    """
    Solid angle defined by the radius of the globe and two angles.

    Args:
        radius: Radius of the globe.
        angle_1: First angle defining the sector.
        angle_2: Second angle defining the sector.
    """

    def __init__(self, radius: float | int, angle_1: float | int, angle_2: float | int):
        GenericGeometry.__init__(self, sdf_solid_angle, radius, angle_1, angle_2)
        self._radius = radius
        self._angle_1 = angle_1
        self._angle_2 = angle_2

    @property
    def radius(self) -> float | int:
        """Radius of the globe."""
        return self._radius

    @property
    def angle_1(self) -> float | int:
        """First angle defining the sector."""
        return self._angle_1

    @property
    def angle_2(self) -> float | int:
        """Second angle defining the sector."""
        return self._angle_2


class ParametricCurve3D(GenericGeometry):
    """
    Curve connecting the points on the user provided parametric curve.

    Args:
        parametric_curve: Function defining the parametric curve with a single variable (t) and other parameters.
        parametric_curve_parameters: Parameters of the function defining the parametric curve.
        t_range: Range of the t parameter/variable.
            t_range[0] -  start, t_range[1] - end, t_range[2] -  number of steps in between.
        closed: Is the curve closed on itself. False by default.
    """

    def __init__(self,
                 parametric_curve: Callable[[np.ndarray, tuple], np.ndarray],
                 parametric_curve_parameters: tuple,
                 t_range: tuple,
                 closed: bool = False):

        self._curve = parametric_curve
        self._c_params = parametric_curve_parameters
        self._t_range = t_range
        self._closed = closed

        GenericGeometry.__init__(self, self.sdf_closed_curve(),
                                 parametric_curve, parametric_curve_parameters, self.ts)

    def sdf_closed_curve(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Returns the SDF of the closed curve if parameter close==True.

        Returns:
            Modified SDF
        """
        def new_geo_object(co, f, f_parameters, t):

            p0 = f(t[0], *f_parameters)
            p1 = f(t[-1], *f_parameters)

            f1 = sdf_parametric_curve_3d(co, f, f_parameters, t)
            f2 = sdf_segment_3d(co, p0, p1)

            return np.minimum(f1, f2)

        if self.closed:
            return new_geo_object
        else:
            return sdf_parametric_curve_3d

    @property
    def steps(self) -> float | int:
        """Number of points at which the curve is evaluated."""
        return self._t_range[2]

    @property
    def t_start(self) -> float | int:
        """Start value of parameter t."""
        return self._t_range[0]

    @property
    def t_end(self) -> float | int:
        """End value of parameter t."""
        return self._t_range[1]

    @property
    def ts(self) -> np.ndarray:
        """Values of parameter t."""
        return np.linspace(*self._t_range)

    @property
    def closed(self) -> bool:
        """Is the segmented line closed on itself. False by default."""
        return self._closed


class SegmentedParametricCurve3D(GenericGeometry):
    """
    Segmented line connecting the user provided points.
        points: Points to connect.
        t_range: Range of the t parameter.
            t_range[0] -  start, t_range[1] - end, t_range[2] -  number of steps in between.
            t_range[1]-t_range[0] >= number of points.
        closed: Is the segmented line closed on itself. False by default.
    """

    def __init__(self, points: list | tuple | np.ndarray, t_range: tuple, closed: bool = False):
        self._points = np.asarray(points)
        if self._points.shape[1] < self._points.shape[0]:
            self._points = self._points.T
        self._t_range = t_range
        self._closed = closed

        GenericGeometry.__init__(self, self.sdf_closed_curve(), self._points, self.ts)

    def sdf_closed_curve(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Returns the SDF of the closed curve if parameter close==True.

        Returns:
            Modified SDF
        """
        def new_geo_object(co, points, t):

           f1 = sdf_segmented_curve_3d(co, points, t)
           f2 = sdf_segment_3d(co, points[:, 0], points[:, -1])

           return np.minimum(f1, f2)

        if self.closed:
           return new_geo_object
        else:
           return sdf_segmented_curve_3d

    @property
    def steps(self) -> int:
        """Number of points at which the curve is evaluated."""
        return self._t_range[2]

    @property
    def t_start(self) -> float | int:
        """Start value of parameter t."""
        return self._t_range[0]

    @property
    def t_end(self) -> float | int:
        """End value of parameter t."""
        return self._t_range[1]

    @property
    def ts(self) -> np.ndarray:
        """Values of parameter t."""
        tt = np.linspace(self._t_range[0], self._t_range[1]-1, self._t_range[2]) - self._t_range[0]
        ts_ = np.clip(tt, 0, self._points.shape[1]-1.0001)
        return ts_

    @property
    def closed(self) -> bool:
        """Is the segmented line closed on itself. False by default."""
        return self._closed


class SegmentedLine3D(GenericGeometry):
    """
    Segmented line connecting the user provided points.

    Args:
        points: Points to connect.
        closed: Is the segmented line closed on itself. False by default.
    """

    def __init__(self, points: list | tuple | np.ndarray, closed: bool = False):
        self._points = np.asarray(points)
        if self._points.shape[1] < self._points.shape[0]:
            self._points = self._points.T
        self._closed = closed

        GenericGeometry.__init__(self, self.sdf_closed_curve(), self._points)

    def sdf_closed_curve(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Returns the SDF of the closed curve if parameter close==True.

        Returns:
            Modified SDF
        """
        def new_geo_object(co, points):

            f1 = sdf_segmented_line_3d(co, points)
            f2 = sdf_segment_3d(co, points[:, 0], points[:, -1])

            return np.minimum(f1, f2)

        if self.closed:
            return new_geo_object
        else:
            return sdf_segmented_curve_3d

    @property
    def closed(self) -> bool:
        """Is the segmented line closed on itself. False by default."""
        return self._closed


class PointCloud3D(GenericGeometry):
    """
    SDF of the point cloud.

    Args:
        points: Positions of the points in an array of shape (3, N-points).
    """

    def __init__(self, points: list | tuple | np.ndarray):
        self._points = np.asarray(points)
        if self._points.shape[1] < self._points.shape[0]:
            self._points = self._points.T

        GenericGeometry.__init__(self, sdf_point_cloud_3d, self._points)

    @property
    def points(self) -> np.ndarray:
        """Positions of the points in an array of shape (2, N-points)."""
        return self._points


