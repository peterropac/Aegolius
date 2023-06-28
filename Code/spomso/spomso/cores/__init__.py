# Copyright (C) 2023 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

from .combine import CombineGeometry, smoothmax_boltzman, smoothmin_poly2, smoothmin_poly3
from .transformations import EuclideanTransform
from .modifications import ModifyObject
from .geom import GenericGeometry
from .helper_functions import resolution_converison, generate_grid, smarter_reshape, hard_binarization

from .sdf_2D import sdf_circle, sdf_segment_2d, sdf_box_2d, sdf_triangle_2d, sdf_arc, sdf_sector, sdf_inf_sector
from .sdf_2D import sdf_ngon, sdf_segmented_curve_2d, sdf_parametric_curve_2d

from .geom_2d import Circle, NGon, Rectangle, Segment, Triangle, Sector, InfiniteSector, Arc
from .geom_2d import ParametricCurve, SegmentedParametricCurve, SegmentedLine

from .sdf_3D import sdf_sphere, sdf_cylinder, sdf_box, sdf_torus, sdf_chainlink, sdf_braid, sdf_arc_3d
from .sdf_3D import sdf_plane, sudf_plane, sdf_cone, sdf_oriented_infinite_cone, sdf_infinite_cone, sdf_solid_angle
from .sdf_3D import sdf_triangle_3d, sdf_quad_3d

from .geom_3d import InfiniteCylinder, Cylinder, Sphere, Box, Plane, OrientedPlane, Line, Triangle3D, Quad
from .geom_3d import Torus, ChainLink, Braid, Arc3D, Cone, InfiniteCone, OrientedInfiniteCone
from .geom_3d import ParametricCurve3D, SegmentedParametricCurve3D, SegmentedLine3D
