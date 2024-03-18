# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

from .combine import CombineGeometry, smoothmax_boltz, smoothmin_poly2, smoothmin_poly3
from .transformations import EuclideanTransform, EuclideanTransformPoints
from .modifications import ModifyObject, ModifyVectorObject
from .geom import GenericGeometry, Points
from .helper_functions import resolution_conversion, generate_grid
from .helper_functions import smarter_reshape, vector_smarter_reshape, nd_vector_smarter_reshape
from .post_processing import sigmoid_falloff, positive_sigmoid_falloff, capped_exponential
from .post_processing import linear_falloff, relu, smooth_relu, slowstart
from .post_processing import hard_binarization
from .post_processing import gaussian_boundary, gaussian_falloff
from .post_processing import conv_averaging, conv_edge_detection
from .post_processing import custom_post_process
from .post_processing import PostProcess

from .sdf_2D import sdf_circle, sdf_segment_2d, sdf_box_2d, sdf_rounded_box_2d, sdf_triangle_2d
from .sdf_2D import sdf_arc, sdf_sector, sdf_inf_sector, sdf_ngon
from .sdf_2D import sdf_segmented_curve_2d, sdf_segmented_line_2d, sdf_parametric_curve_2d
from .sdf_2D import sdf_point_cloud_2d

from .geom_2d import Circle, NGon, Rectangle, RoundedRectangle, Segment, Triangle, Sector, InfiniteSector, Arc
from .geom_2d import ParametricCurve, SegmentedParametricCurve, SegmentedLine
from .geom_2d import PointCloud2D

from .sdf_3D import sdf_sphere, sdf_cylinder, sdf_box, sdf_torus, sdf_chainlink, sdf_braid, sdf_arc_3d
from .sdf_3D import sdf_plane, sudf_plane, sdf_segment_3d
from .sdf_3D import sdf_cone, sdf_oriented_infinite_cone, sdf_infinite_cone, sdf_solid_angle
from .sdf_3D import sdf_triangle_3d, sdf_quad_3d
from .sdf_3D import sdf_segmented_line_3d, sdf_segmented_curve_3d, sdf_parametric_curve_3d
from .sdf_3D import sdf_x, sdf_y, sdf_z
from .sdf_3D import sdf_point_cloud_3d

from .geom_3d import InfiniteCylinder, Cylinder, Sphere, Box, Plane, OrientedPlane, Line, Triangle3D, Quad
from .geom_3d import Torus, ChainLink, Braid, Arc3D, Cone, InfiniteCone, OrientedInfiniteCone
from .geom_3d import ParametricCurve3D, SegmentedParametricCurve3D, SegmentedLine3D
from .geom_3d import X, Y, Z

from .vector_functions import cartesian_define, spherical_define, cylindrical_define
from .vector_functions import radial_vector_field_spherical
from .vector_functions import radial_vector_field_cylindrical, hyperbolic_vector_field_cylindrical
from .vector_functions import aar_vector_field_cylindrical, awn_vector_field_cylindrical
from .vector_functions import vortex_vector_field_cylindrical, aav_vector_field_cylindrical
from .vector_functions import x_vector_field, y_vector_field, z_vector_field
from .vector_functions import from_sdf
from .geom_vector_special import lcwg1_p1, lcwg1_m1, lcwg1_2d
from .vector_functions_special import compute_crossings_2d

from .vector_modification_functions import batch_normalize
from .vector_modification_functions import add_vectors, subtract_vectors, rescale_vectors
from .vector_modification_functions import rotate_vectors_phi, rotate_vectors_theta
from .vector_modification_functions import rotate_vectors_x_axis, rotate_vectors_y_axis, rotate_vectors_z_axis
from .vector_modification_functions import rotate_vectors_axis
from .vector_modification_functions import revolve_field_x, revolve_field_y, revolve_field_z

from .geom_vector import CartesianVectorField, CylindricalVectorField, SphericalVectorField
from .geom_vector import RadialSphericalVectorField
from .geom_vector import RadialCylindricalVectorField, HyperbolicCylindricalVectorField
from .geom_vector import AngledRadialCylindricalVectorField, WindingCylindricalVectorField
from .geom_vector import VortexCylindricalVectorField, AngledVortexCylindricalVectorField
from .geom_vector import XVectorField, YVectorField, ZVectorField
from .geom_vector import VectorFieldFromSDF
from .geom_vector_special import LCWG2D, LCWG3Dp1, LCWG3Dm1
