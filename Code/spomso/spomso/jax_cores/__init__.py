# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.


from .combine_jax import combine_2_sdfs, combine_multiple_sdfs, parametric_combine_2_sdfs
from .combine_jax import union, union2, subtract2, intersect, intersect2
from .combine_jax import smooth_union2_2o, smooth_union2_3o
from .combine_jax import smooth_intersect2_2o, smooth_intersect2_3o
from .combine_jax import smooth_subtract2_2o, smooth_subtract2_3o
from .combine_jax import smoothmin_poly2, smoothmin_poly3, smoothmax_boltz

from .helper_functions import resolution_conversion, smarter_reshape, vector_smarter_reshape, nd_vector_smarter_reshape

from .post_processing_jax import sigmoid_falloff_jax, positive_sigmoid_falloff_jax, capped_exponential_jax
from .post_processing_jax import linear_falloff_jax, relu_jax, smooth_relu_jax, slowstart_jax
from .post_processing_jax import hard_binarization_jax
from .post_processing_jax import gaussian_boundary_jax, gaussian_falloff_jax
from .post_processing_jax import conv_multiple_jax, conv_edge_detection_jax

from .modifications_jax import boundary, invert, sign, define_volume, onion, concentric, rounding, rounding_cs
from .modifications_jax import elongation, revolution, axis_revolution, extrusion, twist, bend, displacement
from .modifications_jax import infinite_repetition, finite_repetition, finite_repetition_rescaled, linear_instancing
from .modifications_jax import symmetry, mirror, rotational_symmetry
from .modifications_jax import shear_xz, shear_yz, shear_xy, shear_zy, shear_zx, shear_zy
from .modifications_jax import custom_modification, custom_post_process

from .modifications_jax import sigmoid_falloff, positive_sigmoid_falloff, capped_exponential
from .modifications_jax import linear_falloff, relu, smooth_relu, slowstart
from .modifications_jax import hard_binarization
from .modifications_jax import gaussian_boundary, gaussian_falloff
from .modifications_jax import conv_averaging, conv_edge_detection

from .sdf_2D_jax import sdf_circle, sdf_segment_2d, sdf_box_2d, sdf_rounded_box_2d, sdf_triangle_2d
from .sdf_2D_jax import sdf_arc, sdf_sector, sdf_inf_sector, sdf_ngon
from .sdf_2D_jax import sdf_segmented_line_2d, sdf_closed_segmented_line_2d, sdf_polygon_2d

from .sdf_3D_jax import sdf_sphere, sdf_cylinder, sdf_box, sdf_segment_3d, sdf_torus, sdf_arc_3d
from .sdf_3D_jax import sdf_plane, sudf_plane
from .sdf_3D_jax import sdf_cone, sdf_oriented_infinite_cone, sdf_infinite_cone, sdf_solid_angle
from .sdf_3D_jax import sdf_triangle_3d, sdf_quad_3d
from .sdf_3D_jax import sdf_segmented_line_3d, sdf_closed_segmented_line_3d
from .sdf_3D_jax import sdf_x, sdf_y, sdf_z
