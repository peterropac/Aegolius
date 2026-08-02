import inspect
from time import process_time

import numpy as np

from spomso.cores.helper_functions import generate_grid
from spomso.cores import sdf_3D

from _test_helpers import plot_fields, evaluate_nt_sdfs

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (6, 6, 6)
co_resolution = (50, 50, 50)

show = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CANONICAL PARAMETERS

PARAMS = {
    "sdf_x": (0.5,),
    "sdf_y": (0.5,),
    "sdf_z": (0.5,),
    "sdf_sphere": (1.0,),
    "sdf_cylinder": (0.5, 2.0),
    "sdf_box": (np.asarray([1.5, 2.0, 1.0]),),
    "sdf_torus": (1.0, 0.3),
    "sdf_chainlink": (1, 0.3, 2),
    "sdf_braid": (1, 0.3, 2, 0.5),
    "sdf_arc_3d": (1.0, 0.2, 0.0, np.pi),
    "sdf_plane": (np.asarray([0.0, 0.0, 1.0]), 0.5),
    "sudf_plane": (np.asarray([0.0, 0.0, 1.0]), 0.5),
    "sdf_segment_3d": (
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ),
    "sdf_cone": (2.0, np.pi / 6),
    "sdf_oriented_infinite_cone": (np.pi / 6,),
    "sdf_infinite_cone": (np.pi / 6,),
    "sdf_solid_angle": (1.5, -np.pi / 4, np.pi / 4),
    "sdf_triangle_3d": (
        np.asarray([-1.0, -1.0, 0.0]),
        np.asarray([1.0, -1.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    ),
    "sdf_quad_3d": (
        np.asarray([-1.0, -1.0, 0.0]),
        np.asarray([1.0, -1.0, 0.0]),
        np.asarray([1.0, 1.0, 0.0]),
        np.asarray([-1.0, 1.0, 0.0]),
    ),
    "sdf_segmented_curve_3d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T,
        np.linspace(0, 2, 5, endpoint=False)
    ),
    "sdf_segmented_line_3d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_closed_segmented_line_3d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_parametric_curve_3d": (
        lambda t, r, p, h,:
        np.asarray([r * np.cos(2 * np.pi * p * t), r * np.sin(2 * np.pi * p * t), h * (t - 0.5)]),
        (2, 0.5, 1),
        np.linspace(0, 1, 11, endpoint=False)
    ),
    "sdf_point_cloud_3d": (
        np.asarray([[-1.2, 1.5, 0], [2.0, 1.5, 0.0], [1.0, -1.0, 0], [-2.0, -1.0, 0.0]]).T,
    )
}

# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

all_sdfs = {
    name: fn
    for name, fn in inspect.getmembers(sdf_3D, callable)
    if name.startswith("sdf_") or name.startswith("sudf_")
}

missing = [name for name in all_sdfs if name not in PARAMS]
if missing:
    print(f"WARNING: no PARAMS entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results = evaluate_nt_sdfs(all_sdfs, PARAMS, coor, co_resolution)

end_time = process_time()
print(f"Evaluated {len(results)} 3D SDFs in {end_time - start_time:.2f} s")

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELDS

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

# XY
if show:
    plot_fields(results, co_size, coor, co_res_new, plane="XY")
# XZ
if show:
    plot_fields(results, co_size, coor, co_res_new, plane="XZ")
# YZ
if show:
    plot_fields(results, co_size, coor, co_res_new, plane="YZ")