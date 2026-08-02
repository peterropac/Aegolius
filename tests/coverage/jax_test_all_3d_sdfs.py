import inspect
from time import process_time

import numpy as np
import jax.numpy as jnp

from spomso.cores.helper_functions import generate_grid
from spomso.jax_cores import sdf_3D_jax

from _test_helpers import plot_fields, evaluate_sdfs

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (4, 4, 4)
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
    "sdf_box": (jnp.asarray([1.5, 2.0, 1.0]),),
    "sdf_torus": (1.0, 0.3),
    "sdf_arc_3d": (1.0, 0.2, 0.0, jnp.pi),
    "sdf_plane": (jnp.asarray([0.0, 0.0, 1.0]), 0.5),
    "sudf_plane": (jnp.asarray([0.0, 0.0, 1.0]), 0.5),
    "sdf_segment_3d": (
        jnp.asarray([-1.0, 0.0, 0.0]),
        jnp.asarray([1.0, 0.0, 0.0]),
    ),
    "sdf_cone": (2.0, jnp.pi / 6),
    "sdf_oriented_infinite_cone": (jnp.pi / 6,),
    "sdf_infinite_cone": (jnp.pi / 6,),
    "sdf_solid_angle": (1.5, -jnp.pi / 4, jnp.pi / 4),
    "sdf_triangle_3d": (
        jnp.asarray([-1.0, -1.0, 0.0]),
        jnp.asarray([1.0, -1.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
    ),
    "sdf_quad_3d": (
        jnp.asarray([-1.0, -1.0, 0.0]),
        jnp.asarray([1.0, -1.0, 0.0]),
        jnp.asarray([1.0, 1.0, 0.0]),
        jnp.asarray([-1.0, 1.0, 0.0]),
    ),
    "sdf_segmented_line_3d": (
        jnp.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_closed_segmented_line_3d": (
        jnp.asarray([[1.5, 1.0, 0.8], [0.25, -1.5, 0.0], [-1.0, -1.0, -0.5], [-1.0, 1.0, -1.0]]).T,
    ),
}

# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

all_sdfs = {
    name: fn
    for name, fn in inspect.getmembers(sdf_3D_jax, callable)
    if name.startswith("sdf_") or name.startswith("sudf_")
}

missing = [name for name in all_sdfs if name not in PARAMS]
if missing:
    print(f"WARNING: no PARAMS entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results = evaluate_sdfs(all_sdfs, PARAMS, coor, co_resolution)

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