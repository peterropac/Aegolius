import inspect
from time import process_time

import numpy as np
import jax.numpy as jnp

from spomso.cores.helper_functions import generate_grid
from spomso.jax_cores import sdf_2D_jax

from _test_helpers import plot_fields, evaluate_sdfs

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (4, 4)
co_resolution = (200, 200)

show = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CANONICAL PARAMETERS

PARAMS = {
    "sdf_circle": (1.0,),
    "sdf_box_2d": (jnp.asarray([1.5, 2.0]),),
    "sdf_segment_2d": (jnp.asarray([-1.0, 0.0, 0.0]), jnp.asarray([1.0, 0.0, 0.0])),
    "sdf_rounded_box_2d": (
        jnp.asarray([1.5, 2.0]),
        jnp.asarray([0.1, 0.2, 0.1, 0.2]),
    ),
    "sdf_triangle_2d": (
        jnp.asarray([-1.0, -1.0, 0]),
        jnp.asarray([1.0, -1.0, 0]),
        jnp.asarray([0.0, 1.0, 0]),
    ),
    "sdf_arc": (1.0, 0.0, jnp.pi),
    "sdf_sector": (1.0, -jnp.pi / 4, jnp.pi / 4),
    "sdf_inf_sector": (-jnp.pi / 4, jnp.pi / 4),
    "sdf_ngon": (1.0, 5),
    "sdf_segmented_line_2d": (
        jnp.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_closed_segmented_line_2d": (
        jnp.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_polygon_2d": (
        jnp.asarray([[-1.2, 1.5, 0], [2.0, 1.5, 0.0], [1.0, -1.0, 0], [-2.0, -1.0, 0.0]]).T,
    ),
}

# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

all_sdfs = { name: fn for name, fn in inspect.getmembers(sdf_2D_jax, callable) if name.startswith("sdf_")}

missing = [name for name in all_sdfs if name not in PARAMS]
if missing:
    print(f"WARNING: no PARAMS entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results = evaluate_sdfs(all_sdfs, PARAMS, coor, co_resolution)

end_time = process_time()
print(f"Evaluated {len(results)} 2D SDFs in {end_time - start_time:.2f} s")

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELDS

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show:
    plot_fields(results, co_size, coor, co_res_new, plane="XY")