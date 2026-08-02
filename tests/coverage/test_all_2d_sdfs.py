import inspect
from time import process_time

import numpy as np

from spomso.cores.helper_functions import generate_grid
from spomso.cores import sdf_2D

from _test_helpers import plot_fields, evaluate_nt_sdfs

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (6, 6)
co_resolution = (200, 200)

show = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CANONICAL PARAMETERS

PARAMS = {
    "sdf_circle": (1.0,),
    "sdf_neu_circle": (1.0, 3),
    "sdf_box_2d": (np.asarray([1.5, 2.0]),),
    "sdf_segment_2d": (np.asarray([-1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0])),
    "sdf_rounded_box_2d": (
        np.asarray([1.5, 2.0]),
        np.asarray([0.1, 0.2, 0.1, 0.2]),
    ),
    "sdf_triangle_2d": (
        np.asarray([-1.0, -1.0, 0]),
        np.asarray([1.0, -1.0, 0]),
        np.asarray([0.0, 1.0, 0]),
    ),
    "sdf_arc": (1.0, 0.0, np.pi),
    "sdf_sector": (1.0, -np.pi / 4, np.pi / 4),
    "sdf_inf_sector": (-np.pi / 4, np.pi / 4),
    "sdf_ngon": (1.0, 5),
    "sdf_segmented_curve_2d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T,
        np.linspace(0, 2, 5, endpoint=False)
    ),
    "sdf_segmented_line_2d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_closed_segmented_line_2d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_polygon_2d": (
        np.asarray([[-1.2, 1.5, 0], [2.0, 1.5, 0.0], [1.0, -1.0, 0], [-2.0, -1.0, 0.0]]).T,
    ),
    "sdf_parametric_curve_2d": (
        lambda t, r, p, : np.asarray([r*np.cos(2*np.pi*p*t), r*np.sin(2*np.pi*p*t)]),
        (2, 0.5),
        np.linspace(0, 1, 11, endpoint=False)
    ),
    "sdf_point_cloud_2d": (
        np.asarray([[-1.2, 1.5, 0], [2.0, 1.5, 0.0], [1.0, -1.0, 0], [-2.0, -1.0, 0.0]]).T,
    )
}

# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

all_sdfs = { name: fn for name, fn in inspect.getmembers(sdf_2D, callable) if name.startswith("sdf_")}

missing = [name for name in all_sdfs if name not in PARAMS]
if missing:
    print(f"WARNING: no PARAMS entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results = evaluate_nt_sdfs(all_sdfs, PARAMS, coor, co_resolution)

end_time = process_time()
print(f"Evaluated {len(results)} 2D SDFs in {end_time - start_time:.2f} s")

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELDS

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show:
    plot_fields(results, co_size, coor, co_res_new, plane="XY")