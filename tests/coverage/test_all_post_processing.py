import inspect
from time import process_time

import numpy as np

from spomso.cores.helper_functions import generate_grid
from spomso.cores import post_processing
from spomso.cores.sdf_3D import sdf_box

from _test_helpers import plot_fields, evaluate_post_processing, error_report

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (4, 4, 4)
co_resolution = (50, 50, 50)

show = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# BASE SDF

BASE_SDF = sdf_box
BASE_SDF_PARAMS = (np.asarray([1.0, 1.0, 1.0]),)


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS FOR EACH POST-PROCESSING OPERATION

def _custom_pp(sdf_values, threshold):
    return sdf_values - threshold

PARAMS = {
    "sigmoid_falloff":            (1.0, 0.5),
    "positive_sigmoid_falloff":   (1.0, 0.5),
    "capped_exponential":         (1.0, 0.5),
    "hard_binarization":          (0.0,),
    "linear_falloff":             (1.0, 0.5),
    "relu":                       (0.5,),
    "smooth_relu":                (0.1, 0.5, 0.0),
    "slowstart":                  (0.1, 0.5, 0.0, 0.0),
    "gaussian_boundary":          (1.0, 0.5),
    "gaussian_falloff":           (1.0, 0.5),
    "conv_averaging":             ((3, 3, 3), 1),
    "conv_edge_detection":        (),
    "custom_post_process":        (_custom_pp, (0.1,)),
}

# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

all_mods = {
    name: fn
    for name, fn in inspect.getmembers(post_processing, inspect.isfunction) if fn.__module__ == post_processing.__name__
}

missing = [name for name in all_mods if name not in PARAMS]
if missing:
    print(f"WARNING: no PARAMS entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results, failed = evaluate_post_processing(all_mods, PARAMS, coor, co_resolution, BASE_SDF, BASE_SDF_PARAMS)

end_time = process_time()

# ----------------------------------------------------------------------------------------------------------------------
# ERROR REPORT

error_report("post-processing functions", results, failed, end_time - start_time)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELD FOR SNAPSHOT TESTING

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show:
    plot_fields(results, co_size, coor, co_res_new, plane="XY")