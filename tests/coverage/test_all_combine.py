from time import process_time

import numpy as np

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.geom_3d import Box, Sphere
from spomso.cores.combine import CombineGeometry

from _test_helpers import plot_fields, error_report

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (4, 4, 4)
co_resolution = (50, 50, 50)

show = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# BASE GEOMETRY

def make_bases():
    s = Sphere(1.0)
    s.move((-0.5, 0.0, 0.0))
    b = Box(1.0, 1.0, 1.0)
    b.move((0.5, 0.0, 0.0))
    b2 = Box(1.0, 3.0, 1.0)
    b2.rotate(np.pi / 4, (0, 0, 1))
    return s, b, b2

# ----------------------------------------------------------------------------------------------------------------------
# NON-PARAMETRIC OPERATIONS (accessible via CombineGeometry.combine(*objs))

NONPARAM_OPS = [
    "UNION2",
    "UNION",
    "SUBTRACT2",
    "INTERSECT2",
    "INTERSECT",
    "SUM",
    "DIFFERENCE",
]

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETRIC OPERATIONS (accessible via CombineGeometry.combine_parametric(width, *objs))

PARAM_OPS_WIDTH = {
    "SMOOTH_UNION2_2":            0.5,
    "SMOOTH_UNION2":              0.5,
    "SMOOTH_INTERSECT2":          0.5,
    "SMOOTH_INTERSECT2_BOLTZMANN": 10.0,   # boltzmann uses `a` not width
    "SMOOTH_SUBTRACT2":           0.5,
    "SMOOTH_SUBTRACT2_BOLTZMANN": 10.0,
}

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results = {}
failed = {}

# Non-parametric operations (some are pairs, some take a list)
for op in NONPARAM_OPS:
    try:
        combiner = CombineGeometry(op)
        s, b, b2 = make_bases()
        if op in ("UNION", "INTERSECT"):
            combined = combiner.combine(s, b, b2)
        else:
            combined = combiner.combine(s, b)
        results[op] = np.asarray(combined.create(coor))
    except Exception as e:
        failed[op] = f"{type(e).__name__}: {e}"

# Parametric operations (all take a width/parameter plus a pair)
for op, width in PARAM_OPS_WIDTH.items():
    try:
        combiner = CombineGeometry(op)
        s, b, _ = make_bases()
        combined = combiner.combine_parametric(s, b, parameters=width)
        results[op] = np.asarray(combined.create(coor))
    except Exception as e:
        failed[op] = f"{type(e).__name__}: {e}"

end_time = process_time()

# ----------------------------------------------------------------------------------------------------------------------
# ERROR REPORT

error_report("combines", results, failed, end_time - start_time)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELD FOR SNAPSHOT TESTING

_flat_results = {name: np.asarray(v).reshape(-1) for name, v in results.items()}
field = np.stack(list(_flat_results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

results_reshaped = {name: np.asarray(smarter_reshape(v, co_resolution)) for name, v in _flat_results.items()}

if show:
    plot_fields(results_reshaped, co_size, coor, co_res_new, plane="XY")