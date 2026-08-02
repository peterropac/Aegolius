import inspect
from time import process_time

import numpy as np

from spomso.cores.helper_functions import generate_grid
from spomso.cores.geom_3d import Box
from spomso.cores.modifications import ModifyObject
from spomso.cores.helper_functions import smarter_reshape
from scipy.spatial.transform import Rotation

from _test_helpers import plot_fields, error_report, evaluate_class_modifications

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

def make_base():
    return Box(1.0, 1.0, 1.0)

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS FOR EACH MODIFICATION

def _displace_fn(coordinates, amplitude):
    return amplitude * np.cos(np.pi * 2 * coordinates[0])

def _custom_mod(function_, co, sdf_params, mod_params):
    return function_(co, *sdf_params) - mod_params[0]

def _custom_pp(sdf_values, threshold):
    return sdf_values - threshold

def _curve(t, *p):
    return np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), (t - 0.5)])

PARAMS = {
    "elongation":                 (np.asarray([2.0, 0.0, 0.0]),),
    "rounding":                   (0.1,),
    "rounding_cs":                (0.1, 2.0),
    "boundary":                   (),
    "invert":                     (),
    "sign":                       (),
    "define_volume":              (lambda co, r: np.sign(np.linalg.norm(co, axis=0) - r), (1.5,)),
    "onion":                      (0.1,),
    "concentric":                 (1.0,),
    "revolution":                 (1.0,),
    "axis_revolution":            (1.0, np.pi / 4),
    "extrusion":                  (0.5,),
    "twist":                      (0.5,),
    "bend":                       (3.0, np.pi / 3),
    "shear_xz":                   (np.pi / 6,),
    "shear_yz":                   (np.pi / 6,),
    "shear_xy":                   (np.pi / 6,),
    "shear_zy":                   (np.pi / 6,),
    "shear_yx":                   (np.pi / 6,),
    "shear_zx":                   (np.pi / 6,),
    "displacement":               (_displace_fn, (0.1,)),
    "infinite_repetition":        (np.asarray([2.0, 2.0, 2.0]),),
    "finite_repetition":          (np.asarray([3.0, 3.0, 3.0]), np.asarray([2, 2, 2])),
    "finite_repetition_rescaled": (
        np.asarray([3.0, 3.0, 3.0]),
        np.asarray([2, 1, 1]),
        np.asarray([2, 1, 1]),
        np.asarray([0.0, 0.0, 0.0]),
    ),
    "symmetry":                   (0,),
    "mirror":                     (np.asarray([1.0, 1.0, 0.0]), np.asarray([-1.0, -1.0, 0.0])),
    "rotational_symmetry":        (6, 1.2, 0.0),
    "linear_instancing":          (5, np.asarray([-1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0])),
    "custom_modification":        (_custom_mod, (0.2,)),
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
    "conv_averaging":             ((3, 3, 3), 1, co_resolution),
    "conv_edge_detection":        (co_resolution,),
    "custom_post_process":        (_custom_pp, (0.1,)),
    "move_sdf":                       (np.asarray([0.5, 0.0, 0.0]),),
    "rotate_sdf":                     (Rotation.from_euler("z", np.pi / 4).as_matrix(),),
    "scale_sdf":                      (2.0,),
    "shear":                          (np.pi / 6, 0, 1),
    "recover_volume":                 (lambda co, *p: np.sign(np.linalg.norm(co, axis=0) - 0.5),),
    "signed":                         (co_resolution,),
    "curve_instancing":               (_curve, (), (0.0, 1.0, 8)),
    "aligned_curve_instancing":       (_curve, (), (0.0, 1.0, 8)),
    "fully_aligned_curve_instancing": (_curve, (), (0.0, 1.0, 8)),
}

# ----------------------------------------------------------------------------------------------------------------------
# DISCOVERY

_META_PROPS = {"modifications", "modified_object", "original_object"}
all_mods = {
    name for name in dir(ModifyObject)
    if not name.startswith("_")
    and name not in _META_PROPS
    and callable(getattr(ModifyObject, name))
}

missing = sorted(all_mods - set(PARAMS))
if missing:
    print(f"NOTE: ModifyObject methods with no PARAMS entry: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results, failed = evaluate_class_modifications(all_mods, PARAMS, coor, make_base)

end_time = process_time()

# ----------------------------------------------------------------------------------------------------------------------
# ERROR REPORT

error_report("modifications", results, failed, end_time - start_time)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELD FOR SNAPSHOT TESTING

_flat_results = {
    name: np.asarray(v).reshape(-1)
    for name, v in results.items()
}

field = np.stack(list(_flat_results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

# For plotting, reshape each flat SDF back to the 3D grid.
results_reshaped = {name: np.asarray(smarter_reshape(v, co_resolution)) for name, v in _flat_results.items()}

if show:
    plot_fields(results_reshaped, co_size, coor, co_res_new, plane="XY")
    plot_fields(results_reshaped, co_size, coor, co_res_new, plane="XZ")
    plot_fields(results_reshaped, co_size, coor, co_res_new, plane="YZ")