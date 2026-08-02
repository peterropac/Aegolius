import inspect
from time import process_time

import numpy as np
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from spomso.cores.helper_functions import generate_grid
from spomso.jax_cores import modifications_jax
from spomso.jax_cores.sdf_3D_jax import sdf_box
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf

from _test_helpers import plot_fields, evaluate_modifications, error_report

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

BASE_SDF_PARAMS = (jnp.asarray([1.0, 1.0, 1.0]),)
_identity_rot = Rotation.from_euler("x", 0).as_matrix()
_zero_vec = jnp.zeros(3)
BASE_SDF = compound_euclidean_transform_sdf(sdf_box, _identity_rot, _zero_vec, 1.0)

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS FOR EACH MODIFICATION

def _displace_fn(coordinates, amplitude):
    return amplitude * jnp.cos(jnp.pi * 2 * coordinates[0])

def _custom_mod(function_, co, sdf_params, mod_params):
    return function_(co, *sdf_params) - mod_params[0]

def _custom_pp(sdf_values, threshold):
    return sdf_values - threshold


PARAMS = {
    "elongation":                 (jnp.asarray([2.0, 0.0, 0.0]),),
    "rounding":                   (0.1,),
    "rounding_cs":                (0.1, 2.0),
    "boundary":                   (),
    "invert":                     (),
    "sign":                       (),
    "define_volume":              (lambda co, r: jnp.sign(jnp.linalg.norm(co, axis=0) - r), (1.5,)),
    "onion":                      (0.1,),
    "concentric":                 (1.0,),
    "revolution":                 (1.0,),
    "axis_revolution":            (1.0, jnp.pi / 4),
    "extrusion":                  (0.5,),
    "twist":                      (0.5,),
    "bend":                       (3.0, jnp.pi / 3),
    "shear_xz":                   (jnp.pi / 6,),
    "shear_yz":                   (jnp.pi / 6,),
    "shear_xy":                   (jnp.pi / 6,),
    "shear_zy":                   (jnp.pi / 6,),
    "shear_yx":                   (jnp.pi / 6,),
    "shear_zx":                   (jnp.pi / 6,),
    "displacement":               (_displace_fn, (0.1,)),
    "infinite_repetition":        (jnp.asarray([2.0, 2.0, 2.0]),),
    "finite_repetition":          (jnp.asarray([3.0, 3.0, 3.0]), jnp.asarray([2, 2, 2])),
    "finite_repetition_rescaled": (
        jnp.asarray([3.0, 3.0, 3.0]),
        jnp.asarray([2, 1, 1]),
        jnp.asarray([2, 1, 1]),
        jnp.asarray([0.0, 0.0, 0.0]),
    ),
    "symmetry":                   (0,),
    "mirror":                     (jnp.asarray([1.0, 1.0, 0.0]), jnp.asarray([-1.0, -1.0, 0.0])),
    "rotational_symmetry":        (6, 1.2, 0.0),
    "linear_instancing":          (5, jnp.asarray([-1.0, 0.0, 0.0]), jnp.asarray([1.0, 0.0, 0.0])),
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
}

# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

all_mods = {
    name: fn
    for name, fn in inspect.getmembers(modifications_jax, callable) if fn.__module__ == modifications_jax.__name__
}

missing = [name for name in all_mods if name not in PARAMS]
if missing:
    print(f"WARNING: no PARAMS entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results, failed = evaluate_modifications(all_mods, PARAMS, coor, co_resolution, BASE_SDF, BASE_SDF_PARAMS)

end_time = process_time()

# ----------------------------------------------------------------------------------------------------------------------
# ERROR REPORT

error_report("modifications", results, failed, end_time - start_time)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELD FOR SNAPSHOT TESTING

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show:
    plot_fields(results, co_size, coor, co_res_new, plane="XY")