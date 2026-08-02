import inspect
from time import process_time

import numpy as np
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.jax_cores import combine_jax
from spomso.jax_cores.sdf_3D_jax import sdf_sphere, sdf_box
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf

from _test_helpers import plot_fields, error_report, evaluate_combine, evaluate_meta_combine

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (4, 4, 4)
co_resolution = (50, 50, 50)

show = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# BASE SDFs

_ident_rot = Rotation.from_euler("x", 0).as_matrix()

_sphere_fn = compound_euclidean_transform_sdf(
    sdf_sphere, _ident_rot, jnp.asarray([-0.5, 0.0, 0.0]), 1.0
)
_sphere_params = (1.0,)

_box_fn = compound_euclidean_transform_sdf(
    sdf_box, _ident_rot, jnp.asarray([0.5, 0.0, 0.0]), 1.0
)
_box_params = (jnp.asarray([1.0, 1.0, 1.0]),)

_box_2_fn = compound_euclidean_transform_sdf(
    sdf_box,
    Rotation.from_euler("z", 45, degrees=True).as_matrix(),
    jnp.asarray([0.0, 0.0, 0.0]),
    1.0
)
_box_2_params = (jnp.asarray([1.0, 3.0, 1.0]),)

_sphere_vals = _sphere_fn(coor, *_sphere_params)
_box_vals = _box_fn(coor, *_box_params)
_box_2_vals = _box_2_fn(coor, *_box_2_params)

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETER REGISTRIES BY SHAPE

ELEMENTWISE = {
    "union2":         (),
    "subtract2":      (),
    "intersect2":     (),
    "add":            (),
    "difference":     (),
    "smoothmin_poly2": (0.5,),
    "smoothmin_poly3": (0.5,),
    "smoothmax_boltz": (10.0,),
}

LIST_COMBINERS = {
    "union":     (),
    "intersect": (),
}

SMOOTH_PAIR = {
    "smooth_union2_2o":     (0.5,),
    "smooth_union2_3o":     (0.5,),
    "smooth_intersect2_2o": (0.5,),
    "smooth_intersect2_3o": (0.5,),
    "smooth_subtract2_2o":  (0.5,),
    "smooth_subtract2_3o":  (0.5,),
}

HIGHER_ORDER = {"combine_2_sdfs", "combine_multiple_sdfs", "parametric_combine_2_sdfs"}


# ----------------------------------------------------------------------------------------------------------------------
# DISCOVERY

all_combines = {
    name: fn for name, fn in inspect.getmembers(combine_jax, callable) if fn.__module__ == combine_jax.__name__
}

known = set(ELEMENTWISE) | set(LIST_COMBINERS) | set(SMOOTH_PAIR) | HIGHER_ORDER
missing = [name for name in all_combines if name not in known]
if missing:
    print(f"WARNING: no entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

# element-wise
ew_r, ew_f = evaluate_combine(ELEMENTWISE, all_combines, (_sphere_vals, _box_vals), co_resolution)

# list combiners
lc_r, lc_f = evaluate_combine(LIST_COMBINERS, all_combines,
                                        (jnp.asarray([_sphere_vals, _box_vals, _box_2_vals]), ),
                                        co_resolution)

# smooth pair
sp_r, sp_f = evaluate_combine(SMOOTH_PAIR, all_combines, (_sphere_vals, _box_vals), co_resolution)

# combine_2_sdfs
f_combined = all_combines["combine_2_sdfs"](
    _sphere_fn, _box_fn, _sphere_params, _box_params, combine_jax.union2
)
c2_r, c2_f = evaluate_meta_combine("combine_2_sdfs", f_combined, coor, co_resolution)

# combine_multiple_sdfs
f_combined = all_combines["combine_multiple_sdfs"](
    [_sphere_fn, _box_fn, _box_2_fn],
    [_sphere_params, _box_params, _box_2_params],
    combine_jax.union,
)
cm_r, cm_f = evaluate_meta_combine("combine_multiple_sdfs", f_combined, coor, co_resolution)

# parametric_combine_2_sdfs
f_combined = all_combines["parametric_combine_2_sdfs"](
    _sphere_fn, _box_fn, _sphere_params, _box_params,
    combine_jax.smooth_union2_2o, 0.2,
)
pc2_r, pc2_f = evaluate_meta_combine("parametric_combine_2_sdfs", f_combined, coor, co_resolution)

results = ew_r | lc_r | sp_r | c2_r | cm_r | pc2_r
failed = ew_f | lc_f | sp_f | c2_f | cm_f | pc2_f

end_time = process_time()

# ----------------------------------------------------------------------------------------------------------------------
# ERROR REPORT

error_report("combines", results, failed, end_time - start_time)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELD FOR SNAPSHOT TESTING

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show:
    plot_fields(results, co_size, coor, co_res_new, plane="XY")