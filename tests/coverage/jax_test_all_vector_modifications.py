import inspect
from time import process_time

import numpy as np
import jax.numpy as jnp

from spomso.cores.helper_functions import generate_grid
from spomso.jax_cores import vector_modifications_jax as vmj
from spomso.jax_cores.vector_functions_jax import radial_vector_field_spherical

from _test_helpers import plot_vector_fields, batch_calculate_vector_modifications, error_report

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (4, 4, 4)
co_resolution = (30, 30, 30)

show_grid = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM + BASE FIELD

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# BASE FIELD

BASE_VEC = radial_vector_field_spherical(coor)

# ----------------------------------------------------------------------------------------------------------------------
# CANONICAL PARAMETERS
# For each function, the tuple of arguments to pass *after* the base vector field.

# Modifications that take (vec, *extras) — the common case.
PARAMS = {
    "batch_normalize":         (),
    "add_vectors":             (jnp.array([1.0, 0.0, 0.0]),),
    "subtract_vectors":        (jnp.array([1.0, 0.0, 0.0]),),
    "rescale_vectors":         (2.0,),
    "rotate_vectors_phi":      (jnp.pi / 4,),
    "rotate_vectors_theta":    (jnp.pi / 4,),
    "rotate_vectors_x_axis":   (jnp.pi / 4,),
    "rotate_vectors_y_axis":   (jnp.pi / 4,),
    "rotate_vectors_z_axis":   (jnp.pi / 4,),
    "rotate_vectors_axis":     (jnp.array([0.0, 0.0, 1.0]), jnp.pi / 4),
}

REVOLVE_FUNCS = {
    "revolve_field_x",
    "revolve_field_y",
    "revolve_field_z",
}

# ----------------------------------------------------------------------------------------------------------------------
# DISCOVERY

all_funcs = {
    name: fn for name, fn in inspect.getmembers(vmj, callable) if fn.__module__ == vmj.__name__
}

known = set(PARAMS) | REVOLVE_FUNCS
missing = [name for name in all_funcs if name not in known]
if missing:
    print(f"WARNING: no entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results, failed = batch_calculate_vector_modifications(coor, vmj,
                                                       {"params": PARAMS,
                                                        "revolve_mods": REVOLVE_FUNCS,
                                                        "base": BASE_VEC})

end_time = process_time()

# ----------------------------------------------------------------------------------------------------------------------
# ERROR REPORT

error_report("vector modifications", results, failed, end_time - start_time)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELD FOR SNAPSHOT TESTING

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show_grid:
    plot_vector_fields(results, co_size, coor, plane="XY")
    plot_vector_fields(results, co_size, coor, plane="XZ")
    plot_vector_fields(results, co_size, coor, plane="YZ")