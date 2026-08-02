import inspect
from time import process_time

import numpy as np
import jax.numpy as jnp

from spomso.cores.helper_functions import generate_grid
from spomso.jax_cores import vector_functions_jax as vfj
from spomso.jax_cores.sdf_3D_jax import sdf_sphere

from _test_helpers import plot_vector_fields, batch_calculate_vector_fields, error_report

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

co_size = (4, 4, 4)
co_resolution = (30, 30, 30)  # coarse grid for a legible quiver plot

show_grid = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CANONICAL PARAMETERS

VF_PARAMS = {
    "radial_vector_field_spherical":        (),
    "radial_vector_field_cylindrical":      (),
    "hyperbolic_vector_field_cylindrical":  (),
    "awn_vector_field_cylindrical":         (2.0,),
    "vortex_vector_field_cylindrical":      (),
    "aar_vector_field_cylindrical":         (jnp.pi / 4,),
    "aav_vector_field_cylindrical":         (jnp.pi / 4,),
    "x_vector_field":                       (),
    "y_vector_field":                       (),
    "z_vector_field":                       (),
}

_n = coor.shape[1]
DEFINE_INPUTS = {
    "cartesian_vector_field":  jnp.stack([
        jnp.ones(_n),
        jnp.zeros(_n),
        jnp.linspace(-1.0, 1.0, _n),
    ]),
    "spherical_vector_field":  jnp.stack([
        jnp.ones(_n),
        jnp.linspace(0.0, 2 * jnp.pi, _n),
        jnp.linspace(0.0, jnp.pi, _n),
    ]),
    "cylindrical_vector_field": jnp.stack([
        jnp.ones(_n),
        jnp.linspace(0.0, 2 * jnp.pi, _n),
        jnp.linspace(-1.0, 1.0, _n),
    ]),
}

# ----------------------------------------------------------------------------------------------------------------------
# DISCOVERY

all_funcs = {
    name: fn
    for name, fn in inspect.getmembers(vfj, callable) if fn.__module__ == vfj.__name__ and not name.startswith("_")
}

known = set(VF_PARAMS) | set(DEFINE_INPUTS) | {"from_sdf"}
missing = [name for name in all_funcs if name not in known]
if missing:
    print(f"WARNING: no entry for: {missing}")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE

start_time = process_time()

results, failed = batch_calculate_vector_fields(coor, vfj,
                                                {"params": VF_PARAMS, "inputs": DEFINE_INPUTS,
                                                 "base": sdf_sphere, "resolution": co_resolution})

end_time = process_time()


# ----------------------------------------------------------------------------------------------------------------------
# ERROR REPORT

error_report("vector fields", results, failed, end_time - start_time)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELD FOR SNAPSHOT TESTING

field = np.stack(list(results.values()))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show_grid:
    plot_vector_fields(results, co_size, coor, plane="XY")