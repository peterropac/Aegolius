import numpy as np
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


from spomso.cores.helper_functions import generate_grid

from spomso.cores import sdf_2D as np_sdf2d
from spomso.jax_cores import sdf_2D_jax as jax_sdf2d

from spomso.cores import sdf_3D as np_sdf3d
from spomso.jax_cores import sdf_3D_jax as jax_sdf3d

from spomso.cores import vector_functions as np_vf
from spomso.jax_cores import vector_functions_jax as jax_vf

from spomso.cores import vector_modification_functions as np_vfm
from spomso.jax_cores import vector_modifications_jax as jax_vfm

from spomso.cores import modifications as np_mods_module
from spomso.jax_cores import modifications_jax

from spomso.jax_cores import combine_jax


from spomso.cores.geom_3d import Box, Sphere
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf




from _test_helpers import batch_compare_sdf_implementations, compare_fields
from _test_helpers import batch_calculate_vector_fields, batch_calculate_vector_modifications
from _test_helpers import batch_compare_modifications, batch_compare_combines


# ----------------------------------------------------------------------------------------------------------------------
# COORDINATES

grid_size = (4, 4, 4)
co_resolution = (40, 40, 40)
coor, _ = generate_grid(grid_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# SDF 2D PARAMS

SDF_2D_PARAMS = {
    "sdf_circle": (1.0,),
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
    "sdf_segmented_line_2d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).T,
    ),
    "sdf_polygon_2d": (
        np.asarray([[-1.2, 1.5, 0], [2.0, 1.5, 0.0], [1.0, -1.0, 0], [-2.0, -1.0, 0.0]]).T,
    )
}

# ----------------------------------------------------------------------------------------------------------------------
# SDF 3D PARAMS

SDF_3D_PARAMS = {
    "sdf_x": (0.5,),
    "sdf_y": (0.5,),
    "sdf_z": (0.5,),
    "sdf_sphere": (1.0,),
    "sdf_cylinder": (0.5, 2.0),
    "sdf_box": (np.asarray([1.5, 2.0, 1.0]),),
    "sdf_torus": (1.0, 0.3),
    "sdf_arc_3d": (1.0, 0.2, 0.0, np.pi),
    "sdf_plane": (np.asarray([0.0, 0.0, 1.0]), 0.5),
    "sudf_plane": (np.asarray([0.0, 0.0, 1.0]), 0.5),
    "sdf_segment_3d": (
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ),
    "sdf_cone": (2.0, np.pi / 6),
    "sdf_oriented_infinite_cone": (np.pi / 6,),
    "sdf_infinite_cone": (np.pi / 6,),
    "sdf_solid_angle": (1.5, -np.pi / 4, np.pi / 4),
    "sdf_triangle_3d": (
        np.asarray([-1.0, -1.0, 0.0]),
        np.asarray([1.0, -1.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    ),
    "sdf_quad_3d": (
        np.asarray([-1.0, -1.0, 0.0]),
        np.asarray([1.0, -1.0, 0.0]),
        np.asarray([1.0, 1.0, 0.0]),
        np.asarray([-1.0, 1.0, 0.0]),
    ),
    "sdf_segmented_line_3d": (
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [1.0, 0.0, 0.0]]).T,
    ),
}

# ----------------------------------------------------------------------------------------------------------------------
# VECTOR PARAMS

VF_PARAMS = {
    "radial_vector_field_spherical":        (),
    "radial_vector_field_cylindrical":      (),
    "hyperbolic_vector_field_cylindrical":  (),
    "awn_vector_field_cylindrical":         (2.0,),
    "vortex_vector_field_cylindrical":      (),
    "aar_vector_field_cylindrical":         (np.pi / 4,),
    "aav_vector_field_cylindrical":         (np.pi / 4,),
    "x_vector_field":                       (),
    "y_vector_field":                       (),
    "z_vector_field":                       (),
}

_n = coor.shape[1]
VF_INPUTS = {
    "cartesian_define":  np.stack([
        np.ones(_n),
        np.zeros(_n),
        np.linspace(-1.0, 1.0, _n),
    ]),
    "spherical_define":  np.stack([
        np.ones(_n),
        np.linspace(0.0, 2 * np.pi, _n),
        np.linspace(0.0, np.pi, _n),
    ]),
    "cylindrical_define": np.stack([
        np.ones(_n),
        np.linspace(0.0, 2 * np.pi, _n),
        np.linspace(-1.0, 1.0, _n),
    ]),
}

VF_INPUTS_JAX = {
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
# VECTOR MODIFICATIONS PARAMS

PARAMS = {
    "batch_normalize":         (),
    "add_vectors":             (np.array([1.0, 0.0, 0.0]),),
    "subtract_vectors":        (np.array([1.0, 0.0, 0.0]),),
    "rescale_vectors":         (2.0,),
    "rotate_vectors_phi":      (np.pi / 4,),
    "rotate_vectors_theta":    (np.pi / 4,),
    "rotate_vectors_x_axis":   (np.pi / 4,),
    "rotate_vectors_y_axis":   (np.pi / 4,),
    "rotate_vectors_z_axis":   (np.pi / 4,),
    "rotate_vectors_axis":     (np.array([0.0, 0.0, 1.0]), np.pi / 4),
}

REVOLVE_FUNCS = {
    "revolve_field_x",
    "revolve_field_y",
    "revolve_field_z",
}

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE SDF

results_2d = batch_compare_sdf_implementations(coor,
                                               np_sdf2d, jax_sdf2d,
                                               SDF_2D_PARAMS, SDF_2D_PARAMS)

results_3d = batch_compare_sdf_implementations(coor,
                                               np_sdf3d, jax_sdf3d,
                                               SDF_3D_PARAMS, SDF_3D_PARAMS)

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE VECTOR FUNCTIONS

results_vf_np, _ = batch_calculate_vector_fields(coor,
                                                 np_vf,
                                                 {"params": VF_PARAMS, "inputs": VF_INPUTS,
                                                  "base": np_sdf3d.sdf_sphere, "resolution": co_resolution})
results_vf_jax, _ = batch_calculate_vector_fields(coor,
                                                  jax_vf,
                                                  {"params": VF_PARAMS, "inputs": VF_INPUTS_JAX,
                                                  "base": jax_sdf3d.sdf_sphere, "resolution": co_resolution})
results_vf = {}
for name in results_vf_np.keys():
    name_ = name.replace("_define", "_vector_field")
    results_vf[name] = compare_fields(results_vf_np[name], results_vf_jax[name_])

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE VECTOR MODIFICATIONS

results_vfm_np, _ = batch_calculate_vector_modifications(coor, np_vfm,
                                                       {"params": PARAMS,
                                                        "revolve_mods": REVOLVE_FUNCS,
                                                        "base": np_vf.radial_vector_field_spherical(coor)})


results_vfm_jax, _ = batch_calculate_vector_modifications(coor, jax_vfm,
                                                       {"params": PARAMS,
                                                        "revolve_mods": REVOLVE_FUNCS,
                                                        "base": jax_vf.radial_vector_field_spherical(coor)})

results_vfm = {}
for name in results_vfm_np.keys():
    results_vfm[name] = compare_fields(results_vfm_np[name], results_vfm_jax[name])

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE MODIFICATIONS

def _displace_fn(coordinates, amplitude):
    return amplitude * np.cos(np.pi * 2 * coordinates[0])

def _displace_fn_jax(coordinates, amplitude):
    return amplitude * jnp.cos(jnp.pi * 2 * coordinates[0])

def _custom_mod(function_, co, sdf_params, mod_params):
    return function_(co, *sdf_params) - mod_params[0]

def _custom_pp(sdf_values, threshold):
    return sdf_values - threshold


MOD_PARAMS = {
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
    "custom_post_process":        (_custom_pp, (0.1,)),
}

def _make_np_box():
    return Box(1.0, 1.0, 1.0)

base_params_jax = (jnp.asarray([1.0, 1.0, 1.0]),)

results_mods = batch_compare_modifications(
    coor, _make_np_box,
    jax_sdf3d.sdf_box, base_params_jax,
    MOD_PARAMS, MOD_PARAMS,
    modifications_jax,
)

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE COMBINES

def _make_np_pair():
    s = Sphere(1.0)
    s.move((-0.5, 0.0, 0.0))
    b = Box(1.0, 1.0, 1.0)
    b.move((0.5, 0.0, 0.0))
    return s, b

# Pre-evaluate the two JAX base SDFs once — element-wise combines take value arrays.
_jax_sphere_fn = compound_euclidean_transform_sdf(
    jax_sdf3d.sdf_sphere, jnp.eye(3), jnp.asarray([-0.5, 0.0, 0.0]), 1.0
)
_jax_box_fn = compound_euclidean_transform_sdf(
    jax_sdf3d.sdf_box, jnp.eye(3), jnp.asarray([0.5, 0.0, 0.0]), 1.0
)
_jax_pre = {
    "first": _jax_sphere_fn(jnp.asarray(coor), 1.0),
    "second": _jax_box_fn(jnp.asarray(coor), jnp.asarray([1.0, 1.0, 1.0])),
}

# Map CombineGeometry operation names -> JAX function names.
NONPARAM_MAP = {
    "UNION2":     "union2",
    "SUBTRACT2":  "subtract2",
    "INTERSECT2": "intersect2",
    "SUM":        "add",
    "DIFFERENCE": "difference",
}
PARAM_MAP = {
    "SMOOTH_UNION2_2":            ("smoothmin_poly2", 0.5),
    "SMOOTH_UNION2":              ("smoothmin_poly3", 0.5),
    "SMOOTH_INTERSECT2_BOLTZMANN": ("smoothmax_boltz", 10.0),
}

COMBINE_MAP = NONPARAM_MAP | PARAM_MAP

results_combines = batch_compare_combines(coor, _make_np_pair, _jax_pre, COMBINE_MAP, combine_jax)

# ----------------------------------------------------------------------------------------------------------------------
# AGGREGATE FIELDS

results = {"2D": results_2d, "3D": results_3d, "VF": results_vf, "VFM": results_vfm,
           "MODS": results_mods, "COMBINES": results_combines}

field = np.asarray([
    diff
    for cat_results in results.values()
    for _, diff in cat_results.values()
])

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE
for name_s, val_s in results.items():
    print(f"\n{name_s}:")
    for name, (ok, diff) in val_s.items():
        if not ok:
            print(f"MISMATCH: {name}, max_diff={diff:.4e}")
        else:
            print(f"OK: {name}, max_diff={diff:.4e}")