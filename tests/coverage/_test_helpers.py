import inspect
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax.scipy.spatial.transform import Rotation

from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf
from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.combine import CombineGeometry


# ----------------------------------------------------------------------------------------------------------------------
# COMMON

def compare_fields(a, b, atol=1e-5, rtol=1e-5):
    match = np.allclose(a, b, atol=atol, rtol=rtol)
    diff = np.max(np.abs(a - b))
    return match, diff

def error_report(operations_string, results, failed, time_diff):
    print(f"Applied {len(results)} {operations_string} in {time_diff:.2f} s")
    if failed:
        print(f"\n{len(failed)} FAILED:")
        for name, err in sorted(failed.items()):
            print(f"  {name}: {err[:200]}")

# ----------------------------------------------------------------------------------------------------------------------
# PLOTS

def plot_fields(results, co_size, coor, co_res_new, plane="XY"):
    n = len(results)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axs = np.atleast_1d(axs).flatten()

    if plane.upper()=="XY":
        midplane_slice = np.index_exp[:, :, co_res_new[2] // 2]
        extent_ = (-co_size[0] / 2, co_size[0] / 2, -co_size[1] / 2, co_size[1] / 2)
        mask_ = coor[2] == 0
        cx = coor[0, mask_].reshape(co_res_new[0], co_res_new[1])
        cy = coor[1, mask_].reshape(co_res_new[0], co_res_new[1])
    if plane.upper()=="XZ":
        midplane_slice = np.index_exp[:, co_res_new[1] // 2, :]
        extent_ = (-co_size[0] / 2, co_size[0] / 2, -co_size[2] / 2, co_size[2] / 2)
        mask_ = coor[1] == 0
        cx = coor[0, mask_].reshape(co_res_new[0], co_res_new[2])
        cy = coor[2, mask_].reshape(co_res_new[0], co_res_new[2])
    if plane.upper()=="YZ":
        midplane_slice = np.index_exp[co_res_new[0] // 2, :, :]
        extent_ = (-co_size[1] / 2, co_size[1] / 2, -co_size[2] / 2, co_size[2] / 2)
        mask_ = coor[0] == 0
        cx = coor[1, mask_].reshape(co_res_new[1], co_res_new[2])
        cy = coor[2, mask_].reshape(co_res_new[1], co_res_new[2])

    for ax, (name, vol) in zip(axs, sorted(results.items())):
        if len(vol.shape) == 2:
            midplane = vol[:, :]
        else:
            midplane = vol[midplane_slice]
        ax.imshow(midplane.T, cmap="binary_r", extent=extent_, origin="lower")
        try:
            cs = ax.contour(cx, cy, midplane, cmap="plasma_r", linewidths=1)
            ax.clabel(cs, inline=True, fontsize=9)
        except Exception:
            pass
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[len(results):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()

def plot_vector_fields(results, co_size, coor, plane="XY"):
    n = len(results)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axs = np.atleast_1d(axs).flatten()

    if plane.upper() == "XY":
        extent_ = (-co_size[0] / 2, co_size[0] / 2, -co_size[1] / 2, co_size[1] / 2)
        mask_ = coor[2] == 0
        msx = np.index_exp[0, mask_]
        msy = np.index_exp[1, mask_]
    if plane.upper() == "XZ":
        extent_ = (-co_size[0] / 2, co_size[0] / 2, -co_size[2] / 2, co_size[2] / 2)
        mask_ = coor[1] == 0
        msx = np.index_exp[0, mask_]
        msy = np.index_exp[2, mask_]

    if plane.upper() == "YZ":
        extent_ = (-co_size[1] / 2, co_size[1] / 2, -co_size[2] / 2, co_size[2] / 2)
        mask_ = coor[0] == 0
        msx = np.index_exp[1, mask_]
        msy = np.index_exp[2, mask_]

    for ax, (name, vec_field) in zip(axs, sorted(results.items())):
        x = coor[msx]
        y = coor[msy]
        u = vec_field[msx]
        v = vec_field[msy]

        step = max(1, len(x) // 400)
        ax.quiver(x[::step], y[::step], u[::step], v[::step],
                  scale_units="xy", scale=4, width=0.005)
        ax.set_xlim(extent_[0], extent_[1])
        ax.set_ylim(extent_[2], extent_[3])
        ax.set_aspect("equal")

        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[len(results):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE




def evaluate_sdfs(sdfs, params, coor, resolution):
    vec = jnp.zeros(3)
    rot_mat = Rotation.from_euler("z", 0).as_matrix()
    scale = 1.0

    results = {}
    for name, fn in sorted(sdfs.items()):
        if name not in params:
            continue
        f = compound_euclidean_transform_sdf(fn, rot_mat, vec, scale)
        sdf_flat = f(coor, *params[name])
        results[name] = np.asarray(smarter_reshape(sdf_flat, resolution))

    return results

def evaluate_nt_sdfs(sdfs, params, coor, resolution):

    results = {}
    for name, fn in sorted(sdfs.items()):
        if name not in params:
            continue
        sdf_flat = fn(coor, *params[name])
        results[name] = np.asarray(smarter_reshape(sdf_flat, resolution))

    return results

def compare_sdf_implementations(coor, fn_np, fn_jax, args, args_jax):
    a = np.asarray(fn_np(coor.copy(), *args))
    b = np.asarray(fn_jax(jnp.asarray(coor), *args_jax))
    return compare_fields(a, b, atol=1e-5, rtol=1e-5)


def evaluate_modifications(mods, params, coor, resolution, base_sdf, base_params):
    results = {}
    failed = {}

    for name, mod_fn in sorted(mods.items()):
        if name not in params:
            continue
        try:
            modified = mod_fn(base_sdf, *params[name])
            sdf_flat = modified(coor, *base_params)
            results[name] = np.asarray(smarter_reshape(sdf_flat, resolution))
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    return results, failed

def evaluate_class_modifications(mods, params, coor, base_function):
    results = {}
    failed = {}

    for name in sorted(params):
        if name not in mods:
            failed[name] = "not a ModifyObject method"
            continue
        try:
            obj = base_function()
            method = getattr(obj, name)
            method(*params[name])
            sdf_flat = obj.create(coor)
            results[name] = np.asarray(sdf_flat)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    return results, failed


def evaluate_post_processing(mods, params, coor, resolution, base_sdf, base_params):
    results = {}
    failed = {}

    for name, mod_fn in sorted(mods.items()):
        if name not in params:
            continue
        try:
            sdf_ = base_sdf(coor, *base_params)
            field = smarter_reshape(sdf_, resolution)
            results[name] = mod_fn(field, *params[name])
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    return results, failed


def evaluate_combine(params, combines, inputs, resolution):
    results = {}
    failed = {}

    for name, extra in params.items():
        fn = combines.get(name)
        if fn is None:
            failed[name] = "not found"
            continue
        try:
            out = fn(*inputs, *extra)
            results[name] = np.asarray(smarter_reshape(out, resolution))
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    return results, failed

def evaluate_meta_combine(name, combined_function, coor, resolution):
    results = {}
    failed = {}

    try:
        out = combined_function(coor)
        results[name] = np.asarray(smarter_reshape(out, resolution))
    except Exception as e:
        failed[name] = f"{type(e).__name__}: {e}"
    return results, failed


def evaluate_vector_fields(vecs, params, coor):
    results = {}
    failed = {}

    for name, extra in params.items():
        fn = vecs.get(name)
        if fn is None:
            failed[name] = "not found"
            continue
        try:
            if coor is not None:
                out = fn(coor, *extra)
            else:
                out = fn(extra)
            results[name] = np.asarray(out)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    return results, failed

def evaluate_vector_field_modifications(vecs, params, coor, base):
    results = {}
    failed = {}

    for name, extra in params.items():
        fn = vecs.get(name)
        if fn is None:
            failed[name] = "not found"
            continue
        try:
            if coor is not None:
                out = fn(coor, base)
            else:
                out = fn(base, *extra)
            results[name] = np.asarray(out)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    return results, failed

# ----------------------------------------------------------------------------------------------------------------------
# BATCH EVALUATE

def batch_compare_sdf_implementations(coor, fns_np, fns_jax, params, params_jax):
    results = {}
    for name in params.keys():
        fn_np = getattr(fns_np, name, None)
        fn_jax = getattr(fns_jax, name, None)
        if fn_np is None or fn_jax is None: continue
        ok, diff = compare_sdf_implementations(coor, fn_np, fn_jax, params[name], params_jax[name])
        results[name] = (ok, diff)
    return results


def batch_compare_modifications(coor, make_np_base, jax_base_fn, jax_base_params,
                                np_params, jax_params, mods_jax):
    results = {}
    for name in sorted(np_params.keys() & jax_params.keys()):
        try:
            obj = make_np_base()
            getattr(obj, name)(*np_params[name])
            a = np.asarray(obj.create(coor.copy())).reshape(-1)

            jax_mod_fn = getattr(mods_jax, name, None)
            if jax_mod_fn is None:
                results[name] = (False, float("inf"))
                continue
            modified = jax_mod_fn(jax_base_fn, *jax_params[name])
            b = np.asarray(modified(jnp.asarray(coor), *jax_base_params)).reshape(-1)

            results[name] = compare_fields(a, b, atol=1e-5, rtol=1e-5)
        except Exception as e:
            results[name] = (False, float("nan"))
    return results


def batch_compare_combines(coor, np_pair_builder, jax_pre_evaluated,
                           params, jax_combine_module):
    results = {}

    for np_op, s_params in params.items():
        l = isinstance(s_params, tuple)
        if l:
            jax_name, width = s_params
        else:
            jax_name = s_params

        try:
            combiner = CombineGeometry(np_op)
            s, p = np_pair_builder()

            combined = combiner.combine_parametric(s, p, parameters=width) if l else combiner.combine(s, p)
            a = np.asarray(combined.create(coor.copy())).reshape(-1)

            jax_fn = getattr(jax_combine_module, jax_name, None)
            if jax_fn is None:
                results[np_op] = (False, float("inf"))
                continue

            b1, b2 = jax_pre_evaluated["first"], jax_pre_evaluated["second"]
            b = np.asarray(jax_fn(b1, b2, width)).reshape(-1) if l else np.asarray(jax_fn(b1, b2)).reshape(-1)

            results[np_op] = compare_fields(a, b, atol=1e-5, rtol=1e-5)
        except Exception:
            results[np_op] = (False, float("nan"))

    return results


def batch_calculate_vector_fields(coor, vf, vf_dict):

    vf_params = vf_dict["params"]
    vf_inputs = vf_dict["inputs"]
    base = vf_dict["base"]
    co_resolution = vf_dict["resolution"]


    all_funcs = {
        name: fn
        for name, fn in inspect.getmembers(vf, callable) if fn.__module__ == vf.__name__ and not name.startswith("_")
    }

    results_1, failed_1 = evaluate_vector_fields(all_funcs, vf_params, coor)

    results_2, failed_2 = evaluate_vector_fields(all_funcs, vf_inputs, None)

    results = results_1 | results_2
    failed = failed_1 | failed_2

    try:
        sdf_vals = base(coor, 1.0)
        out = vf.from_sdf(sdf_vals, co_resolution)
        results["from_sdf"] = np.asarray(out)
    except Exception as e:
        failed["from_sdf"] = f"{type(e).__name__}: {e}"

    return results, failed


def batch_calculate_vector_modifications(coor, vm, vm_dict):

    PARAMS = vm_dict["params"]
    REVOLVE_FUNCS = vm_dict["revolve_mods"]
    BASE_VEC = vm_dict["base"]

    all_funcs = {
        name: fn for name, fn in inspect.getmembers(vm, callable) if fn.__module__ == vm.__name__
    }

    results = {}
    failed = {}

    for name, extras in PARAMS.items():
        fn = all_funcs.get(name)
        if fn is None:
            failed[name] = "not found"
            continue
        try:
            out = fn(BASE_VEC, *extras)
            results[name] = np.asarray(out)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    for name in sorted(REVOLVE_FUNCS):
        fn = all_funcs.get(name)
        if fn is None:
            failed[name] = "not found"
            continue
        try:
            out = fn(coor, BASE_VEC)
            results[name] = np.asarray(out)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"

    return results, failed

















