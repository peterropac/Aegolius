# Changelog

## [1.4.0] — 2026-05-15

This release bundles a series of bug fixes and an API consolidation pass.

---

## Bug fixes

### Silent correctness bugs

- **`subtract_vectors` now actually subtracts** (`cores/vector_modification_functions.py`). Previously, `ModifyVectorObject.subtract(field)` performed `vf + field` instead of `vf - field`.
- **`move`/`move_sdf` was a not working due to the wrong `np.subtract` call** (`cores/modifications.py:1283`, `jax_cores/transformations_jax.py:31`). `np.subtract(co.T - move_vector)` raised on the inner expression — fixed to `np.subtract(co.T, move_vector)`.
- **`smoothmin_poly3(x, y, 0)` returned `np.minimum(y, y)`** instead of `np.minimum(x, y)` (`cores/combine.py`). Single-character typo, but for `a=0` the function silently dropped `x`.
- **`smoothmin_poly2(x, y, 0)` divided by zero** (no `a==0` guard). Now guarded to match `smoothmin_poly3`.
- **`sigmoid_falloff` appended to the wrong list** (`cores/modifications.py:1372`). It was using `self._pmod.append(...)`, but `ModifyObject` only has `self._mod`. Any call raised `AttributeError`.
- **In-place mutation of caller's `co` array.** Five SDF functions wrote to `co[...]` without copying first, corrupting the caller's coordinate grid. Fixed by `co = co.copy()` at function entry:
  - `sdf_3D.sdf_arc_3d`
  - `sdf_3D.sdf_cone`
  - `sdf_3D.sdf_solid_angle`
  - `sdf_3D.sdf_chainlink`
  - `sdf_3D.sdf_braid`

### Packaging / import bugs

- **Fixed imports from `spomso.jax_cores` in the `__init__.py`**. Functions `shear_xz, shear_yz, shear_xy, shear_zy, shear_yx, shear_zx` are now correctly imported.
- **Removed dead code from `cores/sdf_2D.py`**. Two functions `sdf_arc_positive_only`, `sdf_sector_old`, and the `SectorOld` class are gone.
- **Cleaned up unused JAX imports** (`fori_loop`, `functools.partial`).

### Mathematical bugs

- **`Box`, `Rectangle`, `RoundedRectangle` `.a`/`.b`/`.c`/`.size` properties returned half the documented value.** `Box(2, 4, 6).a` was `1.0`. The constructors stored half-extents internally; the SDF needed half-extents but the public properties exposed them too. Fixed by changing the SDF functions to take full extent and storing full extent in the properties.
- **`smoothmin_poly3` was an inverse copy of `smoothmin_poly2` in JAX as a divide-by-zero safety pattern, then back-ported.** Both poly variants now share the same `|a| + 0.001`-style guard semantics across both backends, except non-JAX retains the explicit `if not a == 0` branch (kept for backwards compatibility — output unchanged for non-zero `a`).
- **`elongation` now elongates by the full requested length, not half.** The non-JAX implementation pre-divided `elongate_vector` by 2, while the JAX implementation did not. The non-JAX version was reading the docstring incorrectly: docstring says "elongate by the length of the vector in each respective direction", which means `(2, 0, 0)` should give a 2-unit elongation along x. Both backends now agree.
- **`sdf_triangle_3d` in `jax_cores` had leftover `/1` cruft** removed (cosmetic; no semantic effect).

### `_mod` label alignment

Four `ModifyObject` methods appended labels that didn't match the method name. After the fix `obj.modifications` lists what was actually called:

| method | old label | new label |
|---|---|---|
| `axis_revolution` | `revolution` | `axis_revolution` |
| `curve_instancing` | `parametric_curve_instancing` | `curve_instancing` |
| `aligned_curve_instancing` | `aligned_parametric_curve_instancing` | `aligned_curve_instancing` |
| `fully_aligned_curve_instancing` | `fully_aligned_parametric_curve_instancing` | `fully_aligned_curve_instancing` |

---

## Breaking API changes

These will require user code changes when upgrading from 1.3.0:

1. **`sdf_box(co, a, b, c)` → `sdf_box(co, size)`.** Now takes a single 3-tuple/array of full side lengths. Old code: `sdf_box(co, 1, 2, 3)`. New: `sdf_box(co, (1, 2, 3))`.

2. **`sdf_box_2d(co, size)` semantics:** `size` now means **full extent**. Old behaviour treated `size` as half-extent. Code calling `sdf_box_2d(co, (1, 2))` directly now produces a 1×2 rectangle (previously 2×4). The OOP `Rectangle(a, b)` class is unaffected — it always produced an a×b rectangle.

3. **`sdf_rounded_box_2d(co, size, rounding)` semantics:** same change as `sdf_box_2d`.

4. **`Box(a, b, c).a`, `.b`, `.c` now return `a`, `b`, `c`** (previously returned `a/2`, `b/2`, `c/2`). Same for `Rectangle.a`, `Rectangle.b`, `Rectangle.size`, `RoundedRectangle.a`, `RoundedRectangle.b`, `RoundedRectangle.size`. Anyone reading these properties to read back side lengths will now get the correct (documented) value.

5. **`compound_euclidian_transform_sdf` renamed to `compound_euclidean_transform_sdf`** (`jax_cores/transformations_jax.py`). Correct spelling of the word ***Euclidean*** throughout the project.

6. **`spomso.cores` no longer re-exports** `LCWG2D`, `LCWG3Dp1`, `LCWG3Dm1`, `lcwg1_p1`, `lcwg1_m1`, `lcwg1_2d`. These are still importable from `spomso.cores.geom_vector_special` and `spomso.cores.vector_functions_special` respectively.

7. **`sdf_arc_positive_only`, `sdf_sector_old`, and the `SectorOld` class are removed.** They were unfinished placeholders.

8. **`smoothmax_boltz(x, y, width)` renamed third parameter to `a`** to match the JAX backend and the other `smooth*` functions.

9. **`elongation(elongate_vector)` now elongates by the full vector length, not half.** If your old code passed `(4, 0, 0)` to mean a 4-unit elongation, that already gave you a 4-unit elongation in non-JAX (because of the `/2` plus an internal `/2` in the SDF — which cancelled to 2 actual units of *added* length, but matched the user's mental model "elongation = vector length"). After the fix the same `(4, 0, 0)` still gives a 4-unit-long elongation but the OOP and JAX paths agree. Examples that called `torus.elongation((4, 0, 0))` should be reviewed — `examples/scalar/3D/braid_3D.py` was updated to `(2., 0., 0.)`.

