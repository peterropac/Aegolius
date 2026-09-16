---
# Changelog
---

## [1.5.2.dev2] — 2026-09-16

### Documentation

- Added the `privacy` plugin to `mkdocs.yml`.

### Fixes

- Fixed some docstring typos.


---

## [1.5.1] — 2026-09-15

A packaging and documentation release. There are no API changes and no
behavioural changes to geometry, fields or the JAX backend. The version bump
exists so that the corrected package metadata and README reach PyPI, since
those are only published as part of a release.

---

### Packaging

- **README on PyPI.** Added a `README.md` pypi.org.

- **Summary rewritten.** The one-line description shown in PyPI search results
  now mentions GPU execution and automatic differentiation.

- **Updated keywords.** Such as: signed distance
  function/field, implicit surface, procedural geometry, constructive solid
  geometry, level set, vector field, JAX, automatic differentiation.

- **Trove classifiers.** Expanded to a full set: intended audience
  (Science/Research, Developers, Education), supported Python versions
  (3.10–3.13), and subject topics under
  `Scientific/Engineering` (Mathematics, Physics, Visualization) and
  `Multimedia :: Graphics :: 3D Modeling`.

### Documentation

- **Material for MkDocs.** The documentation theme moved from the built-in
  `readthedocs` theme to Material. 

- **Open in Colab badges** added to every example notebook.
  Each notebook gains a badge linking to Colab and a setup cell that installs
  SPOMSO only when running there, so examples can be run in a browser with no
  local installation.

- **Status badges** in the README.

### Fixes

- Corrected the docstring of `sdf_rounded_box_2D`.

### Project

- **`CITATION.cff`** added, so GitHub renders a "Cite this repository" button
  with one-click APA and BibTeX export pointing at the Zenodo DOI.

- **GitHub Discussions** enabled for questions and usage help.

- SPOMSO submitted to the `awesome-jax` curated list.

---

## [1.5.0] — 2026-07-31

This release adds `__repr__` methods across the geometry and vector-field
classes, a JAX conversion of the vector-field modules, a coverage-oriented
test harness including a numpy⟷JAX parity test, and fixes several silent
backend inconsistencies uncovered by that test.

---

### New features

- **`__repr__` methods** on the base `GenericGeometry` and `VectorField` classes,
  inherited by every subclass. Uses `inspect.signature(cls.__init__)` and the
  established `arg → self._arg` convention to auto-extract constructor arguments,
  and shows any modifications and transformations that have been applied. For
  callables (parametric curves, custom modification functions) and NumPy arrays,
  a `_short_repr` helper keeps the output readable. Custom modifications that
  follow the `self._mod.append("method_name")` convention are picked up
  automatically.

  Example:
  ```
  >>> Box(2, 4, 6)
  Box(a=2, b=4, c=6)
  >>> t = Torus(1.0, 0.2); t.twist(0.5); t.rotate(np.pi/2, (1,0,0))
  >>> t
  Torus(primary_radius=1.0, secondary_radius=0.2)  Modifications: [twist]  Transformations: [rotate]
  ```

  Note that an object returned by `CombineGeometry.combine()` is a bare
  `GenericGeometry` wrapping an anonymous function, so its `repr` shows the
  wrapper rather than the shapes that went into it.

- **`spomso.jax_cores.vector_functions_jax`** — full JAX conversion of
  `spomso.cores.vector_functions`. All 14 functions
  (`cartesian_vector_field`, `spherical_vector_field`, `cylindrical_vector_field`,
  the `radial` / `hyperbolic` / `awn` / `vortex` / `aar` / `aav_vector_field_cylindrical` vector fields,
  `x_vector_field`, `y_vector_field`, `z_vector_field`, `from_sdf`) work under
  `@jax.jit` with sdf_2D_jax-style docstrings.

- **`spomso.jax_cores.vector_modifications_jax`** — full JAX conversion of
  `spomso.cores.vector_modification_functions`. All 13 functions
  (`batch_normalize`, `add_vectors`, `subtract_vectors`, `rescale_vectors`,
  the `phi` / `theta` / `x` / `y` / `z` / `axis_rotation` functions, and
  `revolve_field_x/y/z`) work under `@jax.jit`.

- **Autodiff optimization example for vector fields**:
  `examples/autodiff/vector_field_optimization.py` (with matching `.ipynb`).
  Two counter-rotating donut vortex vector fields are placed at optimizable
  positions in a 2D plane; the optimizer maximizes the peak value of the total
  field's x-projection along a horizontal monitor line. Demonstrates using
  SPOMSO's `vortex_vector_field_cylindrical`, `sdf_circle`, and
  `gaussian_boundary_jax` together with JAX autodiff and Optax.

### Testing infrastructure

- **`tests/run_tests.py`** — a single-script test harness. Walks every `.py`
  example with an `.ipynb` counterpart plus every `jax_test_all_*.py` file in
  `tests/coverage/`, runs each with headless matplotlib/plotly, and compares
  summary statistics (`shape`, `min`, `max`, `mean`) of a designated output
  variable against a stored JSON snapshot in `tests/test_snapshots/`. Usage:

  ```
  python tests/run_tests.py             # check against stored snapshots
  python tests/run_tests.py --update    # accept current output as new baseline
  python tests/run_tests.py --slow      # also run examples on the SLOW list
  ```

  Examples signal their output by assigning the final result to a recognized
  variable name (`sdf_values`, `field`, `result`, `components`).

- **Coverage-oriented test files** in `tests/coverage/`, each exercising every
  function in one `jax_cores` submodule against a canonical grid and aggregating
  results into a `field` variable that `run_tests.py` picks up:

  | file | module | coverage                                |
  |---|---|-----------------------------------------|
  | `jax_test_all_2d_sdfs.py` | `sdf_2D_jax` | 12 SDFs                                 |
  | `jax_test_all_3d_sdfs.py` | `sdf_3D_jax` | 19 SDFs                                 |
  | `jax_test_all_modifications.py` | `modifications_jax` | 42 modifications                        |
  | `jax_test_all_combine.py` | `combine_jax` | 19 combines (all four signature shapes) |
  | `jax_test_all_vector_functions.py` | `vector_functions_jax` | 13 vector fields                        |
  | `jax_test_all_vector_modifications.py` | `vector_modifications_jax` | 13 vector field modifications           |

  Combined, these directly cover 118 of the ~141 public symbols in
  `spomso.jax_cores` (~83%). The remaining are helper functions and
  post-processing utilities exercised indirectly.

- **Similar numpy-based coverage-oriented test files** are also present in `tests/coverage/`.
  More specifically:

  | file                               | module                          | coverage                      |
  |------------------------------------|---------------------------------|-------------------------------|
  | `test_all_2d_sdfs.py`              | `sdf_2D`                        | 16 SDFs                       |
  | `test_all_3d_sdfs.py`              | `sdf_3D`                        | 19 SDFs                       |
  | `test_all_combine.py`              | `combine`                       | 13 combines                   |
  | `test_all_modifications.py`        | `modifications`                 | 51 modifications              |
  | `test_all_post_processing.py`      | `post_processing`               | 13 post-processing functions  |
  | `test_all_vector_functions.py`     | `vector_functions`              | 13 vector fields              |
  | `test_all_vector_modifications.py` | `vector_modification_functions` | 13 vector field modifications |

- **`.github/workflows/test.yml`** — CI runs the test suite on every push and
  pull request under Python 3.10.

> **A note on comparing the two backends.** The parity claims below hold when
> 64-bit JAX is enabled (`config.update("jax_enable_x64", True)`). Under JAX's
> default float32, the same comparisons differ by ~2e-7 purely from precision,
> which can look like a remaining bug. Enable x64 before checking parity.

### Repository restructure

The `Code/` directory has been flattened. The publishable pip package now
lives at the repo top level rather than under `Code/spomso/`, and examples,
docs, and tests sit next to it:

```
Aegolius/
├── spomso/          # publishable package
│   ├── pyproject.toml
│   └── spomso/      # source
├── examples/
├── docs/
├── files/
├── tests/
└── ...
```

This is a repo-layout change only; installed users are unaffected.

`pyproject.toml` also gained explicit optional dependency groups:
`[autodiff]` (jax, jaxlib, optax), `[plot]` (matplotlib, plotly), and `[all]`.

---

### Bug fixes

- **`SegmentedLine3D(points, closed=False)` raised `TypeError` on every call —
  and `closed=False` is the default.** The open-curve branch was wired to
  `sdf_segmented_curve_3d`, whose signature is `(co, points, t)`, but only
  `points` was passed:

  ```
  TypeError: sdf_segmented_curve_3d() missing 1 required positional argument: 't'
  ```

  The class was therefore unusable in its default configuration. It is now
  wired to `sdf_segmented_line_3d` (open) and `sdf_closed_segmented_line_3d`
  (closed). The 2D `SegmentedLine` was unaffected — it already used the right
  function for both branches.
- **26 stray `print()` calls removed** from library code paths that run on
  ordinary use, where they dumped raw intermediate arrays to stdout:
  - `cores/triangulation_functions.py` — 23 calls across `check_intersection_all`
    (10), `create_points_sets` (11), and `interior_polygon` (2). Reached by
    `Polygon` with concave vertices and by any direct `triangulate()` call.
  - `cores/vector_functions.py` — 3 calls in `awn_vector_field_cylindrical`,
    `aar_vector_field_cylindrical`, and `aav_vector_field_cylindrical`. Reached
    via `WindingCylindricalVectorField`,
    `AngledRadialCylindricalVectorField`, and
    `AngledVortexCylindricalVectorField`.
- **`sdf_polygon_2d` mutated the caller's coordinate array.** It sliced with
  `co = co[:2]` and then wrote into the result, corrupting the caller's grid —
  the same class of bug as the five SDFs fixed in 1.4.0. Fixed by
  `co = co[:2].copy()`.
- **`revolve_field_y` in `jax_cores/vector_modifications_jax.py`** had a
  one-character typo on the z-component: `vec[1, :] * sa` instead of
  `vec[0, :] * sa`. The output disagreed with the non-JAX version by up to
  ~7 on a random test grid — a large silent error at any point where the
  field has a non-trivial x-component. Fixed; JAX and non-JAX versions now
  agree to floating-point precision on all three `revolve_field_*` functions.
- **Two functions in `spomso.cores.vector_functions` were previously unusable**:
  `hyperbolic_vector_field_cylindrical` and `awn_vector_field_cylindrical`
  called `cylindrical_define(1, alpha, np.zeros(...))` with three positional
  arguments, but `cylindrical_define` takes only one. Both now inline the
  correct math (`u = cos(alpha)`, `v = sin(alpha)`, `z = zeros(...)`) and
  work. This also means `HyperbolicCylindricalVectorField` and
  `WindingCylindricalVectorField` are usable for the first time.
- **`sdf_ngon` in `jax_cores/sdf_2D_jax.py`** was missing the `co = co[:2, :]`
  2D-slice at the top of the function, which meant when called with a 3D
  coordinate grid, `r_ = jnp.linalg.norm(co, axis=0)` incorrectly included
  the z-component. Disagreement with the numpy version was up to ~1.9 at
  points with large |z|. Fixed; both backends now match to floating-point
  precision.
- **`smoothmin_poly2` and `smoothmin_poly3` were inconsistent between backends**.
  The JAX versions regularized with `a = jnp.abs(a) + 0.001` to avoid
  divide-by-zero under `@jax.jit`, while the numpy versions branched with
  `if a == 0` but didn't apply `abs`. Outputs disagreed by up to ~2.5e-4
  for `a > 0`, and by even more for `a < 0` because the numpy branch
  didn't handle negative `a` the same way. Both are now aligned: JAX uses
  a gradient-safe `jnp.where(a == 0, ...)` branch, numpy uses its
  existing `if a == 0` branch, and both apply `abs(a)`. Backends now
  agree to floating-point precision for all `a`.

  Note that dropping the `+ 0.001` regularization changes **JAX** output for
  *every* non-zero `a`, not just at `a = 0`: by ~2.5e-4 for `smoothmin_poly2`
  and ~1.7e-4 for `smoothmin_poly3`. If you tuned smoothing widths against
  1.4.0 JAX output, re-check them.
- **`VectorField.theta` in `cores/geom.py`** only returned correct values for normalized vector fields.
  Now `VectorField.theta` is calculated via `np.arccos(vec[2] / mag)`, and returns
  `0.0` rather than `NaN` for zero-length vectors. Calling `.normalize()` purely to
  make `.theta()` behave is no longer necessary.
- **`generate_grid` added to `jax_cores/helper_functions.py`**, and re-exported
  from `spomso.jax_cores`. It mirrors the numpy version but returns `jnp` arrays.
  As a side effect it is the better choice for autodiff work: `jnp.linspace` does
  not produce a bitwise-exact `0.0` at the domain midpoint, so the resulting grid
  misses the `r = 0` singularity of the radial SDFs by ~1e-17 and gradients stay
  finite. The numpy `generate_grid` places a point at exactly `r == 0`, which makes
  `jacfwd` through e.g. `sdf_circle` return `NaN` unless the coordinates are nudged.
- **`generate_grid` in `cores/helper_functions.py` crashed on 1-D inputs** such as
  `generate_grid((4,), (10,))`, raising
  `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
  Fixed in both implementations by squeezing `size` and `resolution` before use.
  Relatedly, the jitted `resolution_conversion` in `jax_cores` now casts its
  result with `.astype(int)`; it could previously return a float dtype and break
  downstream reshapes.
- **Some JAX and Numpy SDFs in `cores/sdf_2D`, `cores/sdf_3D`, `jax_cores/sdf_2D_jax`, `jax_cores/sdf_3D_jax` previously
  rejected `jnp.ndarray` or `np.ndarray` inputs**.  Now non-`jnp.ndarray` or non-`np.ndarray`
  inputs are converted to `jnp.ndarray` or `np.ndarray`, where needed. Tuples,
  lists, NumPy arrays, and JAX arrays are now accepted interchangeably for both
  coordinates and vector parameters, in both backends.
- **`sdf_triangle_2d` (both backends) now accept 2- or 3-component vertex vectors.**
  They previously assumed the vertex vectors matched
  the internal coordinate slicing, so passing 2-component vertices to the 3D
  variant broke the edge vectors. The functions now slice consistently
  (`p0[:2]`, `p1[:2]`, `p2[:2]`).

### Consistency and completeness

- **`sdf_closed_segmented_line_2d` and `sdf_closed_segmented_line_3d`** added to
  `spomso.cores.sdf_2D` and `spomso.cores.sdf_3D`. The JAX side already had both;
  the numpy side had neither. Implemented as wrappers around
  `sdf_segmented_line_*d` that take the minimum with the closing segment. Both
  are re-exported from `spomso.cores`.
- **`check_convex_all` and `interior_convex`** now re-exported from
  `spomso.jax_cores` at the top level (they were only reachable through
  `spomso.jax_cores.sdf_2D_jax`). The non-JAX side already exported them.
- **`VectorField` and `transformations_jax` functions added to
  package-level namespaces**: `spomso.cores` now exports `VectorField`
  directly; `spomso.jax_cores` now exports `compound_euclidean_transform_sdf`,
  `move_sdf`, `rotate_sdf`, `scale_sdf`, and `generate_grid`.

  In total `spomso.cores` gained five names (`VectorField`, `SolidAngle`,
  `PointCloud3D`, `sdf_closed_segmented_line_2d`, `sdf_closed_segmented_line_3d`)
  and `spomso.jax_cores` gained 34. **No names were removed from either
  namespace.** In particular the `LCWG*` and `lcwg1_*` names remain
  deliberately absent from `spomso.cores` — breaking change #6 of 1.4.0 was not
  reverted by this pass, and they are still imported from
  `spomso.cores.geom_vector_special` and `spomso.cores.vector_functions_special`.
- **`sdf_braid` parameter order** reordered from `(co, length, R, r, pitch)`
  to `(co, R, r, length, pitch)` for consistency with `sdf_chainlink`,
  from which it is derived. The `Braid` class signature is unchanged.
- `SolidAngle` and `PointCloud3D` added to `spomso.cores` (previously
  only reachable through `spomso.cores.geom_3d`).
- **`Polygon.n_sides`** is now derived from the vertex array
  (`self._vertices.shape[1]`) instead of being cached at construction, so it
  stays correct if the vertices change.
- **Private attribute renames to satisfy the `arg → self._arg` convention** that
  the new `__repr__` relies on. These are private, but noted for anyone reaching
  into them: `_pradius` → `_primary_radius`, `_sradius` → `_secondary_radius`,
  `_sangle` → `_start_angle`, `_eangle` → `_end_angle`,
  `_round_corners` → `_rounding`, `_curve` → `_parametric_curve`,
  `_c_params` → `_parametric_curve_parameters`. `VectorFieldFromSDF` also now
  stores `_grid_resolution`, and the `LCWG*` classes store `_co_resolution`
  and `_sign`.
- **Docstring and comment corrections**: `NGon.radius` / `NGon.n_sides` now say
  "regular polygon" rather than "n-gon"; `RadialCylindricalVectorField` and
  `HyperbolicCylindricalVectorField` docstrings now identify the line
  `x=0, y=0` as the z-axis; the section banner in
  `cores/vector_modification_functions.py` corrected from
  "VECTOR TRANSFORM FUNCTIONS" to "VECTOR MODIFICATION FUNCTIONS".

---

### New Guides

**New LLM guides generated by Claude Opus 5**, which are consistent with version **1.5.0** of **SPOMSO**.

### Removed

- **`ModifyObject.signed_old`** — an unused, slower duplicate of `ModifyObject.signed`.
  Same algorithm (grid-based UDF→SDF via ray-crossing count), but with a Python loop
  instead of `np.cumsum` and no boundary padding. If you were using it, `signed`
  produces better results faster and it is a drop-in replacement.
- **`GenericGeometry2D` and `GenericGeometry3D`** — dead duplicates of
  `GenericGeometry` in `cores/geom_2d.py` and `cores/geom_3d.py`. Neither was
  exported or referenced anywhere; use `GenericGeometry` for both 2D and 3D.
- **`SegmentedLine.sdf_closed_curve()` and `SegmentedLine3D.sdf_closed_curve()`** —
  the closing logic moved into the new module-level
  `sdf_closed_segmented_line_2d` / `sdf_closed_segmented_line_3d` functions. The
  method still exists on `ParametricCurve` and `SegmentedParametricCurve`.

---

## [1.4.0] — 2026-05-15

This release bundles a series of bug fixes and an API consolidation pass.

---

### Bug fixes

#### Silent correctness bugs

- **`subtract_vectors` now actually subtracts** (`cores/vector_modification_functions.py`). Previously, `ModifyVectorObject.subtract(field)` performed `vf + field` instead of `vf - field`.
- **`move`/`move_sdf` was not working due to a malformed `np.subtract` call** (`cores/modifications.py`, `jax_cores/transformations_jax.py`). `np.subtract(co.T - move_vector)` passes a single argument to a two-argument ufunc, which raises `TypeError` — fixed to `np.subtract(co.T, move_vector)`.
- **`smoothmin_poly3(x, y, 0)` returned `np.minimum(y, y)`** instead of `np.minimum(x, y)` (`cores/combine.py`). Single-character typo, but for `a=0` the function silently dropped `x`.
- **`smoothmin_poly2(x, y, 0)` divided by zero** (no `a==0` guard). Now guarded to match `smoothmin_poly3`. (The two backends were still not fully aligned after this change; see the `smoothmin` entry under 1.5.0.)
- **`sigmoid_falloff` appended to the wrong list** (`cores/modifications.py`). It was using `self._pmod.append(...)`, but `ModifyObject` only has `self._mod` — `_pmod` belongs to `PostProcess`. Any call raised `AttributeError`.
- **In-place mutation of caller's `co` array.** Five SDF functions wrote to `co[...]` without copying first, corrupting the caller's coordinate grid. Fixed by `co = co.copy()` at function entry:
  - `sdf_3D.sdf_arc_3d`
  - `sdf_3D.sdf_cone`
  - `sdf_3D.sdf_solid_angle`
  - `sdf_3D.sdf_chainlink`
  - `sdf_3D.sdf_braid`

  (`sdf_2D.sdf_polygon_2d` had the same bug and was missed; it is fixed in 1.5.0.)

#### Packaging / import bugs

- **Fixed imports from `spomso.jax_cores` in the `__init__.py`**. Functions `shear_xz, shear_yz, shear_xy, shear_zy, shear_yx, shear_zx` are now correctly imported.
- **Removed dead code from `cores/sdf_2D.py`**: the functions `sdf_arc_positive_only` and `sdf_sector_old`, plus the `SectorOld` class.
- **Cleaned up unused JAX imports** (`fori_loop`, `functools.partial`).

#### Mathematical bugs

- **`Box`, `Rectangle`, `RoundedRectangle` `.a`/`.b`/`.c`/`.size` properties returned half the documented value.** `Box(2, 4, 6).a` was `1.0`. The constructors stored half-extents internally; the SDF needed half-extents but the public properties exposed them too. Fixed by changing the SDF functions to take full extent and storing full extent in the properties.
- **`elongation` now elongates by the full requested length, not half.** The non-JAX implementation pre-divided `elongate_vector` by 2, while the JAX implementation did not. The docstring says "elongate by the length of the vector in each respective direction", which means `(2, 0, 0)` should give a 2-unit elongation along x. Both backends now agree.
- **`sdf_triangle_3d` in `jax_cores` had leftover `/1` cruft** removed (cosmetic; no semantic effect).

#### `_mod` label alignment

Four `ModifyObject` methods appended labels that didn't match the method name. After the fix `obj.modifications` lists what was actually called:

| method | old label | new label |
|---|---|---|
| `axis_revolution` | `revolution` | `axis_revolution` |
| `curve_instancing` | `parametric_curve_instancing` | `curve_instancing` |
| `aligned_curve_instancing` | `aligned_parametric_curve_instancing` | `aligned_curve_instancing` |
| `fully_aligned_curve_instancing` | `fully_aligned_parametric_curve_instancing` | `fully_aligned_curve_instancing` |

---

### Breaking API changes

These will require user code changes when upgrading from 1.3.0:

1. **`sdf_box(co, a, b, c)` → `sdf_box(co, size)`.** Now takes a single 3-tuple/array of full side lengths. Old code: `sdf_box(co, 1, 2, 3)`. New: `sdf_box(co, (1, 2, 3))`.

2. **`sdf_box_2d(co, size)` semantics:** `size` now means **full extent**. Old behaviour treated `size` as half-extent. Code calling `sdf_box_2d(co, (1, 2))` directly now produces a 1×2 rectangle (previously 2×4). The OOP `Rectangle(a, b)` class is unaffected — it always produced an a×b rectangle.

3. **`sdf_rounded_box_2d(co, size, rounding)` semantics:** same change as `sdf_box_2d`.

4. **`Box(a, b, c).a`, `.b`, `.c` now return `a`, `b`, `c`** (previously returned `a/2`, `b/2`, `c/2`). Same for `Rectangle.a`, `Rectangle.b`, `Rectangle.size`, `RoundedRectangle.a`, `RoundedRectangle.b`, `RoundedRectangle.size`. Anyone reading these properties to read back side lengths will now get the correct (documented) value.

5. **`compound_euclidian_transform_sdf` renamed to `compound_euclidean_transform_sdf`** (`jax_cores/transformations_jax.py`). Correct spelling of the word ***Euclidean*** throughout the project.

6. **`spomso.cores` no longer re-exports** `LCWG2D`, `LCWG3Dp1`, `LCWG3Dm1`, `lcwg1_p1`, `lcwg1_m1`, `lcwg1_2d`. These are still importable from `spomso.cores.geom_vector_special` and `spomso.cores.vector_functions_special` respectively.

7. **`sdf_arc_positive_only`, `sdf_sector_old`, and the `SectorOld` class are removed.** They were unfinished placeholders.

8. **`smoothmax_boltz(x, y, width)` renamed third parameter to `a`** to match the JAX backend and the other `smooth*` functions.

9. **`elongation(elongate_vector)` now elongates by the full vector length, not half.** The non-JAX path previously pre-divided the vector by 2, so `(4, 0, 0)` added 2 units of length instead of 4; the JAX path was already correct. Both now add the full 4 units and agree bit-for-bit. Non-JAX code that relied on the old behaviour should halve its vectors — `examples/scalar/3D/braid_3D.py` was updated to `(2., 0., 0.)`.