# SPOMSO – Automatic Differentiation with JAX
### A practical guide to computing gradients of SDF-based geometry

**Applies to SPOMSO 1.5.0.**

---

## Table of Contents
1. [Why autodiff?](#1-why-autodiff)
2. [Installation](#2-installation)
3. [Two APIs: OOP vs. Functional](#3-two-apis-oop-vs-functional)
4. [The JAX Functional API](#4-the-jax-functional-api)
5. [Euclidean Transformations in the Functional API](#5-euclidean-transformations-in-the-functional-api)
6. [Modifications in the Functional API](#6-modifications-in-the-functional-api)
7. [Combining SDFs in the Functional API](#7-combining-sdfs-in-the-functional-api)
8. [Computing Gradients with `jacfwd`](#8-computing-gradients-with-jacfwd)
9. [Chaining Gradients through Post-processing with `jvp`](#9-chaining-gradients-through-post-processing-with-jvp)
10. [Geometry Optimisation with Optax](#10-geometry-optimisation-with-optax)
11. [Differentiable Vector Fields](#11-differentiable-vector-fields)
12. [Practical Patterns and Pitfalls](#12-practical-patterns-and-pitfalls)
13. [Quick-Reference Cheat Sheet](#13-quick-reference-cheat-sheet)

---

## 1. Why autodiff?

The standard SPOMSO OOP API (described in the main guide) is built on **NumPy** and is the right tool for evaluating geometry. However, NumPy is not differentiable — you cannot ask "how does this SDF field change if I nudge parameter X?"

The **JAX-backed functional API** (`spomso.jax_cores`) solves this by reimplementing SDFs, modifications, combination operations, and — as of 1.5.0 — vector fields as pure functions that JAX can trace. This enables:

- Computing the **gradient of an SDF field with respect to any geometric parameter** (radius, position, rotation angle, smoothing width, …)
- **Propagating those gradients through post-processing** functions (Gaussian falloff, binarisation, etc.)
- **Gradient-based optimisation** of geometry and field parameters using standard optimisers (e.g. `optax`)

---

## 2. Installation

JAX support is an optional extra. Install it alongside SPOMSO:

```bash
pip install -e "path/to/Aegolius/spomso[autodiff]"
# installs jax + jaxlib + optax

# or everything at once:
pip install -e "path/to/Aegolius/spomso[all]"
```

> **Note:** as of 1.5.0 the repository layout was flattened — the installable package sits at `Aegolius/spomso`, not `Aegolius/Code/spomso`. The `[autodiff]` extra includes `optax` as well as `jax` and `jaxlib`.

For GPU/TPU, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) to get the right `jaxlib` wheel.

Enable 64-bit precision (strongly recommended for geometry work):

```python
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)   # optional but very useful during development
```

`jax_debug_nans` is worth keeping on while developing: it turns silent NaN propagation into an immediate error, which matters here because several SDFs are non-differentiable at the origin (see [section 12](#12-practical-patterns-and-pitfalls)).

---

## 3. Two APIs: OOP vs. Functional

| Feature | OOP API (`spomso.cores.*`) | Functional API (`spomso.jax_cores.*`) |
|---|---|---|
| Backend | NumPy | JAX (jit-compiled) |
| Usage style | Method calls on objects | Function composition |
| Transformations | `.move()`, `.rotate()` etc. | `compound_euclidean_transform_sdf(sdf, R, v, s)` |
| Modifications | `.rounding()`, `.onion()` etc. | `rounding(sdf, r)`, `onion(sdf, t)` etc. |
| Combinations | `CombineGeometry("UNION").combine(a, b)` | `combine_2_sdfs(a, b, pa, pb, union2)` |
| Vector fields | `VectorField` subclasses | Plain functions (new in 1.5.0) |
| Differentiability | ✗ | ✓ via `jax.jacfwd`, `jax.jvp`, `jax.grad` |
| Grid setup | `generate_grid(...)` | `generate_grid(...)`, JAX-native (new in 1.5.0) |

Both `generate_grid` and `smarter_reshape` now exist in **both** backends:

```python
from spomso.cores import generate_grid, smarter_reshape        # NumPy; accepts JAX arrays too
from spomso.jax_cores import generate_grid, smarter_reshape    # JAX-native, returns jnp arrays
```

`generate_grid` was added to `spomso.jax_cores` in 1.5.0. **For autodiff work, prefer the JAX version** — beyond returning `jnp` arrays, it avoids a NaN-gradient trap that the NumPy version walks straight into. See [section 12](#12-practical-patterns-and-pitfalls); this is the single most valuable thing to know in this guide.

Either `smarter_reshape` works on either array type; use the JAX one when the result must stay a traced array inside a jitted function.

---

## 4. The JAX Functional API

### Importing primitives

All JAX SDF functions mirror the OOP primitives:

```python
# 2D primitives
from spomso.jax_cores.sdf_2D_jax import (
    sdf_circle, sdf_box_2d, sdf_rounded_box_2d,
    sdf_segment_2d, sdf_triangle_2d,
    sdf_arc, sdf_sector, sdf_inf_sector, sdf_ngon,
    sdf_segmented_line_2d, sdf_closed_segmented_line_2d, sdf_polygon_2d,
)

# 3D primitives
from spomso.jax_cores.sdf_3D_jax import (
    sdf_sphere, sdf_cylinder, sdf_box, sdf_torus, sdf_arc_3d,
    sdf_cone, sdf_infinite_cone, sdf_oriented_infinite_cone, sdf_solid_angle,
    sdf_segment_3d, sdf_segmented_line_3d, sdf_closed_segmented_line_3d,
    sdf_plane, sudf_plane,
    sdf_triangle_3d, sdf_quad_3d,
    sdf_x, sdf_y, sdf_z,
)
```

Two easily confused plane functions:
- `sdf_plane(co, normal, offset)` — a **half-space** (matches the OOP `OrientedPlane`)
- `sudf_plane(co, normal, thickness)` — an **infinite slab** (matches the OOP `Plane`)

### Calling a primitive

Every JAX SDF has the signature `sdf(co, *params)` where `co` is the coordinate cloud `(3, N)`:

```python
from spomso.jax_cores import generate_grid
from spomso.jax_cores.sdf_2D_jax import sdf_circle

coor, res = generate_grid((4, 4), (200, 200))

radius = 1.0
field = sdf_circle(coor, radius)   # shape (N,)
```

Crucially, `radius` is a plain Python float or JAX scalar — JAX can differentiate through it.

### Box-like SDFs take a size vector

As of 1.4.0, the box SDFs take a single vector of **full** side lengths (not `a, b, c` separately, and not half-extents), in both backends:

```python
sdf_box(coor, (1.0, 2.0, 3.0))
sdf_box_2d(coor, (1.0, 2.0))
sdf_rounded_box_2d(coor, (1.0, 2.0), (0.1, 0.1, 0.1, 0.1))
```

As of 1.5.0 the SDF functions coerce their arguments, so tuples, lists, NumPy arrays, and JAX arrays are all accepted interchangeably — and a NumPy coordinate array works in a JAX SDF and vice versa. Earlier versions rejected some combinations (notably a plain tuple into `sdf_box_2d`, which raised `TypeError`).

You still need `jnp` arrays for any value you intend to **differentiate with respect to**, since those have to be traced.

---

## 5. Euclidean Transformations in the Functional API

Transformations wrap an SDF function. As of 1.5.0 they are re-exported from the `spomso.jax_cores` package root:

```python
from spomso.jax_cores import (
    compound_euclidean_transform_sdf,
    move_sdf, rotate_sdf, scale_sdf,
)
# the submodule path also works, and is required on 1.4.0 and earlier:
# from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf
import jax.numpy as jnp

# translate to (1, 0.5, 0), no rotation, no scaling
vec   = jnp.asarray([1.0, 0.5, 0.0])
rot   = jnp.eye(3)
scale = 1.0

moved_circle = compound_euclidean_transform_sdf(sdf_circle, rot, vec, scale)

# moved_circle is now a callable: moved_circle(co, *params)
field = moved_circle(coor, radius)
```

The three single-purpose wrappers are available when you only need one operation:

```python
sdf = move_sdf(sdf_circle, jnp.asarray([1.0, 0.0, 0.0]))
sdf = rotate_sdf(sdf, rot_mat)
sdf = scale_sdf(sdf, 2.0)
```

To rotate with a matrix built from Euler angles:

```python
from jax.scipy.spatial.transform import Rotation

rot_mat = Rotation.from_euler('z', 45, degrees=True).as_matrix()
rotated_circle = compound_euclidean_transform_sdf(sdf_circle, rot_mat, jnp.zeros(3), 1.0)
```

> **Key pattern:** `compound_euclidean_transform_sdf` returns a **new SDF function**. It does not evaluate the field — it produces a composed callable that you evaluate later.

> **Renamed in 1.4.0:** the old misspelled `compound_euclidian_transform_sdf` is gone. Use `compound_euclidean_transform_sdf`.

---

## 6. Modifications in the Functional API

Modifications work the same way — each takes an SDF function and returns a new SDF function:

```python
from spomso.jax_cores.modifications_jax import (
    onion, rounding, rounding_cs, elongation, concentric, boundary,
    mirror, symmetry, rotational_symmetry,
    infinite_repetition, finite_repetition, finite_repetition_rescaled,
    linear_instancing,
    twist, bend, extrusion, revolution, axis_revolution,
    shear_xz, shear_yz, shear_xy, shear_zy, shear_yx, shear_zx,
    displacement, define_volume, invert, sign,
    custom_modification, custom_post_process,
    gaussian_falloff, gaussian_boundary, linear_falloff, sigmoid_falloff,
    positive_sigmoid_falloff, capped_exponential,
    relu, smooth_relu, slowstart, hard_binarization,
    conv_averaging, conv_edge_detection,
)
import jax.numpy as jnp

# Start with the base SDF
sdf = sdf_circle

# Apply onion (hollow shell of thickness 0.2)
sdf = onion(sdf, 0.2)

# Apply mirror (mirror image at (-d/2, 0, 0), original at (d/2, 0, 0))
sdf = mirror(sdf, [-d / 2, 0, 0], [d / 2, 0, 0])

# Apply a Euclidean transform
sdf = compound_euclidean_transform_sdf(sdf, rot_mat, vec, scale)

# Evaluate — all modifications and transforms are applied here
field = sdf(coor, radius)   # radius is passed through to the base sdf_circle
```

Every modification takes `(sdf_function, *modification_params)` and returns a new callable. The chain is built lazily and evaluated in one pass.

### Available modifications

| Function | Arguments | Effect |
|---|---|---|
| `onion(sdf, t)` | thickness | Shell of thickness t |
| `rounding(sdf, r)` | radius | Round edges (thickens) |
| `rounding_cs(sdf, r, bb)` | radius, bounding-box size | Round without growing past the box |
| `concentric(sdf, w)` | width | Two surfaces w apart |
| `boundary(sdf)` | — | Absolute value of the SDF |
| `elongation(sdf, v)` | vector | Stretch by the full length of v |
| `mirror(sdf, a, b)` | point a, point b | Mirror; image at a, original at b |
| `symmetry(sdf, axis)` | axis index | Mirror across an axis plane |
| `rotational_symmetry(sdf, n, r, phase)` | order, radius, phase | n-fold circular pattern |
| `twist(sdf, pitch)` | rad/unit | Twist around z |
| `bend(sdf, r, angle)` | radius, angle | Bend around z |
| `extrusion(sdf, h)` | height | Extrude 2D to 3D |
| `revolution(sdf, r)` | radius | Revolve 2D around the y-axis |
| `axis_revolution(sdf, r, angle)` | radius, tilt | Revolve around a tilted axis |
| `infinite_repetition(sdf, d)` | spacing vector | Tile infinitely |
| `finite_repetition(sdf, size, reps)` | size, repetitions | Tile within a box |
| `linear_instancing(sdf, n, a, b)` | count, endpoints | n copies along a segment |
| `shear_xz(sdf, angle)` (and 5 more) | angle | Named shears |
| `displacement(sdf, fn, params)` | function, params | Perturb the surface |
| `gaussian_falloff(sdf, amp, w)` | amplitude, width | Gaussian falloff at the surface |
| `gaussian_boundary(sdf, amp, w)` | amplitude, width | Gaussian bump on the surface |
| `linear_falloff(sdf, amp, w)` | amplitude, width | Linear falloff at the surface |
| `hard_binarization(sdf, thr)` | threshold | Step function |
| `custom_modification(sdf, fn, params, name)` | function, params | Your own modification |

> The curve-instancing modifications (`curve_instancing` and friends) exist only in the NumPy backend — they are not available in `jax_cores`.

---

## 7. Combining SDFs in the Functional API

```python
from spomso.jax_cores.combine_jax import (
    combine_2_sdfs,
    combine_multiple_sdfs,
    parametric_combine_2_sdfs,
    union2, union, subtract2, intersect2, intersect, add, difference,
    smooth_union2_3o, smooth_union2_2o,
    smooth_subtract2_3o, smooth_subtract2_2o,
    smooth_intersect2_3o, smooth_intersect2_2o,
    smoothmin_poly2, smoothmin_poly3, smoothmax_boltz,
)
```

The `_2o` / `_3o` suffixes select the poly2 or poly3 smoothing kernel; `_3o` is the usual default.

### Non-parametric combination

```python
# combine_2_sdfs(sdf_a, sdf_b, params_a, params_b, operation)
combined = combine_2_sdfs(sdf_1, sdf_2, (radius,), (radius,), union2)

# result is a callable: combined(co)  [no additional params needed]
field = combined(coor)
```

`params_a` and `params_b` are the **fixed parameters** baked into each SDF at combination time. If a parameter is being optimised, pass it through the outer function instead (see section 8).

### n-ary union

```python
# combine_multiple_sdfs(tuple_of_sdfs, tuple_of_params, operation)
combined = combine_multiple_sdfs(
    (sdf_1, sdf_2, sdf_3),
    ((r1,), (r2,), (r3,)),
    union
)
field = combined(coor)
```

### Smooth (parametric) combination

```python
# parametric_combine_2_sdfs(sdf_a, sdf_b, params_a, params_b, operation, smoothing)
smoothed = parametric_combine_2_sdfs(sdf_1, sdf_2, (radius,), (radius,), smooth_union2_3o, 0.3)
field = smoothed(coor)
```

### Chaining combinations

```python
ab  = combine_2_sdfs(sdf_a, sdf_b, (pa,), (pb,), union2)
abc = parametric_combine_2_sdfs(ab, sdf_c, (), (pc,), smooth_union2_3o, smoothing_width)
field = abc(coor)
```

When chaining, the first SDF already has its parameters baked in, so pass `()` as its parameter tuple.

> **Fixed in 1.5.0:** `smoothmin_poly2` and `smoothmin_poly3` disagreed between backends by up to ~2.5e-4 for `a > 0` and more for `a < 0`. JAX now uses a gradient-safe `jnp.where(a == 0, ...)` branch and both backends apply `abs(a)`, so they agree to floating-point precision for all `a` — including `a = 0` and negative `a`. Note that this parity requires 64-bit JAX; with the default float32 the backends differ by ~1e-7 purely from precision. If you calibrated smoothing widths against 1.4.0 output, re-check them.

---

## 8. Computing Gradients with `jacfwd`

`jax.jacfwd` computes the Jacobian with respect to one argument using forward-mode autodiff. For an SDF field this gives the derivative of every point in the field with respect to the chosen parameter.

### The pattern

Wrap your geometry construction in a function where the **geometric parameters are the arguments**:

```python
from jax import jacfwd
from spomso.jax_cores import generate_grid, smarter_reshape, compound_euclidean_transform_sdf
from spomso.jax_cores.sdf_2D_jax import sdf_circle
import jax.numpy as jnp

coor, res = generate_grid((4, 4), (200, 200))


def my_geometry(x0, y0, r):
    """Returns the SDF field for a circle at (x0, y0) with radius r."""
    vec = jnp.asarray([x0, y0, 0.0])
    sdf = compound_euclidean_transform_sdf(sdf_circle, jnp.eye(3), vec, 1.0)
    return sdf(coor, r)


# Evaluate the field
field = my_geometry(1.0, 0.5, 0.8)

# Gradient w.r.t. x0 (argnums=0), y0 (argnums=1), or r (argnums=2)
grad_x0 = jacfwd(my_geometry, argnums=0)(1.0, 0.5, 0.8)   # shape (N,)
grad_r  = jacfwd(my_geometry, argnums=2)(1.0, 0.5, 0.8)
```

The gradient field has the same shape as the SDF field `(N,)`. Reshape and visualise it exactly like any cross-section — remembering to use `res[:2]` on a 2D grid:

```python
grad_2d = smarter_reshape(grad_r, res[:2])
```

### Multi-parameter geometry (3D example)

```python
from jax.scipy.spatial.transform import Rotation
from spomso.jax_cores.sdf_3D_jax import sdf_arc_3d
from spomso.jax_cores.modifications_jax import concentric, elongation, onion
from spomso.jax_cores.combine_jax import (
    combine_2_sdfs, parametric_combine_2_sdfs, union2, smooth_union2_3o,
)
from spomso.jax_cores import compound_euclidean_transform_sdf
import numpy as np

coor, res = generate_grid((6, 6, 6), (100, 100, 100))


def geometry(r, angle_deg, concentric_w, smooth_dist):
    sdf_p = r, 0.0, np.pi * 5 / 6, -np.pi * 5 / 6

    f = concentric(sdf_arc_3d, concentric_w)
    f = elongation(f, jnp.asarray([0.0, 0.0, 0.75]))

    rot_p = Rotation.from_euler('z', angle_deg, degrees=True).as_matrix()
    rot_m = Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix()

    f1 = compound_euclidean_transform_sdf(f, rot_p, jnp.asarray([0, 0, 1.5]), 1.2)
    f2 = compound_euclidean_transform_sdf(f, rot_m, jnp.asarray([0, 0, -1.5]), 1.2)

    combined = combine_2_sdfs(f1, f2, sdf_p, sdf_p, union2)
    combined = parametric_combine_2_sdfs(combined, f, (), sdf_p, smooth_union2_3o, smooth_dist)
    combined = onion(combined, 0.1)

    return combined(coor, ())


p = (2.0, 30.0, 0.5, 1.8)

field = geometry(*p)

# gradient w.r.t. the 4th argument (smooth_dist), argnums=3
grad = jacfwd(geometry, argnums=3)(*p)
```

Note that `angle_deg` here is *evaluated* but not differentiable — see the rotation-matrix pitfall in [section 12](#12-practical-patterns-and-pitfalls).

---

## 9. Chaining Gradients through Post-processing with `jvp`

After computing a gradient with `jacfwd`, you often want to apply a post-processing function to the field **and propagate the gradient through it**. Use `jax.jvp` (Jacobian-Vector Product):

```python
from jax import jvp
from spomso.jax_cores.post_processing_jax import gaussian_falloff_jax

# We already have:
#   field_flat   — the SDF values, shape (N,)
#   grad_flat    — the gradient of the SDF w.r.t. some parameter, shape (N,)

# primals:  (sdf, amplitude, width)
# tangents: (grad_sdf, 0., 0.)  — zeros for parameters we don't differentiate w.r.t.
processed, grad_processed = jvp(
    gaussian_falloff_jax,
    (field_flat, 1.0, 0.5),
    (grad_flat, 0.0, 0.0)
)

# Both have shape (N,) — reshape as usual
from spomso.jax_cores import smarter_reshape

processed_2d = smarter_reshape(processed, res[:2])
grad_processed_2d = smarter_reshape(grad_processed, res[:2])
```

### Available JAX post-processing functions

```python
from spomso.jax_cores.post_processing_jax import (
    hard_binarization_jax,        # step at threshold
    linear_falloff_jax,           # linear ramp at the surface
    relu_jax,                     # one-sided ramp
    smooth_relu_jax,              # smoothed ReLU
    slowstart_jax,                # slow-onset ramp
    gaussian_boundary_jax,        # Gaussian bump on the surface
    gaussian_falloff_jax,         # Gaussian decay from the surface
    sigmoid_falloff_jax,          # sigmoid decay
    positive_sigmoid_falloff_jax, # one-sided sigmoid
    capped_exponential_jax,       # exponential capped at the amplitude
    conv_multiple_jax,            # repeated convolution (needs a grid)
    conv_edge_detection_jax,      # edge detection (needs a grid)
)
```

> **Note on `hard_binarization_jax`:** the step function has zero gradient almost everywhere and undefined gradient at the threshold. Propagating a gradient through it with `jvp` gives zero. Use smooth approximations (`sigmoid_falloff_jax`, `smooth_relu_jax`) when you need differentiable binarisation.

---

## 10. Geometry Optimisation with Optax

`optax` is a gradient-based optimisation library that integrates naturally with JAX. The workflow is:

1. Define a **worker function** mapping parameters → a scalar loss.
2. Use `jax.value_and_grad(worker)(params)` to get loss and gradients in one pass.
3. Feed gradients to an `optax` optimiser to update the parameters.

### Single-object position optimisation

```python
from jax import value_and_grad, config
import jax.numpy as jnp
import optax

from spomso.jax_cores import generate_grid
from spomso.jax_cores.sdf_2D_jax import sdf_circle
from spomso.jax_cores.modifications_jax import gaussian_falloff

config.update("jax_enable_x64", True)

coor, res = generate_grid((8, 8), (200, 200))
radius = 1.0


# Pre-compute the target field (circle at (2.5, -1.0))
def make_field(vec):
    co = jnp.subtract(coor.T, vec).T
    circle = gaussian_falloff(sdf_circle, 1.0, 0.5)
    return circle(co, radius)


target = make_field(jnp.asarray([2.5, -1.0, 0.0]))


# Worker: MSE between current and target
def worker(params):
    field = make_field(params)
    return jnp.sum((field - target) ** 2)


# Initialise optimiser
params = jnp.asarray([0.0, 0.0, 0.0])
optimizer = optax.adam(learning_rate=0.05)
opt_state = optimizer.init(params)

# Optimisation loop
for i in range(500):
    loss, grads = value_and_grad(worker)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    if loss < 1e-6:
        break
    print(f"Step {i + 1:4d}  loss={loss:.6f}  params={params}")

print(f"\nSolution: {params}")
```

Note that this uses `generate_grid` from `spomso.jax_cores`. Had it come from `spomso.cores`, every gradient here would be `NaN` and you would need to nudge the coordinates with `+ 1e-6`. [Section 12](#12-practical-patterns-and-pitfalls) explains why.

### Evaluating the objective on a monitor set, not the whole grid

For many objectives you do not need the field everywhere — only along a line, on a plane, or at a handful of probe points. Building a small coordinate array for the objective and reserving the full grid for final visualisation is dramatically faster:

```python
line_x = jnp.linspace(-3.0, 3.0, 500)
line_coor = jnp.stack([line_x,
                       jnp.full_like(line_x, 0.0),
                       jnp.zeros_like(line_x)])

def worker(p):
    values = my_geometry_on(line_coor, p)
    return -jnp.max(values)
```

The same geometry functions accept any `(3, M)` coordinate array, so nothing else changes.

### Multi-object optimisation

The pattern scales to multiple objects — collect all optimisable parameters into a single array:

```python
# params shape: (3, n_objects) — rows are x, y, z; columns are objects
params = jnp.array([x_init, y_init, jnp.zeros(len(x_init))])

def worker(p):
    sdf = combine_2_sdfs(s_circle, s_circle,
                         (p[:, 0], r0), (p[:, 1], r1),
                         union2)
    field = sdf(coor)
    return jnp.sum((field - target_field) ** 2)
```

### A more robust optimisation loop

The examples in `examples/autodiff/` use a loop with relative-tolerance stopping and best-parameter tracking, which is worth copying:

```python
params = init_params
optimizer = optax.adam(0.1)
opt_state = optimizer.init(params)

prev_value = 1e16
best_value, best_params = 1e16, params

for i in range(max_iterations):
    value, grads = value_and_grad(worker)(params)
    updates, opt_state = optimizer.update(grads, opt_state)

    rtol = jnp.abs(value / prev_value - 1)
    if rtol < 1e-10:
        break
    prev_value = value

    params = optax.apply_updates(params, updates)
    if value < best_value:
        best_value, best_params = value, params.copy()

# fall back to the best parameters seen, not just the last ones
value, params = best_value, best_params
```

### Convergence tips

- Use `jax_enable_x64 = True` — float32 often causes instabilities in SDF geometry.
- The MSE loss works best with smooth fields. Use `gaussian_falloff` or `gaussian_boundary` instead of the raw SDF, especially with sharp geometry.
- Adam with a learning rate of 0.01–0.1 is a reliable starting point.
- Track the relative change in loss and stop when it stalls.
- Choose initial conditions that avoid symmetric local minima. The vector-field example deliberately starts its two particles on opposite sides of the monitor line for exactly this reason.

---

## 11. Differentiable Vector Fields

**New in 1.5.0.** The vector-field layer now has a full JAX counterpart, so vector fields can be optimised the same way as scalar geometry.

```python
from spomso.jax_cores.vector_functions_jax import (
    cartesian_vector_field, spherical_vector_field, cylindrical_vector_field,
    radial_vector_field_spherical, radial_vector_field_cylindrical,
    hyperbolic_vector_field_cylindrical,
    aar_vector_field_cylindrical, awn_vector_field_cylindrical,
    vortex_vector_field_cylindrical, aav_vector_field_cylindrical,
    x_vector_field, y_vector_field, z_vector_field,
    from_sdf,
)
from spomso.jax_cores.vector_modifications_jax import (
    batch_normalize, add_vectors, subtract_vectors, rescale_vectors,
    rotate_vectors_phi, rotate_vectors_theta, rotate_vectors_axis,
    rotate_vectors_x_axis, rotate_vectors_y_axis, rotate_vectors_z_axis,
    revolve_field_x, revolve_field_y, revolve_field_z,
)
```

Unlike the OOP layer there is no `VectorField` class — these are plain functions you compose. Both modules are also re-exported from the `spomso.jax_cores` package root, so `from spomso.jax_cores import vortex_vector_field_cylindrical` works (unlike the transformation functions).

### Worked example: two counter-rotating vortices

This mirrors `examples/autodiff/vector_field_optimization.py`. Two vortex fields with Gaussian envelopes are placed at optimisable positions; the objective maximises the peak x-projection of the total field along a horizontal monitor line.

```python
import numpy as np
from jax import value_and_grad, config
import jax.numpy as jnp
import optax

from spomso.jax_cores import generate_grid, smarter_reshape
from spomso.jax_cores.vector_functions_jax import vortex_vector_field_cylindrical
from spomso.jax_cores.sdf_2D_jax import sdf_circle
from spomso.jax_cores.post_processing_jax import gaussian_boundary_jax

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

co_size, co_resolution = (8, 8), (400, 400)
radius, sigma = 1.0, 2.0

coor, co_res_new = generate_grid(co_size, co_resolution)

# the objective is evaluated on a line, not the whole grid
line_x = jnp.linspace(-3.0, 3.0, 500)
line_coor = jnp.stack([line_x, jnp.zeros_like(line_x), jnp.zeros_like(line_x)])


def vortex(co, x0, y0, handedness):
    """A vortex field attenuated by a Gaussian ring envelope."""
    p = jnp.subtract(co.T, jnp.asarray([x0, y0, 0])).T
    vf = vortex_vector_field_cylindrical(p)
    envelope = gaussian_boundary_jax(sdf_circle(p, radius), 1, sigma)
    return (vf * envelope * handedness)[:2]


def total_field(co, params):
    x1, y1, x2, y2 = params
    return vortex(co, x1, y1, +1) + vortex(co, x2, y2, -1)


def worker(p):
    return -jnp.max(total_field(line_coor, p)[0])


# start the two vortices on opposite sides of the monitor line
params = jnp.asarray([-1.5, 1.2, 1.5, -1.2])
optimizer = optax.adam(0.1)
opt_state = optimizer.init(params)

for i in range(300):
    value, grads = value_and_grad(worker)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

print("final params:", np.asarray(params))

# evaluate the optimised field on the full grid for plotting
field_flat = total_field(coor, params)
magnitude = smarter_reshape(jnp.linalg.norm(field_flat, axis=0), co_resolution)
```

Note `[:2]` on the vortex output: on a 2D problem you usually want a two-component field. Note also that `smarter_reshape` is given the **original** `co_resolution` 2-tuple, not `co_res_new`.

### `from_sdf` component count

`from_sdf(sdf, co_resolution)` returns the normalised gradient of an SDF, with as many components as `co_resolution` has axes:

```python
from_sdf(sdf_circle(coor, 1.0), res[:2]).shape   # (2, N) on a 2D grid
from_sdf(sdf_sphere(coor, 1.0), res).shape       # (3, N) on a 3D grid
```

Passing a three-element resolution when the grid only has `nx · ny` points raises `ValueError` from the internal reshape — the same rule as `smarter_reshape`.

> **Fixed in 1.5.0:** `revolve_field_y` had a one-character typo on the z-component that produced errors up to ~7 on a test grid. `sdf_ngon` in `sdf_2D_jax` was missing its 2D slice, so a 3D coordinate grid leaked the z-component into the radius (errors up to ~1.9 at large |z|). Both are fixed, and all JAX functions now match their NumPy counterparts to floating-point precision.

---

## 12. Practical Patterns and Pitfalls

### NaN gradients at the origin — and which `generate_grid` you use

This is the most common autodiff failure in SPOMSO, and it has a one-line fix.

Radial SDFs are built on `linalg.norm`, whose derivative is undefined at zero. `generate_grid` **forces every axis to an odd number of points**, so a grid point lands at the centre of the domain. Whether that point is *exactly* the origin turns out to depend on which backend built the grid:

```python
from spomso.cores     import generate_grid as np_generate_grid
from spomso.jax_cores import generate_grid as jx_generate_grid

def grad_of_shift(coor):
    def shifted(p):
        return jnp.subtract(jnp.asarray(coor).T, jnp.asarray([p, 0., 0.])).T
    return jacfwd(lambda p: jnp.sum(sdf_circle(shifted(p), 1.5)))(0.0)

coor_np, _ = np_generate_grid((4., 4.), (101, 101))
coor_jx, _ = jx_generate_grid((4., 4.), (101, 101))

grad_of_shift(coor_np)   # nan          <- exactly one point at r == 0
grad_of_shift(coor_jx)   # -0.7071...   <- nearest point is at r ~ 3e-17
```

`jnp.linspace` does not produce a bitwise-exact `0.0` at the midpoint, so the JAX grid misses the singularity by ~1e-17 and the gradient stays finite. This held across every grid size and resolution tested.

**So: use `generate_grid` from `spomso.jax_cores` for autodiff work**, and the problem usually disappears.

Don't treat that as a guarantee, though — it is a floating-point accident rather than a documented contract, and the underlying non-differentiability at `r = 0` is real. Defend against it properly:

1. Prefer the JAX `generate_grid`.
2. Keep `config.update("jax_debug_nans", True)` on during development, so a NaN fails loudly instead of silently poisoning the loss.
3. If you build coordinates yourself, or use the NumPy `generate_grid`, offset them: `co = ... + 1e-6`.
4. Better still, evaluate the objective on a monitor line or probe points that avoid the singular point entirely (see [section 10](#10-geometry-optimisation-with-optax)) — that is faster as well as safer.

### Reshaping when the grid has fewer than three extents

SPOMSO assumes a 3D coordinate space throughout, so `generate_grid` always returns a `(3, N)` point cloud and a **three**-element resolution tuple — even when `size` had two elements. In that case the cloud only holds `nx · ny` points, so the resolution you pass to `smarter_reshape` or `from_sdf` must match the number of extents you actually populated:

```python
coor, co_res_new = generate_grid((8.0, 8.0), (400, 400))
print(co_res_new)                           # (401, 401, 401) — always three

smarter_reshape(field, co_res_new)           # ValueError
smarter_reshape(field, co_res_new[:2])       # correct
smarter_reshape(field, (400, 400))           # also correct
```

Giving the grid a thin z extent instead — `(8.0, 8.0, 0.5)` with `(400, 400, 5)` — sidesteps this entirely and matches the library's 3D-first design.

### Always use `jnp` inside differentiable functions

Inside any function traced by JAX (`jacfwd`, `jvp`, `grad`), use `jax.numpy` for all array operations. Mixing `numpy` and `jax.numpy` inside traced functions can cause silent errors or tracer leaks.

```python
import jax.numpy as jnp   # use everywhere inside geometry functions
import numpy as np        # safe outside (grid setup, plotting, reporting)
```

### Pass parameters as arguments, not closures

JAX differentiates with respect to **function arguments**, not Python variables captured by closure:

```python
# CORRECT — radius is an argument
def geometry(r):
    return sdf_circle(coor, r)

grad = jacfwd(geometry)(1.0)   # works

# INCORRECT — radius is captured by closure
r = 1.0
def geometry():
    return sdf_circle(coor, r)
# jacfwd has nothing to differentiate with respect to
```

### `coor` can be a closure

The coordinate grid is constant — it is fine to capture it from the enclosing scope. Only the **geometric parameters** need to be arguments.

### Rotation matrices from `Rotation.from_euler` are not differentiable

`Rotation.from_euler('z', angle, degrees=True).as_matrix()` is not traceable with respect to `angle`. To differentiate with respect to a rotation angle, build the matrix analytically inside the function:

```python
def geometry(angle_rad):
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    rot = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    sdf = compound_euclidean_transform_sdf(sdf_circle, rot, jnp.zeros(3), 1.0)
    return sdf(coor, radius)

grad = jacfwd(geometry)(np.pi / 4)
```

### Convolution modifications need a grid

`conv_averaging`, `conv_edge_detection`, `conv_multiple_jax`, and `conv_edge_detection_jax` operate on a reshaped grid, not the flat `(N,)` field. The modification versions take the resolution as an argument for this reason.

### Import locations that trip people up

| Symbol | Import |
|---|---|
| `compound_euclidean_transform_sdf`, `move_sdf`, `rotate_sdf`, `scale_sdf` | `spomso.jax_cores` root (1.5.0+) or `spomso.jax_cores.transformations_jax` |
| `generate_grid` | `spomso.jax_cores` (1.5.0+, preferred for autodiff) or `spomso.cores` |
| `smarter_reshape` | either backend root |
| `check_convex_all`, `interior_convex` | `spomso.jax_cores` root (1.5.0+) or `spomso.jax_cores.sdf_2D_jax` |
| JAX vector fields and modifications | `spomso.jax_cores` root, or the specific submodule |
| SDF primitives | the specific submodule (`sdf_2D_jax`, `sdf_3D_jax`) |

In short: on 1.5.0 nearly everything is reachable from the `spomso.jax_cores` root. On 1.4.0 and earlier the transformation functions were only importable from `transformations_jax`.

---

## 13. Quick-Reference Cheat Sheet

### Imports

```python
from jax import jacfwd, jvp, value_and_grad, config
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import optax

# prefer the JAX generate_grid for autodiff (see the NaN pitfall in section 12)
from spomso.jax_cores import (
    generate_grid, smarter_reshape,
    compound_euclidean_transform_sdf, move_sdf, rotate_sdf, scale_sdf,
)

from spomso.jax_cores.sdf_2D_jax import sdf_circle, sdf_box_2d
from spomso.jax_cores.sdf_3D_jax import sdf_sphere, sdf_cylinder, sdf_box, sdf_arc_3d
from spomso.jax_cores.modifications_jax import onion, rounding, mirror, gaussian_falloff
from spomso.jax_cores.combine_jax import (
    combine_2_sdfs, combine_multiple_sdfs, parametric_combine_2_sdfs,
    union2, union, subtract2, intersect2,
    smooth_union2_3o, smooth_subtract2_3o,
)
from spomso.jax_cores.post_processing_jax import (
    gaussian_falloff_jax, gaussian_boundary_jax,
    linear_falloff_jax, sigmoid_falloff_jax, hard_binarization_jax,
)
from spomso.jax_cores.vector_functions_jax import vortex_vector_field_cylindrical
from spomso.jax_cores.vector_modifications_jax import batch_normalize, rescale_vectors
```

### Minimal differentiable geometry workflow

```python
config.update("jax_enable_x64", True)
coor, res = generate_grid((4, 4), (200, 200))

def my_shape(r, x0):
    vec = jnp.asarray([x0, 0.0, 0.0])
    sdf = compound_euclidean_transform_sdf(sdf_circle, jnp.eye(3), vec, 1.0)
    return sdf(coor, r)             # JAX generate_grid -> no NaN-at-origin issue

# Evaluate
field = my_shape(1.0, 0.5)

# Gradient w.r.t. radius
grad_r = jacfwd(my_shape, argnums=0)(1.0, 0.5)

# Propagate the gradient through a post-processing function
processed, grad_processed = jvp(
    gaussian_falloff_jax,
    (field,  1.0, 0.5),
    (grad_r, 0.0, 0.0)
)

# Reshape for plotting — 2D grid, so slice the resolution
field_2d = smarter_reshape(field, res[:2])
```

### Combining SDFs

| Function | Usage | Note |
|---|---|---|
| `combine_2_sdfs(f1, f2, p1, p2, op)` | Two SDFs | Returns a callable |
| `combine_multiple_sdfs((f1, f2, …), (p1, p2, …), op)` | N SDFs | n-ary operation |
| `parametric_combine_2_sdfs(f1, f2, p1, p2, op, width)` | Two SDFs | Smooth blend |

### Operations

| Function | Meaning |
|---|---|
| `union2` / `union` | `min(a, b)` / `min` over many |
| `intersect2` / `intersect` | `max(a, b)` / `max` over many |
| `subtract2` | `max(a, −b)` |
| `add` / `difference` | field addition / subtraction |
| `smooth_union2_3o` / `_2o` | smooth union, poly3 / poly2 |
| `smooth_subtract2_3o` / `_2o` | smooth subtraction |
| `smooth_intersect2_3o` / `_2o` | smooth intersection |

### Optimisation loop template

```python
params = jnp.array([...])
optimizer = optax.adam(0.01)
opt_state = optimizer.init(params)

prev = 1e16
for i in range(max_iter):
    loss, grads = value_and_grad(worker)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    if jnp.abs(loss / prev - 1) < 1e-10:
        break
    prev = loss
    params = optax.apply_updates(params, updates)
```

### Gotcha checklist

- `generate_grid` from `spomso.jax_cores`, not `spomso.cores` — otherwise NaN gradients from radial SDFs
- `jax_enable_x64 = True`, both for stability and for numpy⟷JAX parity
- `jax_debug_nans = True` while developing
- `res[:2]` when reshaping a grid whose `size` had two elements, or just give it a z extent
- `jnp`, never `np`, inside traced functions
- Parameters as arguments, not closures
- Analytic rotation matrices if you differentiate w.r.t. an angle
- `jnp.asarray` around anything you differentiate with respect to
