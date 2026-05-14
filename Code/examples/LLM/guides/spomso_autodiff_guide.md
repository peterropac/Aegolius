# SPOMSO – Automatic Differentiation with JAX
### A practical guide to computing gradients of SDF-based geometry

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
11. [Practical Patterns and Pitfalls](#11-practical-patterns-and-pitfalls)
12. [Quick-Reference Cheat Sheet](#12-quick-reference-cheat-sheet)

---

## 1. Why autodiff?

The standard SPOMSO OOP API (described in the main guide) is built on **NumPy** and is the right tool for evaluating a geometry. However, NumPy is not differentiable — you cannot ask "how does this SDF field change if I nudge parameter X?"

The **JAX-backed functional API** (`spomso.jax_cores`) solves this by reimplementing all SDFs, modifications, and combination operations as pure functions that JAX can trace. This enables:

- Computing the **gradient of a scalar SDF field with respect to any geometric parameter** (radius, position, rotation angle, smoothing width, …)
- **Propagating those gradients through post-processing** functions (Gaussian falloff, binarisation, etc.)
- **Gradient-based optimisation** of geometry parameters using standard optimisers (e.g. `optax`)

---

## 2. Installation

JAX support is an optional extra. Install it alongside SPOMSO:

```bash
pip install -e "path/to/Aegolius/Code/spomso[autodiff]"
# installs jax + jaxlib

# or everything at once:
pip install -e "path/to/Aegolius/Code/spomso[all]"
```

For GPU/TPU, follow the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) to get the right `jaxlib` wheel.

Enable 64-bit precision (highly recommended for geometry work):

```python
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)   # optional but useful during development
```

---

## 3. Two APIs: OOP vs. Functional

| Feature | OOP API (`spomso.cores.*`) | Functional API (`spomso.jax_cores.*`) |
|---|---|---|
| Backend | NumPy | JAX (jit-compiled) |
| Usage style | Method calls on objects | Function composition |
| Transformations | `.move()`, `.rotate()` etc. | `compound_euclidean_transform_sdf(sdf, R, v, s)` |
| Modifications | `.rounding()`, `.onion()` etc. | `rounding(sdf, r)`, `onion(sdf, t)` etc. |
| Combinations | `CombineGeometry("UNION").combine(a, b)` | `combine_2_sdfs(a, b, pa, pb, union2)` |
| Differentiability | ✗ | ✓ via `jax.jacfwd`, `jax.jvp`, `jax.grad` |
| Grid setup | `generate_grid(...)` from `spomso.cores` | Same — grid setup is shared |

The grid setup (`generate_grid`, `smarter_reshape`) is always imported from `spomso.cores.helper_functions` regardless of which API you use.

---

## 4. The JAX Functional API

### Importing primitives

All JAX SDF functions mirror the OOP primitives exactly:

```python
# 2D primitives
from spomso.jax_cores.sdf_2D_jax import (
    sdf_circle, sdf_box_2d, sdf_segment_2d, sdf_triangle_2d,
    sdf_arc, sdf_sector, sdf_ngon,
    sdf_segmented_line_2d, sdf_polygon_2d,
)

# 3D primitives
from spomso.jax_cores.sdf_3D_jax import (
    sdf_sphere, sdf_cylinder, sdf_box, sdf_torus,
    sdf_cone, sdf_segment_3d, sdf_arc_3d,
    sdf_plane, sdf_triangle_3d, sdf_quad_3d,
    sdf_x, sdf_y, sdf_z,
)
```

### Calling a primitive

Every JAX SDF function has the signature `sdf(co, *params)` where `co` is the coordinate cloud `(3, N)` and `*params` are the geometric parameters:

```python
from spomso.cores.helper_functions import generate_grid
from spomso.jax_cores.sdf_2D_jax import sdf_circle

coor, res = generate_grid((4, 4), (200, 200))

radius = 1.0
field = sdf_circle(coor, radius)   # shape (N,)
```

Crucially, `radius` here is a plain Python float or JAX scalar — JAX can differentiate through it.

---

## 5. Euclidean Transformations in the Functional API

Transformations are applied by **wrapping** an SDF function with `compound_euclidean_transform_sdf`:

```python
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

# wrap sdf_circle: translate to (1, 0.5, 0), no rotation, no scaling
vec = jnp.asarray([1.0, 0.5, 0.0])
rot = jnp.eye(3)
scale = 1.0

moved_circle = compound_euclidean_transform_sdf(sdf_circle, rot, vec, scale)

# moved_circle is now a callable: moved_circle(co, *params)
field = moved_circle(coor, radius)
```

To rotate, build a rotation matrix using `jax.scipy.spatial.transform.Rotation`:

```python
rot_mat = Rotation.from_euler('z', 45, degrees=True).as_matrix()
rotated_circle = compound_euclidean_transform_sdf(sdf_circle, rot_mat, jnp.zeros(3), 1.0)
```

> **Key pattern:** `compound_euclidean_transform_sdf` returns a **new SDF function**. It does not evaluate the field — it produces a composed callable that you evaluate later.

---

## 6. Modifications in the Functional API

Modifications work the same way — each one takes an SDF function and returns a new SDF function:

```python
from spomso.jax_cores.modifications_jax import (
    onion, rounding, rounding_cs, elongation,
    mirror, infinite_repetition, finite_repetition,
    twist, bend, extrusion, revolution,
    gaussian_falloff, linear_falloff,
    hard_binarization,
)
import jax.numpy as jnp

# Start with the base SDF
sdf = sdf_circle

# Apply onion (hollow shell of thickness 0.2)
sdf = onion(sdf, 0.2)

# Apply mirror (mirror image at (-d/2, 0, 0), original at (d/2, 0, 0))
sdf = mirror(sdf, [-d/2, 0, 0], [d/2, 0, 0])

# Apply a Euclidean transform
sdf = compound_euclidean_transform_sdf(sdf, rot_mat, vec, scale)

# Evaluate — all modifications + transforms are applied here
field = sdf(coor, radius)   # radius is passed to the base sdf_circle
```

Every modification in the functional API takes `(sdf_function, *modification_params)` and returns a new callable. The chain is built lazily and evaluated in one pass when you call the final function.

### Available modifications

| Function | Arguments | Effect |
|---|---|---|
| `onion(sdf, t)` | thickness | Shell of thickness t |
| `rounding(sdf, r)` | radius | Round edges (thickens) |
| `rounding_cs(sdf, r, bb)` | radius, bounding-box size | Round without thickening |
| `elongation(sdf, v)` | vector | Stretch along v |
| `mirror(sdf, a, b)` | point a, point b | Mirror; copy at a, original at b |
| `twist(sdf, pitch)` | pitch (rad/unit) | Twist around z |
| `bend(sdf, r, angle)` | radius, angle | Bend around z |
| `extrusion(sdf, h)` | height | Extrude 2D to 3D |
| `revolution(sdf, r)` | radius | Revolve 2D around y-axis |
| `infinite_repetition(sdf, d)` | spacing vector | Tile infinitely |
| `finite_repetition(sdf, size, reps)` | size, repetitions | Tile within box |
| `gaussian_falloff(sdf, amp, w)` | amplitude, width | Gaussian falloff at surface |
| `linear_falloff(sdf, amp, w)` | amplitude, width | Linear falloff at surface |
| `hard_binarization(sdf, thr)` | threshold | Step function |

---

## 7. Combining SDFs in the Functional API

```python
from spomso.jax_cores.combine_jax import (
    combine_2_sdfs,
    combine_multiple_sdfs,
    parametric_combine_2_sdfs,
    union2, union, subtract2, intersect2, intersect,
    smooth_union2_3o, smooth_union2_2o,
    smooth_subtract2_3o, smooth_intersect2_3o,
)
```

### Non-parametric combination

```python
# combine_2_sdfs(sdf_a, sdf_b, params_a, params_b, operation)
combined = combine_2_sdfs(sdf_1, sdf_2, (radius,), (radius,), union2)

# result is a callable: combined(co)  [no additional params needed]
field = combined(coor)
```

Note that `params_a` and `params_b` are the **fixed parameters** baked into each SDF at combination time. If a parameter is being optimised, pass it through the outer function instead (see section 8).

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

When chaining, the first SDF in the chain already has its parameters baked in, so pass `()` as its parameter tuple.

---

## 8. Computing Gradients with `jacfwd`

`jax.jacfwd` computes the Jacobian of a function with respect to one of its arguments using forward-mode automatic differentiation. For scalar-output geometry functions, this gives you the gradient of every point in the SDF field with respect to the chosen parameter.

### The pattern

Wrap your geometry construction in a Python function where the **geometric parameters are the arguments**:

```python
from jax import jacfwd
from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.jax_cores.sdf_2D_jax import sdf_circle
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf
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
grad_x0 = jacfwd(my_geometry, argnums=0)(1.0, 0.5, 0.8)  # shape (N,)
grad_r   = jacfwd(my_geometry, argnums=2)(1.0, 0.5, 0.8)
```

The gradient field has the same shape as the original SDF field `(N,)`. Reshape and visualise it exactly like any SDF cross-section.

### Multi-parameter geometry (3D example)

```python
from jax.scipy.spatial.transform import Rotation
from spomso.jax_cores.sdf_3D_jax import sdf_arc_3d
from spomso.jax_cores.modifications_jax import concentric, elongation, onion
from spomso.jax_cores.combine_jax import combine_2_sdfs, parametric_combine_2_sdfs, union2, smooth_union2_3o
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf

coor, res = generate_grid((6, 6, 6), (100, 100, 100))

def geometry(r, angle_deg, concentric_w, smooth_dist):
    sdf_p = r, 0.0, np.pi*5/6, -np.pi*5/6

    f = concentric(sdf_arc_3d, concentric_w)
    f = elongation(f, jnp.asarray([0.0, 0.0, 0.75]))

    rot_p = Rotation.from_euler('z',  angle_deg, degrees=True).as_matrix()
    rot_m = Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix()

    f1 = compound_euclidean_transform_sdf(f, rot_p, jnp.asarray([0, 0,  1.5]), 1.2)
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

---

## 9. Chaining Gradients through Post-processing with `jvp`

After computing a gradient with `jacfwd`, you often want to apply a post-processing function (Gaussian falloff, ReLU, etc.) to the field **and propagate the gradient through it**. Use `jax.jvp` (Jacobian-Vector Product):

```python
from jax import jvp
from spomso.jax_cores.post_processing_jax import gaussian_falloff_jax, linear_falloff_jax

# We already have:
#   field_flat   — the SDF values, shape (N,)
#   grad_flat    — the gradient of the SDF w.r.t. some parameter, shape (N,)

# Apply gaussian_falloff AND propagate the gradient through it
# primals: (sdf, amplitude, width)
# tangents: (grad_sdf, 0., 0.)  — zeroes for params we don't differentiate w.r.t.
processed, grad_processed = jvp(
    gaussian_falloff_jax,
    (field_flat, 1.0, 0.5),
    (grad_flat,  0.0, 0.0)
)

# Both processed and grad_processed have shape (N,) — reshape as usual
from spomso.cores.helper_functions import smarter_reshape
processed_3d    = smarter_reshape(processed,      co_resolution)
grad_processed_3d = smarter_reshape(grad_processed, co_resolution)
```

### Available JAX post-processing functions

```python
from spomso.jax_cores.post_processing_jax import (
    hard_binarization_jax,     # step at threshold
    linear_falloff_jax,        # linear ramp at surface
    relu_jax,                  # one-sided ramp
    gaussian_boundary_jax,     # Gaussian bump at surface
    gaussian_falloff_jax,      # Gaussian decay from surface
    sigmoid_falloff_jax,       # sigmoid decay
    capped_exponential_jax,    # exponential capped at 1
)
```

> **Note on `hard_binarization_jax`:** The step function has zero gradient almost everywhere and undefined gradient at the threshold. When you propagate a gradient through it with `jvp`, you will get a zero result. Use smooth approximations (`sigmoid_falloff_jax`, `smooth_relu_jax`) when you need differentiable binarisation.

---

## 10. Geometry Optimisation with Optax

`optax` is a gradient-based optimisation library that integrates naturally with JAX. The workflow is:

1. Define a **worker function** that maps parameters → a scalar loss.
2. Use `jax.value_and_grad(worker)(params)` to get loss and gradients in one pass.
3. Feed gradients to an `optax` optimiser to update the parameters.

### Single-object position optimisation

```python
from jax import value_and_grad, config
import jax.numpy as jnp
import optax

from spomso.jax_cores.sdf_2D_jax import sdf_circle
from spomso.jax_cores.modifications_jax import gaussian_falloff
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf

config.update("jax_enable_x64", True)

coor, res = generate_grid((8, 8), (200, 200))
radius = 1.0

# Pre-compute the target field (circle at (2.5, -1.0))
def make_field(vec):
    co = jnp.subtract(coor.T, vec).T + 1e-6
    circle = gaussian_falloff(sdf_circle, 1.0, 0.5)
    return circle(co, radius)

target = make_field(jnp.asarray([2.5, -1.0, 0.0]))

# Worker: MSE between current and target
def worker(params):
    field = make_field(params)
    return jnp.sum((field - target)**2)

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
    print(f"Step {i+1:4d}  loss={loss:.6f}  params={params}")

print(f"\nSolution: {params}")
```

### Multi-object optimisation

The same pattern scales to multiple objects — collect all optimisable parameters into a single `params` array:

```python
# params shape: (3, n_objects) — rows are x, y, z; columns are objects
params = jnp.array([x_init, y_init, jnp.zeros(len(x_init))])

def worker(p):
    # build union of circles at each position in p
    sdf = combine_2_sdfs(s_circle, s_circle,
                         (p[:, 0], r0), (p[:, 1], r1),
                         union2)
    field = sdf(coor)
    return jnp.sum((field - target_field)**2)

# Same loop as above
```

### Convergence tips

- Use `jax_enable_x64 = True` — float32 often causes instabilities in SDF geometry.
- The MSE loss works best with smooth fields. Use `gaussian_falloff` instead of the raw SDF, especially when the geometry has sharp boundaries.
- Adam with a learning rate of 0.01–0.05 is a reliable starting point.
- Track the relative change in loss: if `|loss_prev/loss - 1| < 1e-6`, stop early.

---

## 11. Practical Patterns and Pitfalls

### Always use `jnp` inside differentiable functions

Inside any function traced by JAX (`jacfwd`, `jvp`, `grad`), use `jax.numpy` for all array operations. Mixing `numpy` and `jax.numpy` inside traced functions can cause silent errors or tracer leaks.

```python
import jax.numpy as jnp  # use everywhere inside geometry functions
import numpy as np        # safe to use outside (for grid setup, plotting)
```

### Pass parameters as arguments, not closures

JAX differentiates with respect to **function arguments**, not Python variables captured by closure. Structure your geometry function so that every parameter you want to differentiate appears as an argument:

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

### `coor` can be a closure (it's not a parameter)

The coordinate grid is constant — it's fine to capture it from the enclosing scope. Only the **geometric parameters** need to be arguments.

### Rotation matrices are not directly differentiable parameters

`Rotation.from_euler('z', angle, degrees=True).as_matrix()` is not itself JAX-traceable. If you want to differentiate with respect to a rotation angle, compute the rotation matrix analytically inside the function:

```python
def geometry(angle_rad):
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    rot = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    sdf = compound_euclidean_transform_sdf(sdf_circle, rot, jnp.zeros(3), 1.0)
    return sdf(coor, radius)

grad = jacfwd(geometry)(np.pi/4)
```

### Shapes and `smarter_reshape`

The `smarter_reshape` from `spomso.cores.helper_functions` works on both NumPy and JAX arrays. The gradient field has the same shape as the SDF field — reshape and visualise identically.

---

## 12. Quick-Reference Cheat Sheet

### Imports

```python
from jax import jacfwd, jvp, value_and_grad, config
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import optax

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.jax_cores.sdf_2D_jax import sdf_circle, sdf_box_2d
from spomso.jax_cores.sdf_3D_jax import sdf_sphere, sdf_cylinder, sdf_box, sdf_arc_3d
from spomso.jax_cores.transformations_jax import compound_euclidean_transform_sdf
from spomso.jax_cores.modifications_jax import onion, rounding, mirror, gaussian_falloff
from spomso.jax_cores.combine_jax import (
    combine_2_sdfs, combine_multiple_sdfs, parametric_combine_2_sdfs,
    union2, union, subtract2, intersect2,
    smooth_union2_3o, smooth_subtract2_3o,
)
from spomso.jax_cores.post_processing_jax import (
    gaussian_falloff_jax, linear_falloff_jax, sigmoid_falloff_jax,
    hard_binarization_jax,
)
```

### Minimal differentiable geometry workflow

```python
config.update("jax_enable_x64", True)
coor, res = generate_grid((4, 4), (200, 200))

def my_shape(r, x0):
    vec = jnp.asarray([x0, 0.0, 0.0])
    sdf = compound_euclidean_transform_sdf(sdf_circle, jnp.eye(3), vec, 1.0)
    return sdf(coor, r)

# Evaluate
field = my_shape(1.0, 0.5)

# Gradient w.r.t. radius
grad_r = jacfwd(my_shape, argnums=0)(1.0, 0.5)

# Propagate gradient through a post-processing function
processed, grad_processed = jvp(
    gaussian_falloff_jax,
    (field,  1.0, 0.5),
    (grad_r, 0.0, 0.0)
)
```

### Combining SDFs

| Function | Usage | Note |
|---|---|---|
| `combine_2_sdfs(f1, f2, p1, p2, op)` | Two SDFs | Returns a callable |
| `combine_multiple_sdfs((f1, f2, ...), (p1, p2, ...), op)` | N SDFs | n-ary operation |
| `parametric_combine_2_sdfs(f1, f2, p1, p2, op, width)` | Two SDFs | Smooth blend |

### Operations

| String/function | Meaning |
|---|---|
| `union2` | min(a, b) |
| `union` | min(a, b, …) |
| `subtract2` | max(a, −b) |
| `intersect2` | max(a, b) |
| `smooth_union2_3o` | smooth union (poly3) |
| `smooth_subtract2_3o` | smooth subtraction |
| `smooth_intersect2_3o` | smooth intersection |

### Optimisation loop template

```python
params = jnp.array([...])
optimizer = optax.adam(0.01)
opt_state = optimizer.init(params)

for i in range(max_iter):
    loss, grads = value_and_grad(worker)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if loss < tol:
        break
```
