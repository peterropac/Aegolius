# SPOMSO – Vector Fields
### A practical guide to generating and manipulating vector fields

**Applies to SPOMSO 1.5.0.**

---

## Table of Contents
1. [What are vector fields in SPOMSO?](#1-what-are-vector-fields-in-spomso)
2. [The VectorField Base Class](#2-the-vectorfield-base-class)
3. [Built-in Vector Fields](#3-built-in-vector-fields)
4. [Defining Fields by Components](#4-defining-fields-by-components)
5. [Vector Field Modifications](#5-vector-field-modifications)
6. [Deriving a Vector Field from an SDF](#6-deriving-a-vector-field-from-an-sdf)
7. [Revolving a Vector Field into 3D](#7-revolving-a-vector-field-into-3d)
8. [Custom Vector Fields](#8-custom-vector-fields)
9. [Extracting Components and Angles](#9-extracting-components-and-angles)
10. [Visualising Vector Fields](#10-visualising-vector-fields)
11. [Special Fields: Liquid Crystal Waveguides](#11-special-fields-liquid-crystal-waveguides)
12. [Differentiable Vector Fields (JAX)](#12-differentiable-vector-fields-jax)
13. [Common Pitfalls](#13-common-pitfalls)
14. [Quick-Reference Cheat Sheet](#14-quick-reference-cheat-sheet)

---

## 1. What are vector fields in SPOMSO?

While the SDF subsystem assigns one scalar value per grid point (the signed distance to the nearest surface), the **vector field** subsystem assigns a vector per grid point. This enables applications like:

- director fields in liquid crystals
- flow fields around geometry
- surface normal maps
- any physically motivated orientation field derived from geometry

All vector fields share the same coordinate grid as scalar SDFs — set up once with `generate_grid` and reused everywhere. As with SDFs, SPOMSO's space is always three-dimensional: `generate_grid` returns a `(3, N)` point cloud and a three-element resolution whatever you pass it, and vectors are three-component by default.

**Additional imports for vector fields:**

```python
from spomso.cores import generate_grid, smarter_reshape
from spomso.cores import vector_smarter_reshape, nd_vector_smarter_reshape
```

`vector_smarter_reshape` converts a flat `(3, N)` field into a `(3, nx, ny, nz)` grid. `nd_vector_smarter_reshape` does the same for a field with any number of components, e.g. the `(2, N)` output you get on a 2D grid.

---

## 2. The VectorField Base Class

Every vector field object is an instance of `VectorField`. You rarely instantiate it directly — you use a built-in subclass or the `VectorField(my_function, *params)` constructor for custom fields.

```python
from spomso.cores import VectorField          # 1.5.0+
# from spomso.cores.geom import VectorField   # also works, any version
```

> `VectorField` is re-exported from `spomso.cores` as of 1.5.0. On earlier versions only the `spomso.cores.geom` path works.

### Evaluating a field

```python
field = vf.create(p)   # shape (3, N) — one vector per grid point
```

The input `p` is either the coordinate cloud `coor` (shape `(3, N)`) **or** a pre-computed SDF array (shape `(N,)`) **or** a component array you built yourself, depending on the type of field:

| Field type | What `.create()` expects |
|---|---|
| Built-in geometric fields (radial, vortex, …) | `coor`, shape `(3, N)` |
| `CartesianVectorField`, `CylindricalVectorField`, `SphericalVectorField` | a component array you built, shape `(3, N)` |
| `VectorFieldFromSDF` | an SDF array, shape `(N,)` |
| `LCWG2D` | a waveguide SDF array, shape `(N,)` |
| `LCWG3Dm1` / `LCWG3Dp1` | a tuple of two SDF arrays |

### Reshaping for visualisation

```python
from spomso.cores import vector_smarter_reshape

field_3d = vector_smarter_reshape(field, co_res_new)
# shape (3, nx, ny, nz)
# field_3d[0] = x-components, field_3d[1] = y-components, field_3d[2] = z-components
```

On a **2D** grid, slice the resolution to two elements — `co_res_new[:2]` — because `generate_grid` always returns a three-element tuple. See [Common Pitfalls](#13-common-pitfalls).

---

## 3. Built-in Vector Fields

All built-in fields live in `spomso.cores.geom_vector` and are re-exported from `spomso.cores`. They take the coordinate cloud `coor` as input to `.create()`.

```python
from spomso.cores import (
    RadialSphericalVectorField,
    RadialCylindricalVectorField,
    VortexCylindricalVectorField,
    HyperbolicCylindricalVectorField,
    WindingCylindricalVectorField,
    AngledRadialCylindricalVectorField,
    AngledVortexCylindricalVectorField,
    XVectorField, YVectorField, ZVectorField,
    VectorFieldFromSDF,
)
```

### Catalogue of built-in fields

| Class | Constructor | Description |
|---|---|---|
| `RadialSphericalVectorField` | `()` | Vectors point radially outward from the origin in 3D |
| `RadialCylindricalVectorField` | `()` | Radial outward from the z-axis; z-component = 0 |
| `VortexCylindricalVectorField` | `()` | Tangent to circles around the z-axis — a curl field |
| `HyperbolicCylindricalVectorField` | `()` | Hyperbolic saddle field centred on the z-axis |
| `WindingCylindricalVectorField` | `(gamma)` | In-plane direction winds with winding number `gamma` (may be fractional or negative) |
| `AngledRadialCylindricalVectorField` | `(alpha)` | Radial field rotated by `alpha` radians in the plane |
| `AngledVortexCylindricalVectorField` | `(alpha)` | Vortex field rotated by `alpha` radians in the plane |
| `XVectorField` | `()` | Uniform field pointing along +x |
| `YVectorField` | `()` | Uniform field pointing along +y |
| `ZVectorField` | `()` | Uniform field pointing along +z |
| `VectorFieldFromSDF` | `(grid_resolution)` | Normalised gradient of an SDF |

> **Fixed in 1.5.0:** `HyperbolicCylindricalVectorField` and `WindingCylindricalVectorField` were **unusable** in earlier versions. Their underlying functions called `cylindrical_define(1, alpha, zeros)` with three positional arguments, but `cylindrical_define` takes only one. Both now work. If you tried these in 1.3.0 or 1.4.0 and gave up, try again.

### Minimal example

```python
import numpy as np
from spomso.cores import generate_grid, vector_smarter_reshape
from spomso.cores import RadialSphericalVectorField

coor, res = generate_grid((4, 4, 4), (50, 50, 50))   # res is (51, 51, 51)

vf = RadialSphericalVectorField()
field = vf.create(coor)                       # shape (3, N)
field_3d = vector_smarter_reshape(field, res) # shape (3, 51, 51, 51)
```

---

## 4. Defining Fields by Components

Use the coordinate-system classes when you want to specify each component explicitly from a formula.

```python
from spomso.cores import (
    CartesianVectorField,
    CylindricalVectorField,
    SphericalVectorField,
)
```

The workflow is:

1. Build a component array of shape `(3, N)` using any numpy operations on `coor`.
2. Instantiate the appropriate class.
3. Optionally apply modifications.
4. Call `.create(components)` — passing the **components**, not `coor`.

### Cartesian components — rows are `(x, y, z)`

```python
u = np.ones(coor.shape[1])           # x-component: constant 1
v = np.zeros(coor.shape[1])          # y-component: 0
w = coor[2]                          # z-component: varies with z

components = np.asarray((u, v, w))
vf = CartesianVectorField()
field = vf.create(components)
```

### Cylindrical components — rows are `(r, phi, z)`

The class converts to Cartesian automatically: `x = r·cos(phi)`, `y = r·sin(phi)`, `z = z`.

```python
r   = np.ones(coor.shape[1])
phi = np.arctan2(coor[1], coor[0])   # azimuthal angle at each point
z   = np.zeros(coor.shape[1])

components = np.asarray((r, phi, z))
vf = CylindricalVectorField()
field = vf.create(components)
```

### Spherical components — rows are `(r, phi, theta)`

Converts as `x = r·cos(phi)·sin(theta)`, `y = r·sin(phi)·sin(theta)`, `z = r·cos(theta)`.

```python
r     = np.ones(coor.shape[1])
phi   = np.arctan2(coor[1], coor[0])
theta = np.arccos(coor[2] / (np.linalg.norm(coor, axis=0) + 1e-12))

components = np.asarray((r, phi, theta))
vf = SphericalVectorField()
field = vf.create(components)
```

> **Note:** the output of all three classes is always in Cartesian `(x, y, z)`, regardless of which coordinate system you used for the input. Note also that `phi` and `theta` here are *vector* angles, not positions — the example above happens to derive them from position, but they can be any per-point arrays.

---

## 5. Vector Field Modifications

All `VectorField` objects inherit from `ModifyVectorObject`. Modifications chain and are applied in call order when you later call `.create()`.

### Rotation

```python
vf.rotate_phi(np.pi/4)          # rotate every vector about the z-axis
vf.rotate_theta(-np.pi/6)       # rotate toward/away from the z-axis
vf.rotate_x(alpha)              # rotate about the x-axis
vf.rotate_y(alpha)              # rotate about the y-axis
vf.rotate_z(alpha)              # rotate about the z-axis
vf.rotate_axis((1, 0, 0), alpha)  # rotate about an arbitrary axis
```

When `alpha` is an **array of shape `(N,)`**, each grid point gets its own rotation angle — this is how spatially varying rotations are applied (e.g. twisting a field around a curve).

Note the argument order: `rotate_axis(axis, alpha)` takes the axis first.

### Scaling

```python
vf.rescale(2.0)           # multiply all vector lengths by 2
vf.rescale(scale_array)   # per-point scaling with an array of shape (N,)
```

### Adding / subtracting another field

```python
second_field = another_vf.create(components)
vf.add(second_field)       # element-wise vector addition
vf.subtract(second_field)  # element-wise vector subtraction
```

> **Fixed in 1.4.0:** `subtract()` used to perform *addition*. Any 1.3.0 code that relied on the buggy behaviour will now produce different (correct) results.

### Normalising

```python
vf.normalize()   # set all vectors to unit length
```

Zero-length vectors are left as zeros rather than producing NaN.

### Inspection

```python
print(vf.modifications)   # list of applied modification names
print(vf)                 # repr: constructor args + modification chain (new in 1.5.0)
```

### Full example with chained modifications

```python
from spomso.cores import SphericalVectorField

co_size = (4.0, 4.0, 4.0)

r_     = np.ones(coor.shape[1])
phi_   = np.pi * coor[0] / (co_size[0] / 2)
theta_ = np.pi * coor[2] / co_size[2] + np.pi / 2

components = np.asarray((r_, phi_, theta_))
vf = SphericalVectorField()

# rotate every vector 90° about the z-axis
vf.rotate_phi(np.pi / 2)

# tilt every vector toward the z-axis
vf.rotate_theta(-np.pi / 4)

# scale vectors by a value that increases away from z = 0
scale = np.abs(coor[2] / co_size[2]) + 0.1
vf.rescale(scale)

# add a second field and normalise
second = SphericalVectorField()
second_field = second.create(np.asarray((r_, phi_ * 0, -theta_)))
vf.add(second_field)
vf.normalize()

field = vf.create(components)
```

---

## 6. Deriving a Vector Field from an SDF

`VectorFieldFromSDF` computes the **normalised gradient** of any SDF. This gives a field pointing perpendicularly away from the SDF's zero level-set — the surface normals of the encoded geometry.

`VectorFieldFromSDF` is a fully three-dimensional tool — it computes the gradient of an SDF on whatever grid you give it, and in normal 3D use it returns a `(3, N)` field that supports every modification:

```python
import numpy as np
from spomso.cores import generate_grid, VectorFieldFromSDF, Sphere
from spomso.cores.post_processing import linear_falloff

coor, co_res_new = generate_grid((4.0, 4.0, 4.0), (100, 100, 100))

sdf_values = Sphere(1.0).create(coor)         # shape (N,)

vf = VectorFieldFromSDF(co_res_new)

# spatially varying rotation driven by distance to the surface
phi_rotation = linear_falloff(sdf_values, np.pi / 2, 0.5)
vf.rotate_z(phi_rotation)

field = vf.create(sdf_values)                 # (3, N)
```

> **Key distinction:** `VectorFieldFromSDF.create()` takes the **SDF array** `(N,)`, not the coordinate cloud. The field is the numerical gradient of the SDF on the grid (`np.gradient` internally), which is why the resolution is required at construction time.

### Component count follows the resolution you pass

The number of returned components is inferred from `grid_resolution`:

```python
VectorFieldFromSDF(co_res_new).create(sdf).shape      # (3, N) — the normal case
VectorFieldFromSDF(co_res_new[:2]).create(sdf).shape  # (2, N) — 2D-extent grid
```

The two-component case only arises if you deliberately build a grid with a two-element `size` and then slice the resolution to match. It is worth knowing about, because **all rotation and revolution modifications index `vec[2]`** and therefore raise `IndexError` on a two-component field:

| Modification | Works on `(2, N)`? |
|---|---|
| `normalize`, `rescale`, `add`, `subtract` | yes |
| `rotate_phi`, `rotate_theta`, `rotate_x`, `rotate_y`, `rotate_z`, `rotate_axis` | **no** — `IndexError` |
| `revolution_x`, `revolution_y`, `revolution_z` | **no** — `IndexError` |

The simplest fix is to work in 3D space, which is the library's default assumption anyway. For a problem that is essentially planar, give the grid a thin z extent rather than dropping to two dimensions — this is what the shipped examples in `examples/vector/` do:

```python
# planar in spirit, three-dimensional in practice
coor, co_res_new = generate_grid((100.0, 100.0, 5.5), (100, 100, 11))
vf = VectorFieldFromSDF(co_res_new)     # -> (3, N), all modifications available
```

If you genuinely want a two-component field, `nd_vector_smarter_reshape` will grid it as-is; just stick to `normalize`, `rescale`, `add`, and `subtract`.

### Smoothing artefacts

The gradient is smoothest when the SDF is smooth. If you see artefacts, smooth the SDF first — note that `conv_averaging` needs a **grid**, not a flat array:

```python
from spomso.cores.post_processing import conv_averaging
from spomso.cores import smarter_reshape

sdf_3d = smarter_reshape(sdf_values, co_res_new)
sdf_smooth = conv_averaging(sdf_3d, (5, 5, 1), iterations=1)
sdf_values_smooth = sdf_smooth.reshape(-1)

vf = VectorFieldFromSDF(co_res_new)
field = vf.create(sdf_values_smooth)
```

---

## 7. Revolving a Vector Field into 3D

A 2D vector field can be revolved around one of the coordinate axes to produce a full 3D field. This is the vector analogue of the SDF `.revolution()` modification.

```python
vf = VectorFieldFromSDF(co_res_new)

# revolve the field about the z-axis
vf.revolution_z(coor)   # coor is needed to compute the angle at each point

# also available:
# vf.revolution_x(coor)
# vf.revolution_y(coor)

field = vf.create(sdf_values)
```

The revolution methods rotate each vector by the azimuthal or polar angle of its grid point, creating a rotationally symmetric 3D field.

> **Fixed in 1.5.0:** the JAX version of `revolve_field_y` had a one-character typo on the z-component (`vec[1] * sa` instead of `vec[0] * sa`), causing errors up to ~7 on a test grid. All three `revolve_field_*` functions now agree between the NumPy and JAX backends to floating-point precision — provided you enable 64-bit JAX (`config.update("jax_enable_x64", True)`). With JAX's default float32 the backends differ by ~2e-7 purely from precision.

---

## 8. Custom Vector Fields

Wrap any function `f(p, *params) -> np.ndarray` of shape `(3, N)` using `VectorField` directly:

```python
from spomso.cores import VectorField, batch_normalize, smarter_reshape


def custom_radial_vf(co, order):
    """Radial field based on the L^order norm."""
    u = np.linalg.norm(co, axis=0, ord=order)
    u_3d = smarter_reshape(u, co_res_new)
    grad = np.asarray(np.gradient(u_3d)).reshape(3, -1)
    return batch_normalize(grad)


vf = VectorField(custom_radial_vf, 3)   # order = 3
field = vf.create(coor)
```

The function signature must be `f(p, *params)` where `p` is the input (either `coor`, an SDF array, or a component array) and `*params` are any additional parameters passed after the function in the constructor.

### Low-level transformation utilities

The functions in `spomso.cores.vector_modification_functions` operate directly on `(3, N)` arrays and can be called inside custom field functions:

```python
from spomso.cores import (
    batch_normalize,        # normalise each column to unit length
    add_vectors,            # add two fields
    subtract_vectors,       # subtract two fields
    rescale_vectors,        # scale by a scalar or per-point array
    rotate_vectors_phi,     # rotate by azimuthal angle(s)
    rotate_vectors_theta,   # rotate by polar angle(s)
    rotate_vectors_x_axis,  # rotate about the x-axis
    rotate_vectors_y_axis,  # rotate about the y-axis
    rotate_vectors_z_axis,  # rotate about the z-axis
    rotate_vectors_axis,    # rotate about an arbitrary axis
    revolve_field_x,        # revolve about the x-axis
    revolve_field_y,        # revolve about the y-axis
    revolve_field_z,        # revolve about the z-axis
)
```

All rotation functions accept either a scalar or an array of per-point angles.

---

## 9. Extracting Components and Angles

You can extract individual scalar maps via convenience methods — all accepting the same input `p` you would pass to `.create()`:

```python
x      = vf.x(p)       # x-component, shape (N,)
y      = vf.y(p)       # y-component
z      = vf.z(p)       # z-component
phi    = vf.phi(p)     # azimuthal angle atan2(y, x), range [-π, π]
theta  = vf.theta(p)   # polar angle acos(z), range [0, π]
length = vf.length(p)  # vector magnitude |v|
```

> **Fixed in 1.5.0:** `.theta()` used to compute `arccos(z)` on the raw z-component, which was only correct for a normalised field. It now divides by the magnitude — `arccos(z / |v|)` — so it is correct for any field, and returns 0 rather than `NaN` for zero-length vectors. If you were calling `.normalize()` purely to make `.theta()` behave, you no longer need to.

Reshape each to a grid with `smarter_reshape`:

```python
phi_3d = smarter_reshape(phi, co_res_new)   # shape (nx, ny, nz)
```

---

## 10. Visualising Vector Fields

### 2D quiver plot (matplotlib)

The standard pattern is to decimate the field (show every `N`-th vector) to avoid overplotting:

```python
import matplotlib.pyplot as plt
from spomso.cores import smarter_reshape, vector_smarter_reshape

field_3d = vector_smarter_reshape(field, co_res_new)   # (3, nx, ny, nz)
x_3d = smarter_reshape(vf.x(coor), co_res_new)

decimate = 5
depth = co_res_new[2] // 2   # mid-plane along z

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(x_3d[:, :, depth].T, cmap="bwr", origin="lower",
          extent=(-co_size[0] / 2, co_size[0] / 2,
                  -co_size[1] / 2, co_size[1] / 2))
ax.quiver(
    smarter_reshape(coor[0], co_res_new)[::decimate, ::decimate, depth],
    smarter_reshape(coor[1], co_res_new)[::decimate, ::decimate, depth],
    field_3d[0, ::decimate, ::decimate, depth],
    field_3d[1, ::decimate, ::decimate, depth],
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
plt.show()
```

### 3D cone plot (Plotly)

```python
import plotly.graph_objects as go

decimate = 4
xx = smarter_reshape(coor[0], co_res_new)[::decimate, ::decimate, :].flatten()
yy = smarter_reshape(coor[1], co_res_new)[::decimate, ::decimate, :].flatten()
zz = smarter_reshape(coor[2], co_res_new)[::decimate, ::decimate, :].flatten()
vx = field_3d[0, ::decimate, ::decimate, :].flatten()
vy = field_3d[1, ::decimate, ::decimate, :].flatten()
vz = field_3d[2, ::decimate, ::decimate, :].flatten()

fig = go.Figure(data=go.Cone(
    x=xx, y=yy, z=zz,
    u=vx, v=vy, w=vz,
    colorscale="Blues",
    sizemode="absolute",
    sizeref=0.5
))
fig.show()
```

### Colour-mapping a scalar component

Any of the scalar maps (`x`, `y`, `z`, `phi`, `theta`, `length`) can be visualised with `imshow` exactly like an SDF cross-section. Use a diverging colormap (`"bwr"`) for signed quantities like `x`, `y`, `z`, and `phi`, and a sequential one (`"viridis"`) for unsigned quantities like `length` and `theta`.

---

## 11. Special Fields: Liquid Crystal Waveguides

`spomso.cores.geom_vector_special` provides three specialised vector fields for liquid crystal director patterns inside waveguides. These take an **SDF array** (the distance field of the waveguide geometry) as their primary input, not `coor`.

```python
from spomso.cores.geom_vector_special import LCWG2D, LCWG3Dm1, LCWG3Dp1
```

> **Breaking change in 1.4.0:** `spomso.cores` no longer re-exports `LCWG2D`, `LCWG3Dp1`, `LCWG3Dm1`, `lcwg1_2d`, `lcwg1_p1`, or `lcwg1_m1`. Importing them from `spomso.cores` raises `ImportError`. Use `spomso.cores.geom_vector_special` for the classes and `spomso.cores.vector_functions_special` for the underlying `lcwg1_*` functions.

| Class | Description |
|---|---|
| `LCWG2D(parameters, co_resolution, sign)` | 2D director field — z-component is always 0 |
| `LCWG3Dm1(parameters, co_resolution, sign)` | 3D field with winding number −1 in the YZ plane |
| `LCWG3Dp1(parameters, co_resolution, sign)` | 3D field with winding number +1 in the YZ plane |

**Parameters:**
- `LCWG2D`: `parameters` = total width of the waveguide (scalar)
- `LCWG3D*`: `parameters` = `(width, thickness)` tuple

**`sign`** controls the handedness of the director rotation:
- `None` or a `float` — computed automatically from the SDF topology (a float is used as the threshold)
- `+1` or `-1` — enforces a uniform sign everywhere
- a numpy array of ±1 — per-point control

**Typical usage:**

```python
from spomso.cores.geom_vector_special import LCWG2D, LCWG3Dm1
from spomso.cores.geom_3d import Z

# assume wg_pattern is the SDF of the waveguide (from wg.create(coor))

# 2D director field
vf_2d = LCWG2D(width, co_res_new, sign=None)
field_2d = vf_2d.create(wg_pattern)

# 3D director field — also needs the vertical SDF (distance to the midplane)
vertical = Z(-d / 2).create(coor)
vf_3d = LCWG3Dm1((width, thickness), co_res_new, sign=None)
field_3d = vf_3d.create((wg_pattern, vertical))   # tuple input
```

Also available: `compute_crossings_2d` from `spomso.cores.vector_functions_special`, for locating defect crossings in a 2D field.

---

## 12. Differentiable Vector Fields (JAX)

**New in 1.5.0:** the whole vector-field layer has a JAX counterpart, so vector fields can now be differentiated and optimised.

```python
from spomso.jax_cores import (
    cartesian_vector_field, spherical_vector_field, cylindrical_vector_field,
    radial_vector_field_spherical, radial_vector_field_cylindrical,
    hyperbolic_vector_field_cylindrical,
    aar_vector_field_cylindrical, awn_vector_field_cylindrical,
    vortex_vector_field_cylindrical, aav_vector_field_cylindrical,
    x_vector_field, y_vector_field, z_vector_field,
    from_sdf,
)
from spomso.jax_cores import (
    batch_normalize, add_vectors, subtract_vectors, rescale_vectors,
    rotate_vectors_phi, rotate_vectors_theta, rotate_vectors_axis,
    rotate_vectors_x_axis, rotate_vectors_y_axis, rotate_vectors_z_axis,
    revolve_field_x, revolve_field_y, revolve_field_z,
)
```

The JAX layer is **functional** — there is no `VectorField` class. You compose functions instead of chaining methods:

```python
import jax.numpy as jnp
from spomso.cores import generate_grid
from spomso.jax_cores import vortex_vector_field_cylindrical, rescale_vectors
from spomso.jax_cores.sdf_2D_jax import sdf_circle
from spomso.jax_cores.post_processing_jax import gaussian_boundary_jax

coor, res = generate_grid((8.0, 8.0), (400, 400))

def vortex(co, x0, y0, handedness, radius=1.0, sigma=2.0):
    p = jnp.subtract(co.T, jnp.asarray([x0, y0, 0])).T
    vf = vortex_vector_field_cylindrical(p)
    envelope = gaussian_boundary_jax(sdf_circle(p, radius), 1, sigma)
    return (vf * envelope * handedness)[:2]
```

This composes with `jax.grad`, `jax.jacfwd`, and `optax` exactly like the scalar SDF layer. The full worked example is in `examples/autodiff/vector_field_optimization.py`, and the autodiff guide covers the gradient machinery in detail.

Every function in both JAX vector modules works under `@jax.jit` and agrees with its NumPy counterpart to floating-point precision.

---

## 13. Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'LCWG2D' from 'spomso.cores'` | no longer re-exported since 1.4.0 | import from `spomso.cores.geom_vector_special` |
| `ImportError: cannot import name 'VectorField' from 'spomso.cores'` | pre-1.5.0 | upgrade, or import from `spomso.cores.geom` |
| `ValueError: Cannot reshape the pattern` | resolution length doesn't match the extents in `size` | slice to `co_res_new[:2]`, or give the grid a z extent |
| `VectorFieldFromSDF` returns `(2, N)` not `(3, N)` | component count follows the resolution length | expected on a 2D grid; use `nd_vector_smarter_reshape` |
| `IndexError: index 2 is out of bounds` from a rotation | rotations need 3 components, the field has 2 | use a thin 3D slab, or pad the field to 3 rows |
| Field is all zeros / nonsense | passed `coor` where components were expected (or vice versa) | check the input table in [section 2](#2-the-vectorfield-base-class) |
| `.theta()` looks wrong | pre-1.5.0 it ignored the vector magnitude | upgrade to 1.5.0 |
| `subtract()` seems to add | 1.3.0 bug | upgrade to 1.4.0+ |
| Hyperbolic or winding field raises `TypeError` | 1.3.0/1.4.0 bug | upgrade to 1.5.0 |
| `conv_averaging` dimension error | it needs a grid, not a flat array | `smarter_reshape` first |

---

## 14. Quick-Reference Cheat Sheet

### Imports

```python
from spomso.cores import generate_grid, smarter_reshape
from spomso.cores import vector_smarter_reshape, nd_vector_smarter_reshape
from spomso.cores import VectorField
from spomso.cores import (
    RadialSphericalVectorField, RadialCylindricalVectorField,
    VortexCylindricalVectorField, AngledVortexCylindricalVectorField,
    HyperbolicCylindricalVectorField, WindingCylindricalVectorField,
    AngledRadialCylindricalVectorField,
    CartesianVectorField, CylindricalVectorField, SphericalVectorField,
    XVectorField, YVectorField, ZVectorField,
    VectorFieldFromSDF,
)
from spomso.cores.geom_vector_special import LCWG2D, LCWG3Dm1, LCWG3Dp1
from spomso.cores import batch_normalize
```

### Typical workflow

```python
coor, res = generate_grid(co_size, co_resolution)

# 1. Build or choose a field
vf = RadialSphericalVectorField()

# 2. Apply modifications (optional)
vf.rotate_phi(np.pi / 6)
vf.normalize()

# 3. Evaluate
field = vf.create(coor)                        # shape (3, N)

# 4. Reshape
field_3d = vector_smarter_reshape(field, res)  # shape (3, nx, ny, nz)

# 5. Extract scalar maps
x      = smarter_reshape(vf.x(coor),      res)
phi    = smarter_reshape(vf.phi(coor),    res)
length = smarter_reshape(vf.length(coor), res)
```

On a 2D grid use `res[:2]` everywhere `res` appears above.

### VectorField modification methods

| Method | Effect |
|---|---|
| `.rotate_phi(alpha)` | Rotate about the z-axis by angle(s) alpha |
| `.rotate_theta(alpha)` | Rotate toward/away from the z-axis |
| `.rotate_x(alpha)` | Rotate about the x-axis |
| `.rotate_y(alpha)` | Rotate about the y-axis |
| `.rotate_z(alpha)` | Rotate about the z-axis |
| `.rotate_axis(axis, alpha)` | Rotate about an arbitrary axis (axis first) |
| `.revolution_x(coor)` | Revolve the field about the x-axis |
| `.revolution_y(coor)` | Revolve the field about the y-axis |
| `.revolution_z(coor)` | Revolve the field about the z-axis |
| `.rescale(f)` | Scale all vectors by f (scalar or array) |
| `.add(field)` | Add another `(3, N)` field element-wise |
| `.subtract(field)` | Subtract another `(3, N)` field element-wise |
| `.normalize()` | Set all vectors to unit length |

### VectorField accessor methods (call with the same input as `.create()`)

| Method | Returns |
|---|---|
| `.x(p)` | x-component, shape `(N,)` |
| `.y(p)` | y-component, shape `(N,)` |
| `.z(p)` | z-component, shape `(N,)` |
| `.phi(p)` | azimuthal angle, range `[-π, π]` |
| `.theta(p)` | polar angle `arccos(z/|v|)`, range `[0, π]` |
| `.length(p)` | vector magnitude |
