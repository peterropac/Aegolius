# SPOMSO – Vector Fields
### A practical guide to generating and manipulating vector fields

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
12. [Quick-Reference Cheat Sheet](#12-quick-reference-cheat-sheet)

---

## 1. What are vector fields in SPOMSO?

While the SDF subsystem assigns one scalar value per grid point (the signed distance to the nearest surface), the **vector field** subsystem assigns a 3D vector per grid point. This enables applications like:

- director fields in liquid crystals
- flow fields around geometry
- surface normal maps
- any physically motivated orientation field derived from geometry

All vector fields share the same coordinate grid as scalar SDFs — set up once with `generate_grid` and reused everywhere.

**Additional import for vector fields:**

```python
from spomso.cores.helper_functions import generate_grid, smarter_reshape, vector_smarter_reshape
```

The `vector_smarter_reshape` helper converts the flat `(3, N)` vector field result into a `(3, nx, ny, nz)` grid.

---

## 2. The VectorField Base Class

Every vector field object is an instance of `VectorField` (from `spomso.cores.geom`). You rarely instantiate it directly — you use a built-in subclass or the `VectorField(my_function, *params)` constructor for custom fields.

### Evaluating a field

```python
field = vf.create(p)   # shape (3, N) — one 3-vector per grid point
```

The input `p` is either the coordinate cloud `coor` (shape `(3, N)`) **or** a pre-computed SDF array (shape `(N,)`) depending on the type of field. See sections 6 and 11 for the SDF-driven cases.

### Reshaping for visualisation

```python
from spomso.cores.helper_functions import vector_smarter_reshape

field_3d = vector_smarter_reshape(field, co_resolution)
# shape (3, nx, ny, nz)
# field_3d[0] = x-components, field_3d[1] = y-components, field_3d[2] = z-components
```

---

## 3. Built-in Vector Fields

All built-in fields live in `spomso.cores.geom_vector`. They take the coordinate cloud `coor` as input to `.create()`.

```python
from spomso.cores.geom_vector import (
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
| `RadialCylindricalVectorField` | `()` | Radial outward in the XY plane, z-component = 0 |
| `VortexCylindricalVectorField` | `()` | Perpendicular to the radial direction in XY; a curl around the z-axis |
| `HyperbolicCylindricalVectorField` | `()` | Hyperbolic saddle field in XY |
| `WindingCylindricalVectorField` | `(gamma)` | Vectors rotate in XY with winding number `gamma` (can be fractional or negative) |
| `AngledRadialCylindricalVectorField` | `(alpha)` | Radial field rotated by `alpha` radians away from the radial direction |
| `AngledVortexCylindricalVectorField` | `(alpha)` | Vortex field rotated by `alpha` radians |
| `XVectorField` | `()` | Uniform field pointing in +x |
| `YVectorField` | `()` | Uniform field pointing in +y |
| `ZVectorField` | `()` | Uniform field pointing in +z |
| `VectorFieldFromSDF` | `(co_resolution)` | Gradient of an SDF, normalised to unit length |

### Minimal example

```python
import numpy as np
from spomso.cores.helper_functions import generate_grid, smarter_reshape, vector_smarter_reshape
from spomso.cores.geom_vector import RadialSphericalVectorField

coor, res = generate_grid((4, 4, 4), (50, 50, 50))

vf    = RadialSphericalVectorField()
field = vf.create(coor)          # shape (3, N)
field_3d = vector_smarter_reshape(field, res)   # shape (3, 50, 50, 50)
```

---

## 4. Defining Fields by Components

Use coordinate-system classes when you want to specify each vector component explicitly from a formula.

```python
from spomso.cores.geom_vector import (
    CartesianVectorField,
    CylindricalVectorField,
    SphericalVectorField,
)
```

The workflow is:

1. Build a component array of shape `(3, N)` using any numpy operations on `coor`.
2. Instantiate the appropriate class.
3. Optionally apply modifications.
4. Call `.create(components)`.

### Cartesian components

```python
u = np.ones(coor.shape[1])           # x-component: constant 1
v = np.zeros(coor.shape[1])          # y-component: 0
w = coor[2]                          # z-component: varies with z

components = np.asarray((u, v, w))
vf = CartesianVectorField()
field = vf.create(components)
```

### Cylindrical components

The class converts `(r, phi, z)` → Cartesian `(x, y, z)` automatically.

```python
r   = np.ones(coor.shape[1])
phi = np.arctan2(coor[1], coor[0])   # azimuthal angle at each point
z   = np.zeros(coor.shape[1])

components = np.asarray((r, phi, z))
vf = CylindricalVectorField()
field = vf.create(components)
```

### Spherical components

The class converts `(r, phi, theta)` → Cartesian.

```python
r     = np.ones(coor.shape[1])
phi   = np.arctan2(coor[1], coor[0])
theta = np.arccos(coor[2] / (np.linalg.norm(coor, axis=0) + 1e-12))

components = np.asarray((r, phi, theta))
vf = SphericalVectorField()
field = vf.create(components)
```

> **Note:** The output of all three classes is always in Cartesian `(x, y, z)` coordinates, regardless of which coordinate system you used to specify the components.

---

## 5. Vector Field Modifications

All `VectorField` objects inherit from `ModifyVectorObject`, giving them a set of in-place modification methods. Modifications chain and are applied in order.

### Rotation

```python
vf.rotate_phi(np.pi/4)          # rotate every vector by π/4 around z-axis
vf.rotate_theta(-np.pi/6)       # rotate every vector by π/6 in the polar direction
vf.rotate_x(alpha)              # rotate around x-axis (alpha can be a scalar or array)
vf.rotate_y(alpha)              # rotate around y-axis
vf.rotate_z(alpha)              # rotate around z-axis
vf.rotate_axis((1,0,0), alpha)  # rotate around an arbitrary axis
```

When `alpha` is an **array of shape `(N,)`**, each grid point gets its own rotation angle — this is how spatially varying rotations are applied (e.g. twisting a field around a curve).

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

### Normalising

```python
vf.normalize()   # set all vectors to unit length
```

### Inspection

```python
print(vf.modifications)   # list of applied modification names
```

### Full example with chained modifications

```python
from spomso.cores.geom_vector import SphericalVectorField

r_   = np.ones(coor.shape[1])
phi_ = np.pi * coor[0] / (co_size[0] / 2)
theta_ = np.pi * coor[2] / co_size[2] + np.pi/2

components = np.asarray((r_, phi_, theta_))
vf = SphericalVectorField()

# rotate every vector 90° around the z-axis
vf.rotate_phi(np.pi/2)

# apply a spatially varying polar rotation based on z-position
vf.rotate_theta(-np.pi/4)

# scale vectors by a value that increases away from z=0
scale = np.abs(coor[2] / co_size[2]) + 0.1
vf.rescale(scale)

# add a second field and normalise
second = SphericalVectorField()
second_components = np.asarray((r_, phi_ * 0, -theta_))
second_field = second.create(second_components)
vf.add(second_field)
vf.normalize()

field = vf.create(components)
```

---

## 6. Deriving a Vector Field from an SDF

`VectorFieldFromSDF` computes the **normalised gradient** of any SDF. This gives you a field that points perpendicularly away from the SDF's zero level-set (i.e. the surface normals of the encoded geometry).

```python
from spomso.cores.geom_vector import VectorFieldFromSDF
from spomso.cores.geom_2d import Circle

# build a 2D SDF
circle = Circle(30)
sdf_values = circle.create(coor)   # shape (N,)

# create the gradient-based vector field
vf = VectorFieldFromSDF(co_resolution)

# optionally apply spatial rotations based on distance to the surface
from spomso.cores.post_processing import linear_falloff
phi_rotation = linear_falloff(sdf_values, np.pi/2, 20)   # ramps from 0→π/2 over distance 20
vf.rotate_z(phi_rotation)

field = vf.create(sdf_values)   # <-- input is the SDF array, NOT coor
```

> **Key distinction:** `VectorFieldFromSDF.create()` takes the **SDF array** `(N,)` as input, not the coordinate cloud. The field is computed by taking the numerical gradient of the SDF on the grid (using `np.gradient` internally), so it requires `co_resolution` at construction time.

The gradient is smoothest when the SDF is smooth. If you see artefacts, smooth the SDF first:

```python
from spomso.cores.post_processing import conv_averaging
from spomso.cores.helper_functions import smarter_reshape

sdf_3d = smarter_reshape(sdf_values, co_resolution)
sdf_smooth = conv_averaging(sdf_3d, (5, 5, 1), iterations=1)
sdf_values_smooth = sdf_smooth.reshape(sdf_values.shape)

vf = VectorFieldFromSDF(co_resolution)
field = vf.create(sdf_values_smooth)
```

---

## 7. Revolving a Vector Field into 3D

A 2D vector field can be revolved around one of the coordinate axes to produce a full 3D field. This is the vector analogue of the SDF `.revolution()` modification.

```python
vf = VectorFieldFromSDF(co_resolution)

# revolve the 2D field around the z-axis
vf.revolution_z(coor)   # coor is needed to compute the azimuthal angle at each point

# also available:
# vf.revolution_x(coor)
# vf.revolution_y(coor)

field = vf.create(sdf_values)
```

The revolution methods rotate each 2D vector by the azimuthal/polar angle of the corresponding grid point, creating a rotationally symmetric 3D field.

---

## 8. Custom Vector Fields

Wrap any function `f(p, *params) -> np.ndarray` of shape `(3, N)` using `VectorField` directly:

```python
from spomso.cores.geom import VectorField
from spomso.cores.vector_modification_functions import batch_normalize
from spomso.cores.helper_functions import smarter_reshape

def custom_radial_vf(co, order):
    """Radial field based on the L^order norm."""
    u = np.linalg.norm(co, axis=0, ord=order)
    u_3d = smarter_reshape(u, co_resolution)
    grad = np.asarray(np.gradient(u_3d)).reshape(3, -1)
    return batch_normalize(grad)

vf = VectorField(custom_radial_vf, 3)   # order=3
field = vf.create(coor)
```

The function signature must be `f(p, *params)` where `p` is the input (either `coor` or an SDF array) and `*params` are any additional parameters you pass after the function in the constructor.

### Low-level transformation utilities

The functions in `spomso.cores.vector_modification_functions` operate directly on `(3, N)` arrays and can be called inside custom field functions:

```python
from spomso.cores.vector_modification_functions import (
    batch_normalize,         # normalise each column to unit length
    rotate_vectors_phi,      # rotate by azimuthal angle(s)
    rotate_vectors_theta,    # rotate by polar angle(s)
    rotate_vectors_x_axis,   # rotate around x-axis
    rotate_vectors_y_axis,   # rotate around y-axis
    rotate_vectors_z_axis,   # rotate around z-axis
    rotate_vectors_axis,     # rotate around an arbitrary axis
)
```

All rotation functions accept either a scalar or an array of per-point angles, enabling spatially varying rotations.

---

## 9. Extracting Components and Angles

After calling `.create()`, you can extract individual scalar maps via convenience methods — all accepting the same input `p` that was passed to `.create()`:

```python
x      = vf.x(p)       # x-component, shape (N,)
y      = vf.y(p)       # y-component
z      = vf.z(p)       # z-component
phi    = vf.phi(p)     # azimuthal angle φ = atan2(y, x), range [-π, π]
theta  = vf.theta(p)   # polar angle θ = acos(z/r), range [0, π]
length = vf.length(p)  # vector magnitude |v|
```

Reshape each to a 3D grid with `smarter_reshape`:

```python
phi_3d = smarter_reshape(phi, co_resolution)   # shape (nx, ny, nz)
```

---

## 10. Visualising Vector Fields

### 2D quiver plot (matplotlib)

The standard pattern is to decimate the field (show every `N`-th vector) to avoid overplotting:

```python
import matplotlib.pyplot as plt
from spomso.cores.helper_functions import smarter_reshape, vector_smarter_reshape

field_3d = vector_smarter_reshape(field, co_resolution)   # (3, nx, ny, nz)
x_3d = smarter_reshape(x, co_resolution)
y_3d = smarter_reshape(y, co_resolution)

decimate = 5
depth = co_resolution[2] // 2   # mid-plane along z

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(x_3d[:, :, depth].T, cmap="bwr", origin="lower",
          extent=(-co_size[0]/2, co_size[0]/2, -co_size[1]/2, co_size[1]/2))
ax.quiver(
    smarter_reshape(coor[0], co_resolution)[::decimate, ::decimate, depth],
    smarter_reshape(coor[1], co_resolution)[::decimate, ::decimate, depth],
    field_3d[0, ::decimate, ::decimate, depth],
    field_3d[1, ::decimate, ::decimate, depth],
)
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()
```

### 3D cone plot (Plotly)

```python
import plotly.graph_objects as go

decimate = 4
xx = smarter_reshape(coor[0], co_resolution)[::decimate, ::decimate, :].flatten()
yy = smarter_reshape(coor[1], co_resolution)[::decimate, ::decimate, :].flatten()
zz = smarter_reshape(coor[2], co_resolution)[::decimate, ::decimate, :].flatten()
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

Any of the scalar maps (`x`, `y`, `phi`, `theta`, `length`) can be visualised with `imshow` exactly like an SDF cross-section. Use `cmap="bwr"` (diverging) for signed quantities like `x`, `y`, `z` and `phi`, and `cmap="viridis"` for unsigned ones like `length` and `theta`.

---

## 11. Special Fields: Liquid Crystal Waveguides

`spomso.cores.geom_vector_special` provides three highly specialised vector fields designed for liquid crystal director patterns inside waveguides. These take an **SDF array** (the distance field of the waveguide geometry) as their primary input, not `coor`.

| Class | Description |
|---|---|
| `LCWG2D(parameters, co_resolution, sign)` | 2D director field — z-component is always 0 |
| `LCWG3Dm1(parameters, co_resolution, sign)` | 3D field with winding number −1 in the YZ plane |
| `LCWG3Dp1(parameters, co_resolution, sign)` | 3D field with winding number +1 in the YZ plane |

**Parameters:**
- `LCWG2D`: `parameters` = total width of the waveguide (scalar)
- `LCWG3D*`: `parameters` = `(width, thickness)` tuple

**`sign`** controls the handedness of the director rotation:
- `None` or `float` — computed automatically from the SDF topology (float is used as the threshold)
- `+1` or `-1` — enforces a uniform sign everywhere
- numpy array of ±1 — per-point control

**Typical usage:**

```python
from spomso.cores.geom_vector_special import LCWG2D, LCWG3Dm1
from spomso.cores.geom_3d import Z

# assume wg_pattern is the SDF of the waveguide (from wg.create(coor))

# 2D director field
vf_2d = LCWG2D(width, co_resolution, sign=None)
field_2d = vf_2d.create(wg_pattern)

# 3D director field — also needs the vertical SDF (distance to midplane)
vertical = Z(-d/2).create(coor)   # distance to the z = -d/2 plane
vf_3d = LCWG3Dm1((width, thickness), co_resolution, sign=None)
field_3d = vf_3d.create((wg_pattern, vertical))   # tuple input
```

---

## 12. Quick-Reference Cheat Sheet

### Imports

```python
from spomso.cores.helper_functions import generate_grid, smarter_reshape, vector_smarter_reshape
from spomso.cores.geom import VectorField
from spomso.cores.geom_vector import (
    RadialSphericalVectorField, RadialCylindricalVectorField,
    VortexCylindricalVectorField, WindingCylindricalVectorField,
    AngledRadialCylindricalVectorField,
    CartesianVectorField, CylindricalVectorField, SphericalVectorField,
    XVectorField, YVectorField, ZVectorField,
    VectorFieldFromSDF,
)
from spomso.cores.geom_vector_special import LCWG2D, LCWG3Dm1, LCWG3Dp1
from spomso.cores.vector_modification_functions import batch_normalize
```

### Typical workflow

```python
coor, res = generate_grid(co_size, co_resolution)

# 1. Build or choose a field
vf = RadialSphericalVectorField()

# 2. Apply modifications (optional)
vf.rotate_phi(np.pi/6)
vf.normalize()

# 3. Evaluate
field = vf.create(coor)                    # shape (3, N)

# 4. Reshape
field_3d = vector_smarter_reshape(field, res)  # shape (3, nx, ny, nz)

# 5. Extract scalar maps
x      = smarter_reshape(vf.x(coor),      res)
phi    = smarter_reshape(vf.phi(coor),    res)
length = smarter_reshape(vf.length(coor), res)
```

### VectorField modification methods

| Method | Effect |
|---|---|
| `.rotate_phi(alpha)` | Rotate in the XY plane (around z) by angle(s) alpha |
| `.rotate_theta(alpha)` | Rotate toward the z-axis by angle(s) alpha |
| `.rotate_x(alpha)` | Rotate around the x-axis |
| `.rotate_y(alpha)` | Rotate around the y-axis |
| `.rotate_z(alpha)` | Rotate around the z-axis |
| `.rotate_axis(axis, alpha)` | Rotate around an arbitrary axis |
| `.revolution_x(coor)` | Revolve a 2D field around the x-axis |
| `.revolution_y(coor)` | Revolve a 2D field around the y-axis |
| `.revolution_z(coor)` | Revolve a 2D field around the z-axis |
| `.rescale(f)` | Scale all vectors by factor f (scalar or array) |
| `.add(field)` | Add another `(3, N)` field element-wise |
| `.subtract(field)` | Subtract another `(3, N)` field element-wise |
| `.normalize()` | Set all vectors to unit length |

### VectorField accessor methods (call with same input as `.create()`)

| Method | Returns |
|---|---|
| `.x(p)` | x-component, shape `(N,)` |
| `.y(p)` | y-component, shape `(N,)` |
| `.z(p)` | z-component, shape `(N,)` |
| `.phi(p)` | azimuthal angle, range `[-π, π]` |
| `.theta(p)` | polar angle, range `[0, π]` |
| `.length(p)` | vector magnitude |
