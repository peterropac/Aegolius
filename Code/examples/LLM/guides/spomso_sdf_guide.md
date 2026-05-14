# SPOMSO – Procedural Geometry with Signed Distance Functions
### A practical guide for generating 3D (and 2D) geometry in Python

---

## Table of Contents
1. [What is SPOMSO?](#1-what-is-spomso)
2. [Installation](#2-installation)
3. [The Core Idea: Signed Distance Functions](#3-the-core-idea-signed-distance-functions)
4. [The Coordinate Grid](#4-the-coordinate-grid)
5. [Primitive Shapes](#5-primitive-shapes)
6. [Euclidean Transformations](#6-euclidean-transformations)
7. [Modifications](#7-modifications)
8. [Combining Shapes](#8-combining-shapes)
9. [Binarisation and Post-processing](#9-binarisation-and-post-processing)
10. [Extracting a Surface Point Cloud](#10-extracting-a-surface-point-cloud)
11. [Visualising Results](#11-visualising-results)
12. [2D Geometry](#12-2d-geometry)
13. [Advanced: Repetitions and Instancing](#13-advanced-repetitions-and-instancing)
14. [Advanced: Custom SDFs and GenericGeometry](#14-advanced-custom-sdfs-and-genericgeometry)
15. [Order-of-Operations Rules](#15-order-of-operations-rules)
16. [Full Example – A Chair](#16-full-example--a-chair)
17. [Quick-Reference Cheat Sheet](#17-quick-reference-cheat-sheet)

---

## 1. What is SPOMSO?

SPOMSO (v1.3.0) is a Python library for building **procedural geometry** using **Signed Distance Functions (SDFs)**. Instead of describing shapes with triangles or vertices, every shape is represented as a mathematical field: at any point in space the field value tells you the signed distance to the nearest surface (negative inside the object, positive outside, zero on the surface).

This representation makes it trivial to:
- blend, union, subtract, or intersect shapes with a single `min`/`max` call
- round edges, hollow objects, repeat geometry infinitely, twist or bend shapes — all without mesh surgery
- export clean surface point clouds by thresholding the SDF at zero

**Key dependencies:** `numpy`, `scipy`  
**Optional (for plotting):** `matplotlib`, `plotly`  
**Optional (for autodiff):** `jax`, `jaxlib`

---

## 2. Installation

Clone or download the [Aegolius repository](https://github.com/peterropac/Aegolius), then install the package from the `Code/spomso` subdirectory:

```bash
pip install -e path/to/Aegolius/Code/spomso          # core only
pip install -e "path/to/Aegolius/Code/spomso[plot]"  # + matplotlib + plotly
pip install -e "path/to/Aegolius/Code/spomso[all]"   # everything incl. JAX
```

Then in Python:

```python
import spomso          # confirms the install
```

---

## 3. The Core Idea: Signed Distance Functions

Every geometry object in SPOMSO carries an internal SDF `f(co)` where `co` is a coordinate array of shape `(3, N)`. The field value is:

| Region | SDF value |
|--------|-----------|
| Inside the object | negative |
| On the surface | 0 |
| Outside the object | positive |

You never call the SDF manually. Instead you call `.create(coor)` on any geometry object, which evaluates the SDF (after applying all modifications and transformations) over the entire grid and returns a flat numpy array of shape `(N,)`.

```python
sdf_values = my_shape.create(coor)   # shape (N,) — one value per grid point
```

---

## 4. The Coordinate Grid

All geometry is evaluated on a regular 3D grid. Two helper functions from `spomso.cores.helper_functions` set this up:

```python
from spomso.cores.helper_functions import generate_grid, smarter_reshape

co_size       = (4.0, 4.0, 3.0)   # physical extent (x, y, z)
co_resolution = (100, 100, 75)    # number of voxels along each axis

coor, co_res_new = generate_grid(co_size, co_resolution)
# coor       : numpy array, shape (3, N)  — flattened grid of 3D points
# co_res_new : tuple — actual resolution (made odd internally if needed)
```

The grid is always **centred at the origin**. The physical extent of the grid along each axis goes from `-size/2` to `+size/2`.

To convert the flat SDF result back into a 3D array for slicing or plotting:

```python
field_3d = smarter_reshape(sdf_values, co_resolution)
# field_3d : numpy array, shape (nx, ny, nz)
```

**Practical tip:** Start with a coarse resolution (e.g. 50³) for fast iteration, then increase to 150³ or more for final outputs. Very coarse grids will produce jagged surfaces.

---

## 5. Primitive Shapes

All shapes live in `spomso.cores.geom_3d` (3D) or `spomso.cores.geom_2d` (2D). Import only what you need:

```python
from spomso.cores.geom_3d import (
    Box, Sphere, Cylinder, Cone, Torus,
    SegmentedLine3D, ParametricCurve3D,
    InfiniteCylinder, Plane
)
```

### 3D Primitives

| Class | Constructor arguments | Notes |
|---|---|---|
| `Box(a, b, c)` | side lengths along x, y, z | axis-aligned rectangular box |
| `Sphere(r)` | radius | centred at origin |
| `Cylinder(r, h)` | radius, height | axis along z |
| `Cone(h, angle)` | height, slope angle (radians) | tip at top |
| `Torus(R, r)` | primary radius, tube radius | ring in the XZ plane |
| `Plane(normal, thickness)` | normal vector, thickness | infinite slab |
| `InfiniteCylinder(r)` | radius | infinite along z |
| `SegmentedLine3D(points)` | list/array of 3D points | polyline / wire |
| `ParametricCurve3D(func, params, t_range)` | function, params, t range | smooth curve |

```python
box      = Box(1.0, 0.5, 0.25)     # 1 × 0.5 × 0.25 box
sphere   = Sphere(0.4)
cylinder = Cylinder(0.2, 1.0)      # radius 0.2, height 1.0
torus    = Torus(0.8, 0.15)        # ring radius 0.8, tube radius 0.15
cone     = Cone(1.2, np.pi/8)      # height 1.2, slope angle 22.5°
```

---

## 6. Euclidean Transformations

Every geometry object inherits from `EuclideanTransform`. Transformations are applied **after** all modifications (see section 7 for why this matters).

### Translation

```python
shape.move((0.5, 0.0, -1.0))        # relative — accumulates
shape.set_location((0.5, 0.0, -1.0))  # absolute — overrides previous
```

### Rotation

```python
shape.rotate(np.pi/4, (0, 0, 1))    # 45° around the z-axis
shape.set_rotation(np.pi/2, (1, 0, 0))  # absolute — 90° around x
```

`rotate` accepts either `(angle, axis)` or a `(3, 3)` rotation matrix.

### Scaling

```python
shape.rescale(2.0)         # multiply existing scale by 2
shape.set_scale(1.5)       # set absolute scale to 1.5 (overrides previous)
```

Scaling is **isotropic** (uniform in all directions).

### Inspecting the current state

```python
print(shape.center)            # current position
print(shape.scale)             # current scale factor
print(shape.rotation_matrix)   # current 3×3 rotation matrix
print(shape.transformations)   # list of applied transformation names
```

---

## 7. Modifications

Modifications alter the **shape** of the SDF itself (as opposed to transformations, which move/rotate/scale it). They are applied in the order they are called and always run **before** Euclidean transformations.

All modification methods are on every geometry object (inherited from `ModifyObject`).

### Rounding edges

```python
shape.rounding(0.05)               # round all edges by radius 0.05
                                   # (also thickens the object by 0.05)

shape.rounding_cs(0.05, max_dim)   # round without thickening;
                                   # max_dim = largest dimension of the object
```

### Hollowing / surface shell

```python
shape.onion(0.02)    # keeps only a shell of thickness 0.02
```

### Elongation

```python
shape.elongation((0.5, 0.0, 0.0))  # stretch by 0.5 along x
```

### Extrusion (2D → 3D)

```python
circle.extrusion(1.0)   # extrude a 2D circle into a cylinder of height 1.0
```

### Revolution (2D → 3D)

```python
# Revolve a 2D cross-section around the y-axis to create a surface of revolution
profile.revolution(radius=0.5)
```

### Twist

```python
shape.twist(pitch=np.pi)   # twist rate in radians per unit length along z
```

### Bend

```python
shape.bend(radius=1.0, angle=np.pi/2)   # bend the shape around the z-axis
```

### Shear

```python
shape.shear_xz(angle=np.pi/6)   # shear in the XZ plane; also: shear_yz, shear_xy, etc.
```

### Mirror

```python
# Mirror the shape: copy at position `a`, original at position `b`
shape.mirror(a=(-1, 0, 0), b=(1, 0, 0))
```

### Infinite / finite repetition

```python
shape.infinite_repetition((1.0, 1.0, 1.0))   # tile with spacing 1 along each axis

shape.finite_repetition(
    size=(4.0, 4.0, 0.0),   # bounding box of the repeated region
    repetitions=(4, 4, 1)   # number of copies along each axis
)
```

### Inspecting applied modifications

```python
print(shape.modifications)   # list of modification names in order
```

---

## 8. Combining Shapes

Import `CombineGeometry` from `spomso.cores.combine`:

```python
from spomso.cores.combine import CombineGeometry
```

### Non-parametric operations

```python
union     = CombineGeometry("UNION")
result    = union.combine(shape_a, shape_b, shape_c)   # any number of shapes

subtract  = CombineGeometry("SUBTRACT2")
result    = subtract.combine(base, cutter)             # base minus cutter

intersect = CombineGeometry("INTERSECT2")
result    = intersect.combine(shape_a, shape_b)
```

| Operation string | Meaning |
|---|---|
| `"UNION"` | union of any number of shapes (n-ary) |
| `"UNION2"` | union of exactly 2 shapes |
| `"SUBTRACT2"` | shape_a minus shape_b |
| `"INTERSECT"` | intersection of any number of shapes |
| `"INTERSECT2"` | intersection of exactly 2 shapes |

### Smooth (parametric) operations

These create a blended transition between shapes rather than a sharp seam:

```python
smooth_union = CombineGeometry("SMOOTH_UNION2")
result = smooth_union.combine_parametric(shape_a, shape_b, parameters=0.15)
# parameters = smoothing radius (larger → more blending)
```

| Operation string | Meaning |
|---|---|
| `"SMOOTH_UNION2"` | smooth union (poly3 kernel) |
| `"SMOOTH_UNION2_2"` | smooth union (poly2 kernel) |
| `"SMOOTH_SUBTRACT2"` | smooth subtraction |
| `"SMOOTH_INTERSECT2"` | smooth intersection |

### Reusing a combiner

A `CombineGeometry` instance can be reused for multiple calls:

```python
union = CombineGeometry("UNION2")
ab    = union.combine(a, b)
abc   = union.combine(ab, c)   # chain as many times as needed
```

### The result is a `GenericGeometry`

The object returned by `.combine()` or `.combine_parametric()` behaves exactly like any primitive: you can apply further modifications, transformations, and combine it again.

```python
result.move((0, 0, 0.5))
result.rounding(0.03)
final_sdf = result.create(coor)
```

---

## 9. Binarisation and Post-processing

The raw SDF is a continuous field. To get a binary solid (inside = 1, outside = 0):

```python
from spomso.cores.post_processing import hard_binarization

binary = hard_binarization(sdf_values, threshold=0)
# binary : numpy array of 0.0 / 1.0, same shape as sdf_values
```

Other post-processing functions (all operate on the flat SDF array):

| Function | Effect |
|---|---|
| `hard_binarization(u, threshold)` | step function at threshold |
| `linear_falloff(u, amplitude, width)` | linear ramp at the surface |
| `gaussian_falloff(u, amplitude, width)` | Gaussian falloff from surface |
| `gaussian_boundary(u, amplitude, width)` | Gaussian bump centred on surface |
| `sigmoid_falloff(u, amplitude, width)` | smooth step |
| `relu(u, width)` | one-sided linear ramp |
| `conv_averaging(u, kernel_size, iterations)` | spatial smoothing (needs 3D array) |
| `conv_edge_detection(u)` | edge detection (needs 3D array) |

---

## 10. Extracting a Surface Point Cloud

The surface of any shape is the zero level-set of its SDF. Extract it by keeping only the grid points within a small distance of zero:

```python
dx = co_size[0] / co_resolution[0]    # one voxel step
surface_mask = np.abs(sdf_values) < dx * 1.5
surface_pts  = coor[:, surface_mask].T  # shape (N_surface, 3)

np.save("my_shape.npy", surface_pts)
```

The density of the point cloud depends on the grid resolution. A 100³ grid over a 2-unit domain gives roughly 0.02-unit spacing between surface points.

---

## 11. Visualising Results

### Cross-section slice (matplotlib)

```python
import matplotlib.pyplot as plt
from spomso.cores.helper_functions import smarter_reshape
from spomso.cores.post_processing import hard_binarization

field_3d  = smarter_reshape(sdf_values, co_resolution)
binary_3d = hard_binarization(sdf_values, 0)
binary_3d = smarter_reshape(binary_3d, co_resolution)

fig, ax = plt.subplots()
# XZ mid-plane (y = 0, index co_resolution[1]//2)
ax.imshow(
    binary_3d[:, co_resolution[1]//2, :].T,
    cmap="binary_r", origin="lower",
    extent=(-co_size[0]/2, co_size[0]/2, -co_size[2]/2, co_size[2]/2)
)
ax.set_xlabel("x"); ax.set_ylabel("z")
plt.show()
```

Swap the slice axis for other planes:
- XY (top view): `binary_3d[:, :, co_resolution[2]//2]`
- YZ (front view): `binary_3d[co_resolution[0]//2, :, :]`

### Interactive 3D volume (Plotly)

```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Volume(
    x=coor[0], y=coor[1], z=coor[2],
    value=binary_flat,
    isomin=0.5, isomax=1.0,
    opacity=0.08,
    surface_count=3,
))
fig.show()
```

### 3D scatter of the surface point cloud (matplotlib)

```python
fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2],
           c=surface_pts[:, 2], cmap="plasma", s=0.5)
plt.show()
```

---

## 12. 2D Geometry

All 2D primitives live in `spomso.cores.geom_2d`. They work identically to the 3D ones but operate in the XY plane. Use them when you want to:
- visualise a cross-section quickly on a 2D grid
- create a 2D profile and then **extrude** or **revolve** it into 3D

```python
from spomso.cores.geom_2d import Circle, Rectangle, Polygon, Arc, Sector

circle    = Circle(0.5)
rect      = Rectangle(1.0, 0.6)
```

2D shapes can be extruded or revolved via the modification methods:

```python
circle.extrusion(2.0)   # becomes a cylinder of height 2
rect.revolution(1.0)    # becomes a torus-like surface of revolution
```

Available 2D shapes include: `Circle`, `NEUCircle`, `NGon`, `Polygon`, `Rectangle`, `RoundedRectangle`, `Segment`, `Triangle`, `Sector`, `Arc`, `ParametricCurve`, `SegmentedLine`, `PointCloud2D`.

---

## 13. Advanced: Repetitions and Instancing

### Finite repetition

Tile a shape a fixed number of times within a bounding box:

```python
pillar = Cylinder(0.05, 1.0)
pillar.finite_repetition(
    size=(2.0, 2.0, 0.0),     # region in which to repeat
    repetitions=(5, 5, 1)     # 5×5 grid of pillars
)
```

### Infinite repetition

Fill all of space with copies of the shape on a regular lattice:

```python
sphere = Sphere(0.1)
sphere.infinite_repetition((0.4, 0.4, 0.4))   # copy every 0.4 units
```

### Curve instancing

Place copies of a shape along a parametric curve, optionally aligning each copy to the curve's tangent/normal/binormal frame:

```python
def helix(t, R, H, freq):
    return np.asarray([R*np.cos(2*np.pi*freq*t),
                       R*np.sin(2*np.pi*freq*t),
                       H*t - H/2])

box = Box(0.3, 0.15, 0.1)

# Simple placement (no alignment)
box.curve_instancing(helix, (1.0, 2.0, 3.0), (0, 1, 20))

# Align each instance to the tangent of the curve
box.aligned_curve_instancing(helix, (1.0, 2.0, 3.0), (0, 1, 20))

# Align to tangent + normal + binormal (full Frenet frame)
box.fully_aligned_curve_instancing(helix, (1.0, 2.0, 3.0), (0, 1, 20))
```

The second argument is passed as `*params` to the curve function; the third is `(t_start, t_end, n_instances)`.

---

## 14. Advanced: Custom SDFs and GenericGeometry

You can wrap any callable `f(co, *params) -> np.ndarray` as a first-class geometry object:

```python
from spomso.cores.geom import GenericGeometry

def my_sdf(co, radius, twist_rate):
    # co has shape (3, N)
    angle = twist_rate * co[2]
    x_rot = co[0]*np.cos(angle) - co[1]*np.sin(angle)
    y_rot = co[0]*np.sin(angle) + co[1]*np.cos(angle)
    r     = np.sqrt(x_rot**2 + y_rot**2)
    return r - radius

shape = GenericGeometry(my_sdf, 0.5, np.pi)
shape.move((0, 0, 0.5))
sdf_values = shape.create(coor)
```

This lets you define any SDF — procedural noise fields, imported data, physics-derived surfaces — and use it seamlessly within the rest of the API.

### Wrapping an already-modified object

If you need modifications to run **after** transformations (the opposite of the default), wrap the object:

```python
# Normal order:  modifications → transformations
box.rotate(np.pi/4, (0, 0, 1))
box.mirror((-1, 0, 0), (1, 0, 0))  # mirror runs before rotation

# Reorder by wrapping:
box_wrapped = GenericGeometry(box.propagate)
box_wrapped.mirror((0, -1, 0), (0, 1, 0))  # this mirror runs after the rotation
```

---

## 15. Order-of-Operations Rules

This is the most important rule to internalise:

> **Modifications always run before Euclidean transformations, regardless of call order.**

```python
box = Box(1, 1, 1)
box.rotate(np.pi/4, (0, 0, 1))    # called first, but applied second
box.mirror((-1, 0, 0), (1, 0, 0)) # called second, but applied first
```

In the above, the mirror is applied to the unrotated box, and *then* the result is rotated. For mirror, repetition, and instancing operations the order relative to rotation matters geometrically. When you need a specific order, wrap the intermediate result in `GenericGeometry(obj.propagate)` to "freeze" the transformations before adding more modifications.

Modification methods (rounding, twist, bend, etc.) that do not depend on the object's world-space location are unaffected by this rule.

---

## 16. Full Example – A Chair

Below is a complete, self-contained script that builds a four-legged chair and outputs cross-section images plus a surface point cloud.

```python
import numpy as np
import matplotlib.pyplot as plt
from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Box, Cylinder
from spomso.cores.combine import CombineGeometry

# ── Grid ─────────────────────────────────────────────────────────────────────
CO_SIZE       = (1.2, 1.2, 1.8)
CO_RESOLUTION = (80, 80, 120)
coor, co_res_new = generate_grid(CO_SIZE, CO_RESOLUTION)

# ── Seat ──────────────────────────────────────────────────────────────────────
seat = Box(0.90, 0.90, 0.10)
seat.rounding(0.04)
seat.move((0.0, 0.0, -0.05))     # top surface at z = 0

# ── Back rest ─────────────────────────────────────────────────────────────────
backrest = Box(0.84, 0.06, 0.65)
backrest.rounding(0.04)
backrest.move((0.0, -0.42, 0.325))
backrest.rotate(np.deg2rad(5), (1, 0, 0))   # tilt back 5°

top_rail = Box(0.84, 0.08, 0.06)
top_rail.rounding(0.03)
top_rail.move((0.0, -0.42, 0.62))
top_rail.rotate(np.deg2rad(5), (1, 0, 0))

# ── Legs ──────────────────────────────────────────────────────────────────────
leg_positions = [(0.42, 0.42, -0.475), (0.42, -0.42, -0.475),
                 (-0.42, 0.42, -0.475), (-0.42, -0.42, -0.475)]
legs = []
for pos in leg_positions:
    leg = Cylinder(0.045, 0.75)
    leg.rounding(0.01)
    leg.move(pos)
    legs.append(leg)

# ── Support rails under the seat ──────────────────────────────────────────────
front_rail = Box(0.80, 0.04, 0.12); front_rail.rounding(0.01); front_rail.move((0, 0.42, -0.16))
back_rail  = Box(0.80, 0.04, 0.12); back_rail.rounding(0.01);  back_rail.move((0, -0.42, -0.16))
left_rail  = Box(0.04, 0.80, 0.12); left_rail.rounding(0.01);  left_rail.move((0.42, 0, -0.16))
right_rail = Box(0.04, 0.80, 0.12); right_rail.rounding(0.01); right_rail.move((-0.42, 0, -0.16))

# ── Union everything ──────────────────────────────────────────────────────────
chair = CombineGeometry("UNION").combine(
    seat, backrest, top_rail, *legs,
    front_rail, back_rail, left_rail, right_rail
)

sdf = chair.create(coor)

# ── Binary field and cross-sections ───────────────────────────────────────────
binary_3d = smarter_reshape(hard_binarization(sdf, 0), CO_RESOLUTION)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].imshow(binary_3d[:, CO_RESOLUTION[1]//2, :].T,
               cmap="Blues", origin="lower",
               extent=(-CO_SIZE[0]/2, CO_SIZE[0]/2, -CO_SIZE[2]/2, CO_SIZE[2]/2))
axes[0].set_title("Side view (XZ)")

axes[1].imshow(binary_3d[CO_RESOLUTION[0]//2, :, :].T,
               cmap="Purples", origin="lower",
               extent=(-CO_SIZE[1]/2, CO_SIZE[1]/2, -CO_SIZE[2]/2, CO_SIZE[2]/2))
axes[1].set_title("Front view (YZ)")
plt.tight_layout(); plt.show()

# ── Surface point cloud ───────────────────────────────────────────────────────
dx = CO_SIZE[0] / CO_RESOLUTION[0]
surface_pts = coor[:, np.abs(sdf) < dx * 1.5].T
np.save("chair_surface.npy", surface_pts)
print(f"Point cloud saved: {surface_pts.shape[0]:,} points")
```

---

## 17. Quick-Reference Cheat Sheet

### Imports

```python
from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Box, Sphere, Cylinder, Cone, Torus
from spomso.cores.geom_2d import Circle, Rectangle
from spomso.cores.combine import CombineGeometry
from spomso.cores.geom import GenericGeometry
```

### Minimal boilerplate

```python
import numpy as np
from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Sphere
from spomso.cores.combine import CombineGeometry

coor, res = generate_grid((2, 2, 2), (60, 60, 60))

s = Sphere(0.5)
sdf = s.create(coor)

binary = smarter_reshape(hard_binarization(sdf, 0), res)
```

### Transformation methods

| Method | Effect |
|---|---|
| `.move((dx, dy, dz))` | translate (accumulates) |
| `.set_location((x, y, z))` | set absolute position |
| `.rotate(angle, axis)` | rotate (accumulates) |
| `.set_rotation(angle, axis)` | set absolute rotation |
| `.rescale(f)` | multiply scale by f |
| `.set_scale(f)` | set absolute scale |

### Modification methods

| Method | Effect |
|---|---|
| `.rounding(r)` | round edges, thickens by r |
| `.rounding_cs(r, max_dim)` | round edges, no thickening |
| `.onion(t)` | hollow to shell of thickness t |
| `.elongation(v)` | stretch along vector v |
| `.extrusion(h)` | extrude 2D shape to height h |
| `.revolution(r)` | revolve 2D shape around y-axis |
| `.twist(pitch)` | twist around z |
| `.bend(r, angle)` | bend around z |
| `.mirror(a, b)` | mirror: copy at a, original at b |
| `.infinite_repetition(d)` | tile all space with spacing d |
| `.finite_repetition(size, reps)` | tile within bounding box |
| `.curve_instancing(...)` | place copies along a curve |

### Combination operations

| String | Operation |
|---|---|
| `"UNION"` | min(a, b, c, …) — n-ary union |
| `"UNION2"` | min(a, b) — binary union |
| `"SUBTRACT2"` | max(a, −b) — a minus b |
| `"INTERSECT2"` | max(a, b) — intersection |
| `"SMOOTH_UNION2"` | smooth union (parametric) |
| `"SMOOTH_SUBTRACT2"` | smooth subtraction (parametric) |
| `"SMOOTH_INTERSECT2"` | smooth intersection (parametric) |
