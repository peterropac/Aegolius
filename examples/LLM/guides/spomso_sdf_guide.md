# SPOMSO – Procedural Geometry with Signed Distance Functions
### A practical guide for generating 3D (and 2D) geometry in Python

**Applies to SPOMSO 1.5.0.**

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
13. [Repetitions and Instancing](#13-repetitions-and-instancing)
14. [Custom SDFs and GenericGeometry](#14-custom-sdfs-and-genericgeometry)
15. [Order-of-Operations Rules](#15-order-of-operations-rules)
16. [Inspecting Objects with `repr`](#16-inspecting-objects-with-repr)
17. [Common Pitfalls](#17-common-pitfalls)
18. [Full Example – A Chair](#18-full-example--a-chair)
19. [Quick-Reference Cheat Sheet](#19-quick-reference-cheat-sheet)

---

## 1. What is SPOMSO?

SPOMSO is a Python library for building **procedural geometry** using **Signed Distance Functions (SDFs)**. Instead of describing shapes with triangles or vertices, every shape is represented as a mathematical field: at any point in space the field value tells you the signed distance to the nearest surface (negative inside the object, positive outside, zero on the surface).

This representation makes it trivial to:
- blend, union, subtract, or intersect shapes with a single `min`/`max` call
- round edges, hollow objects, repeat geometry infinitely, twist or bend shapes — all without mesh surgery
- export clean surface point clouds by thresholding the SDF at zero

Two design principles run through the whole library and explain a lot of its behaviour:

**Space is always 3D.** SPOMSO takes a "3D space first" approach. The coordinate space is assumed to be three-dimensional everywhere, even when you only supply two-dimensional inputs — `generate_grid` always hands back a `(3, N)` point cloud and a three-element resolution. 2D primitives are not a separate world; they are 3D objects that happen to be uniform along z. This is deliberately less restrictive than a strict 2D/3D split: you can mix `Circle` and `Sphere` in the same scene without ceremony.

**Geometry generation is meshless.** An SDF is just a function of position, so there is no mesh, no tessellation, and no requirement that you evaluate it on a grid at all. Grids are simply the most convenient thing to evaluate on, and the built-in helpers are built around rectilinear ones — but nothing stops you from evaluating a shape on a line, a scattered point cloud, or a single probe point.

**Requires:** Python ≥ 3.10
**Key dependencies:** `numpy`, `scipy`
**Optional (for plotting):** `matplotlib`, `plotly`
**Optional (for autodiff):** `jax`, `jaxlib`, `optax`

SPOMSO has two backends:

| Backend | Module | Use for |
|---|---|---|
| NumPy, object-oriented | `spomso.cores.*` | Building and evaluating geometry (this guide) |
| JAX, functional | `spomso.jax_cores.*` | Differentiable geometry, optimisation (see the autodiff guide) |

---

## 2. Installation

Clone or download the [Aegolius repository](https://github.com/peterropac/Aegolius), then install the package from the `spomso` subdirectory:

```bash
pip install -e path/to/Aegolius/spomso          # core only (numpy + scipy)
pip install -e "path/to/Aegolius/spomso[plot]"  # + matplotlib + plotly
pip install -e "path/to/Aegolius/spomso[all]"   # everything incl. JAX + optax
```

> **Note:** as of 1.5.0 the repository layout was flattened — the installable package now sits at `Aegolius/spomso`, not `Aegolius/Code/spomso`. If you are following older instructions, drop the `Code/` path segment.

You can also install SPOMSO directly with `pip`:

```bash
pip install spomso
```

Then in Python to confirm:

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

There is also `.propagate(coor)`, which returns the same values but is intended for feeding one object into the construction of another (this is what `CombineGeometry` uses internally). Use `.create()` in your own code unless you are wrapping an object — see [section 14](#14-custom-sdfs-and-genericgeometry).

---

## 4. The Coordinate Grid

All geometry is evaluated on a regular grid. Two helper functions set this up:

```python
from spomso.cores import generate_grid, smarter_reshape

co_size = (4.0, 4.0, 3.0)       # physical extent (x, y, z)
co_resolution = (100, 100, 75)  # requested number of voxels along each axis

coor, co_res_new = generate_grid(co_size, co_resolution)
# coor       : numpy array, shape (3, N)  — flattened grid of 3D points
# co_res_new : tuple — ACTUAL resolution, forced to odd numbers
```

The grid is always **centred at the origin**, spanning `-size/2` to `+size/2` along each axis.

### The output is always three-dimensional

This is the 3D-first paradigm in practice. Whatever you pass in, `generate_grid` returns a `(3, N)` point cloud and a **three-element** resolution tuple:

```python
generate_grid((4.0,),            (10,))          # -> coor (3, 11),   res (11, 11, 11)
generate_grid((4.0, 4.0),        (20, 20))       # -> coor (3, 441),  res (21, 21, 21)
generate_grid((4.0, 4.0, 4.0),   (10, 10, 10))   # -> coor (3, 1331), res (11, 11, 11)
generate_grid(4.0,               10)             # -> coor (3, 11),   res (11, 11, 11)
```

The number of *points* follows the dimensionality of `size` — a two-element `size` gives you `nx · ny` points lying in the z = 0 plane — but the coordinate array always has three rows, and the resolution tuple always has three entries. A one-element `size` gives a line of points along x.

> `generate_grid` accepts 1D inputs like `generate_grid((4,), (10,))` as of 1.5.0; earlier versions raised a `TypeError`.

### `generate_grid` always returns odd resolutions

`generate_grid` bumps every even resolution up by one so that each axis has an **odd** number of points. This guarantees a grid point lands exactly on the origin, which is convenient for symmetric geometry — but it means `co_res_new` is not what you asked for. `(100, 100, 75)` becomes `(101, 101, 75)`.

**Always use the returned `co_res_new` for indexing and slicing**, not your original request.

### Reshaping back to a grid

```python
field_3d = smarter_reshape(sdf_values, co_res_new)
# field_3d : numpy array, shape (nx, ny, nz)
```

`smarter_reshape` applies the odd-conversion internally, so passing either your original request or `co_res_new` works for a full 3D grid.

### Reshaping a grid with fewer than three extents

Since the resolution tuple always has three entries but a 2D-extent grid only has `nx · ny` points, `smarter_reshape` needs a resolution whose length matches the number of extents you actually populated:

```python
coor, co_res_new = generate_grid((8.0, 8.0), (400, 400))
print(co_res_new)                            # (401, 401, 401) — always three entries

smarter_reshape(sdf_values, co_res_new)      # ValueError — expects 401³ points
smarter_reshape(sdf_values, co_res_new[:2])  # correct -> (401, 401)
smarter_reshape(sdf_values, (400, 400))      # also fine -> (401, 401)
```

So: slice to `co_res_new[:2]` when your `size` had two elements. The same rule applies to `vector_smarter_reshape`, `nd_vector_smarter_reshape`, and `VectorFieldFromSDF`. For a 1D grid pass a plain integer.

If you would rather not think about this at all, give the grid a genuine z extent — even a thin one such as `co_size = (8.0, 8.0, 0.5)` with `co_resolution = (200, 200, 5)`. You then get a full 3-element resolution that works everywhere, three-component vector fields, and the freedom to slice whichever plane you like. This is what the shipped examples in `examples/vector/` do, and it fits the 3D-first paradigm rather than fighting it.

**Practical tip:** Start with a coarse resolution (e.g. 50³) for fast iteration, then increase to 150³ or more for final outputs. Very coarse grids produce jagged surfaces.

### `co` does not have to be a grid

Because an SDF is a pure function of position, `.create()` accepts **any** array of shape `(3, N)`. Nothing about it is grid-specific:

```python
sphere = Sphere(1.0)

# a line of probe points
line = np.stack([np.linspace(-2, 2, 7), np.zeros(7), np.zeros(7)])
sphere.create(line)        # -> [ 1.  0.33 -0.33 -1.  -0.33  0.33  1. ]

# scattered points
sphere.create(np.random.default_rng(0).uniform(-2, 2, (3, 500)))

# a single point
sphere.create(np.asarray([[0.5], [0.0], [0.0]]))    # -> [-0.5]
```

This is useful well beyond curiosity: evaluating an objective on a handful of probe points or a monitor line instead of a full grid is dramatically cheaper, which matters a lot for optimisation (see the autodiff guide). Grids are a convenience, not a requirement.

---

## 5. Primitive Shapes

All shapes live in `spomso.cores.geom_3d` (3D) or `spomso.cores.geom_2d` (2D). Most are also re-exported from `spomso.cores` directly.

```python
from spomso.cores.geom_3d import (
    Box, Sphere, Cylinder, Cone, Torus,
    SegmentedLine3D, ParametricCurve3D,
    InfiniteCylinder, Plane,
)
```

### 3D primitives

| Class | Constructor arguments | Notes |
|---|---|---|
| `Box(a, b, c)` | full side lengths along x, y, z | axis-aligned, centred at origin |
| `Sphere(radius)` | radius | centred at origin |
| `Cylinder(radius, height)` | radius, height | axis along z, centred |
| `Cone(height, angle)` | height, slope angle (rad) | tip up along +z; positioned by centre of mass (see below) |
| `InfiniteCone(angle)` | slope angle | infinite cone |
| `OrientedInfiniteCone(angle)` | slope angle | one-sided infinite cone |
| `SolidAngle(radius, angle_1, angle_2)` | radius, two angles | spherical wedge |
| `Torus(primary_radius, secondary_radius)` | ring radius, tube radius | **ring lies in the XY plane**, axis along z |
| `ChainLink(primary_radius, secondary_radius, length)` | ring radius, tube radius, length | elongated torus |
| `Braid(length, primary_radius, secondary_radius, pitch)` | length, radii, pitch | twisted double helix |
| `Arc3D(radius, thickness, start_angle, end_angle)` | radius, tube thickness, angles | partial torus |
| `Plane(normal, thickness)` | normal vector, total thickness | infinite slab, centred on the origin plane |
| `OrientedPlane(normal, offset)` | normal vector, offset | half-space |
| `InfiniteCylinder(radius)` | radius | infinite along z |
| `Line(a, b)` | two 3D points | infinite line through a and b |
| `Triangle3D(a, b, c)` | three 3D points | flat triangle |
| `Quad(a, b, c, d)` | four 3D points | flat quad |
| `SegmentedLine3D(points, closed=False)` | list/array of 3D points | polyline / wire |
| `SegmentedParametricCurve3D(points, t_range, closed=False)` | points, t range | interpolated curve |
| `ParametricCurve3D(func, params, t_range, closed=False)` | function, params, t range | smooth curve |
| `PointCloud3D(points)` | array of 3D points | distance to a point set |
| `X(offset)`, `Y(offset)`, `Z(offset)` | offset | signed distance to an axis-aligned plane |

```python
box      = Box(1.0, 0.5, 0.25)     # 1 × 0.5 × 0.25 box (FULL side lengths)
sphere   = Sphere(0.4)
cylinder = Cylinder(0.2, 1.0)      # radius 0.2, height 1.0
torus    = Torus(0.8, 0.15)        # ring radius 0.8 in the XY plane, tube radius 0.15
cone     = Cone(1.2, np.pi/8)      # height 1.2, slope angle 22.5°
```

> As of 1.5.0, `SolidAngle` and `PointCloud3D` are re-exported from `spomso.cores` as well, so every primitive in the table is reachable from either `spomso.cores` or `spomso.cores.geom_3d`.

### `Box` side lengths are full extents

`Box(a, b, c)` has always produced an `a × b × c` box, but before 1.4.0 the read-back properties `.a`, `.b`, `.c` returned *half* the value. As of 1.4.0 they return the full side lengths:

```python
Box(2, 4, 6).a   # -> 2   (was 1.0 in 1.3.0)
```

The same fix applies to `Rectangle.a/.b/.size` and `RoundedRectangle.a/.b/.size`.

### `Cone` is positioned by its centre of volume

Unlike `Box`, `Sphere`, and `Cylinder` — which are centred on their bounding boxes — `Cone` is placed relative to its centre of volume. The reference height is exposed as the `.height_offset` property:

```
height_offset = height · 2^(-1/3) ≈ 0.7937 · height
```

The tip sits at `z = +height_offset` and the base at `z = height_offset - height ≈ -0.206 · height`. So the cone is *not* symmetric about `z = 0`, and its bounding box is not centred there.

If you want the base at `z = 0`, translate by `height - height_offset`:

```python
cone = Cone(1.0, np.pi/8)
print(cone.height_offset)                      # 0.7937005...
cone.move((0, 0, 1.0 - cone.height_offset))    # base now at z = 0
```

---

## 6. Euclidean Transformations

Every geometry object inherits from `EuclideanTransform`. Transformations are applied **after** all modifications (see [section 15](#15-order-of-operations-rules) for why this matters).

### Translation

```python
shape.move((0.5, 0.0, -1.0))          # relative — accumulates
shape.set_location((0.5, 0.0, -1.0))  # absolute — overrides previous
```

### Rotation

```python
shape.rotate(np.pi/4, (0, 0, 1))        # 45° around the z-axis
shape.rotate(some_3x3_matrix)           # or pass a rotation matrix
shape.rotate_rotvec(np.pi/4, (0, 0, 1)) # explicit axis-angle form
shape.rotate_matrix(some_3x3_matrix)    # explicit matrix form
shape.set_rotation(np.pi/2, (1, 0, 0))  # absolute — 90° around x
```

`rotate` dispatches on its arguments: `(angle, axis)` or a single `(3, 3)` matrix.

### Scaling

```python
shape.rescale(2.0)         # multiply existing scale by 2
shape.set_scale(1.5)       # set absolute scale to 1.5
```

Scaling is **isotropic** (uniform in all directions). For anisotropic stretching use `.elongation()`.

### Inspecting the current state

```python
print(shape.center)            # current position
print(shape.scale)             # current scale factor
print(shape.rotation_matrix)   # current 3×3 rotation matrix
print(shape.rotation_angle)    # current rotation angle
print(shape.rotation_axis)     # current rotation axis
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

shape.rounding_cs(0.05, bb_size)   # round without growing past the bounding box;
                                   # bb_size = largest dimension of the object
```

### Hollowing / shells

```python
shape.onion(0.02)       # keeps only a shell of thickness 0.02
shape.concentric(0.02)  # turns an isosurface into two surfaces 0.02 apart
shape.boundary()        # absolute value of the SDF — the surface itself
```

### Elongation

```python
shape.elongation((0.5, 0.0, 0.0))  # stretch by 0.5 along x
```

> **Changed in 1.4.0:** `elongation` now stretches by the **full** vector length in each direction, and the NumPy and JAX backends agree. If you are porting code from 1.3.0 that relied on the old half-length behaviour, halve your vectors.

### Extrusion and revolution (2D → 3D)

```python
circle.extrusion(1.0)          # extrude a 2D shape to a total height of 1.0
profile.revolution(0.5)        # revolve around the y-axis at radius 0.5
profile.axis_revolution(0.5, np.pi/6)  # revolve around an axis tilted from y
```

`revolution` first translates the 2D shape along x by `radius`, then sweeps it around the **y-axis**.

### Twist, bend, shear

```python
shape.twist(np.pi)                  # rad per unit length, around z
shape.bend(1.0, np.pi/2)            # bend radius, bend angle, around z
shape.shear_xz(np.pi/6)             # named shears: xz, yz, xy, zy, yx, zx
shape.shear(np.pi/6, 0, 2)          # generic: (angle, sheared_axis, fixed_axis)
```

### Symmetry and mirroring

```python
shape.mirror((-1, 0, 0), (1, 0, 0))       # mirror image at a, original at b
shape.symmetry(0)                          # mirror across the x = 0 plane
shape.rotational_symmetry(6, 0.5, 0.0)     # 6-fold pattern at radius 0.5, phase 0
```

### Displacement

```python
def bumps(co, freq, amp):
    return amp * np.sin(freq * co[0]) * np.sin(freq * co[1])

shape.displacement(bumps, (10.0, 0.02))   # perturb the surface; apply last
```

### Volume definition (for unsigned fields)

```python
shape.signed(co_res_new)             # convert an unsigned distance field to signed
interior = shape.sign()              # capture the inside/outside mask
shape.recover_volume(interior)       # restore a volume after operations that lose it
shape.define_volume(my_interior_fn, params)  # define inside/outside explicitly
shape.invert()                       # flip inside and outside
```

> `ModifyObject.signed_old` was **removed in 1.5.0**. Use `signed()` — same algorithm, faster, better boundary handling, drop-in replacement.

### Post-processing as a modification

Every post-processing function is also available as a chained modification, so the field is transformed inside the object rather than afterwards:

```python
shape.gaussian_falloff(1.0, 0.2)
shape.hard_binarization(0.0)
shape.sigmoid_falloff(1.0, 0.3)
shape.relu(1.0)
shape.conv_averaging(5, 2, co_res_new)   # needs the grid resolution
```

### Custom modifications

```python
def squash_z(sdf, co, sdf_params, params):
    factor, = params
    co = co.copy()
    co[2] *= factor
    return sdf(co, *sdf_params)

shape.custom_modification(squash_z, (2.0,), "squash_z")
shape.custom_post_process(lambda u, k: np.tanh(k * u), (3.0,), "tanh")
```

The custom modification receives `(sdf_function, coordinates, sdf_parameters, your_parameters)`. Passing a name makes it show up in `.modifications` and in the object's `repr`.

### Low-level SDF-space transforms

`move_sdf`, `rotate_sdf`, and `scale_sdf` apply a transform at the **modification** stage rather than the transformation stage. This is the clean way to control operation order — see [section 15](#15-order-of-operations-rules).

```python
shape.move_sdf((0.2, 0, 0))
shape.rotate_sdf(rotation_matrix)
shape.scale_sdf(2.0)
```

### Inspecting applied modifications

```python
print(shape.modifications)   # list of modification names, in order
```

> **Fixed in 1.4.0:** four methods used to report a different label than the method you called. `axis_revolution`, `curve_instancing`, `aligned_curve_instancing`, and `fully_aligned_curve_instancing` now appear under their own names.

---

## 8. Combining Shapes

```python
from spomso.cores import CombineGeometry
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

| Operation string | Arity | Meaning |
|---|---|---|
| `"UNION"` | n-ary | `min(a, b, c, …)` |
| `"UNION2"` | binary | `min(a, b)` |
| `"INTERSECT"` | n-ary | `max(a, b, c, …)` |
| `"INTERSECT2"` | binary | `max(a, b)` |
| `"SUBTRACT2"` | binary | `max(a, −b)` — a minus b |
| `"SUM"` | binary | `a + b` |
| `"DIFFERENCE"` | binary | `a − b` |

`"SUM"` and `"DIFFERENCE"` are arithmetic on the field, not CSG — useful for building potentials and blends, but the result is no longer a true distance field.

### Smooth (parametric) operations

These blend between shapes rather than producing a sharp seam. Note that `parameters` is **keyword-only**:

```python
smooth_union = CombineGeometry("SMOOTH_UNION2")
result = smooth_union.combine_parametric(shape_a, shape_b, parameters=0.15)
# parameters = smoothing width (larger -> more blending)
```

| Operation string | Meaning |
|---|---|
| `"SMOOTH_UNION2"` | smooth union, poly3 kernel |
| `"SMOOTH_UNION2_2"` | smooth union, poly2 kernel |
| `"SMOOTH_INTERSECT2"` | smooth intersection, poly3 |
| `"SMOOTH_INTERSECT2_BOLTZMANN"` | smooth intersection, Boltzmann smooth-max |
| `"SMOOTH_SUBTRACT2"` | smooth subtraction, poly3 |
| `"SMOOTH_SUBTRACT2_BOLTZMANN"` | smooth subtraction, Boltzmann smooth-max |

You can discover these at runtime:

```python
CombineGeometry("UNION").available_operations
CombineGeometry("SMOOTH_UNION2").available_parametric_operations
```

> **Fixed in 1.4.0/1.5.0:** `smoothmin_poly2` and `smoothmin_poly3` had divide-by-zero and typo bugs at `a = 0`, and the NumPy and JAX backends disagreed slightly. Both now handle `a = 0` and negative `a` consistently and agree to floating-point precision.

### Reusing a combiner

```python
union = CombineGeometry("UNION2")
ab    = union.combine(a, b)
abc   = union.combine(ab, c)   # chain as many times as needed
```

### The result is a `GenericGeometry`

The object returned by `.combine()` or `.combine_parametric()` behaves exactly like a primitive: apply further modifications, transformations, and combine it again.

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

binary = hard_binarization(sdf_values, 0)
# binary : numpy array of 0.0 / 1.0, same shape as sdf_values
```

All post-processing functions take the field as their first argument:

| Function | Effect |
|---|---|
| `hard_binarization(u, threshold)` | step function at threshold |
| `linear_falloff(u, amplitude, width)` | linear ramp at the surface |
| `gaussian_falloff(u, amplitude, width)` | Gaussian decay from the surface |
| `gaussian_boundary(u, amplitude, width)` | Gaussian bump centred on the surface |
| `sigmoid_falloff(u, amplitude, width)` | smooth step |
| `positive_sigmoid_falloff(u, amplitude, width)` | one-sided smooth step |
| `capped_exponential(u, amplitude, width)` | exponential capped at the amplitude |
| `relu(u, width=1)` | one-sided linear ramp |
| `smooth_relu(u, smooth_width, width=1, threshold=0.01)` | smoothed ReLU |
| `slowstart(u, smooth_width, width=1, threshold=0.01, ground=True)` | slow-onset ramp |
| `conv_averaging(u, kernel_size, iterations)` | spatial smoothing — **requires a reshaped grid** |
| `conv_edge_detection(u)` | edge detection — **requires a reshaped grid** |
| `custom_post_process(u, function, parameters)` | apply your own function |

The two convolution functions operate on the **grid**, not the flat array — reshape first:

```python
field_3d = smarter_reshape(sdf_values, co_res_new)
smoothed = conv_averaging(field_3d, (5, 5, 1), 1)
```

### The `PostProcess` class

For chaining several operations, `PostProcess` mirrors the modification API. Note that it wraps an **SDF function**, not an evaluated array — pass `obj.propagate`, and wrap the result back into a `GenericGeometry` to evaluate it:

```python
from spomso.cores.post_processing import PostProcess
from spomso.cores.geom import GenericGeometry

pp = PostProcess(my_shape.propagate)    # a callable, not sdf_values
pp.gaussian_falloff(1.0, 0.2)
pp.hard_binarization(0.5)

processed = GenericGeometry(pp.processed_object, ())
values = processed.create(coor)

print(pp.post_processing_operations)    # ['gaussian_falloff', 'hard_binarization']
```

`pp.processed_object` (equivalently `pp.processed_geo_object`) is the composed callable; `pp.unprocessed_object` is the original function, untouched. If you only need one operation on an already-evaluated field, call the plain function from the table above instead — it is simpler.

> **Fixed in 1.4.0:** `sigmoid_falloff` as a *modification* raised `AttributeError` on every call in 1.3.0. It works now.

---

## 10. Extracting a Surface Point Cloud

The surface of any shape is the zero level-set of its SDF. Extract it by keeping grid points within a small distance of zero:

```python
dx = co_size[0] / co_res_new[0]       # one voxel step
surface_mask = np.abs(sdf_values) < dx * 1.5
surface_pts  = coor[:, surface_mask].T  # shape (N_surface, 3)

np.save("my_shape.npy", surface_pts)
```

The point cloud density depends on grid resolution: a 100³ grid over a 2-unit domain gives roughly 0.02-unit spacing.

For interior (solid) points rather than the surface, `GenericGeometry.point_cloud(coor)` returns the points where the SDF is `≤ 0`. Note that it zeroes the third row, so it is only useful for 2D geometry.

---

## 11. Visualising Results

### Cross-section slice (matplotlib)

```python
import matplotlib.pyplot as plt
from spomso.cores import smarter_reshape
from spomso.cores.post_processing import hard_binarization

binary_3d = smarter_reshape(hard_binarization(sdf_values, 0), co_res_new)

fig, ax = plt.subplots()
# XZ mid-plane (y = 0)
ax.imshow(
    binary_3d[:, co_res_new[1] // 2, :].T,
    cmap="binary_r", origin="lower",
    extent=(-co_size[0] / 2, co_size[0] / 2, -co_size[2] / 2, co_size[2] / 2)
)
ax.set_xlabel("x")
ax.set_ylabel("z")
plt.show()
```

Swap the slice axis for other planes:
- XY (top view): `binary_3d[:, :, co_res_new[2] // 2]`
- YZ (front view): `binary_3d[co_res_new[0] // 2, :, :]`

Always `.T` the 2D slice before `imshow`, since SPOMSO grids are indexed `[x, y, z]` while `imshow` expects `[row, column]`.

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

### 3D scatter of the surface point cloud

```python
fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(surface_pts[:, 0], surface_pts[:, 1], surface_pts[:, 2],
           c=surface_pts[:, 2], cmap="plasma", s=0.5)
plt.show()
```

---

## 12. 2D Geometry

All 2D primitives live in `spomso.cores.geom_2d`. They work identically to the 3D ones but operate in the XY plane. Use them to:
- visualise a cross-section quickly on a 2D grid
- create a 2D profile and then **extrude** or **revolve** it into 3D

```python
from spomso.cores.geom_2d import Circle, Rectangle, Polygon, Arc, Sector

circle = Circle(0.5)
rect   = Rectangle(1.0, 0.6)
```

### 2D primitives

| Class | Constructor arguments |
|---|---|
| `Circle(radius)` | radius |
| `NEUCircle(radius, order)` | radius, superellipse order |
| `NGon(radius, n_sides)` | circumradius, number of sides |
| `Polygon(vertices)` | array of 2D vertices |
| `Rectangle(a, b)` | full side lengths |
| `RoundedRectangle(a, b, rounding)` | side lengths, up to 4 corner radii |
| `Segment(a, b)` | two endpoints |
| `Triangle(a, b, c)` | three vertices |
| `Sector(radius, angle_1, angle_2)` | radius, two angles |
| `InfiniteSector(angle_1, angle_2)` | two angles |
| `Arc(radius, start_angle, end_angle)` | radius, two angles |
| `ParametricCurve(func, params, t_range, closed=False)` | function, params, t range |
| `SegmentedParametricCurve(points, t_range, closed=False)` | points, t range |
| `SegmentedLine(points, closed=False)` | polyline points |
| `PointCloud2D(points)` | array of 2D points |

Extrude or revolve into 3D:

```python
circle.extrusion(2.0)   # becomes a cylinder of height 2
rect.revolution(1.0)    # becomes a torus-like surface of revolution
```

Because of the 3D-first paradigm, a 2D primitive evaluated on a 3D grid is simply **extended along z** — `Circle(1.0)` on a 3D grid behaves as an infinite cylinder. Use `.extrusion()` if you want a finite height. And remember the [reshape rule](#reshaping-a-grid-with-fewer-than-three-extents): if your `size` had two elements, reshape with `co_res_new[:2]`.

At the function level, `sdf_closed_segmented_line_2d` was added to `spomso.cores.sdf_2D` in 1.5.0 (the JAX side already had it), so closed polylines are now available from both backends.

---

## 13. Repetitions and Instancing

### Finite repetition

```python
pillar = Cylinder(0.05, 1.0)
pillar.finite_repetition(
    size=(2.0, 2.0, 0.0),     # bounding box in which to repeat
    repetitions=(5, 5, 1)     # 5×5 grid of pillars
)
```

`finite_repetition_rescaled(size, repetitions, instance_size, padding)` additionally scales each instance to fit its cell.

### Infinite repetition

```python
sphere = Sphere(0.1)
sphere.infinite_repetition((0.4, 0.4, 0.4))   # copy every 0.4 units
```

### Linear instancing

```python
rung = Box(0.4, 0.05, 0.05)
rung.linear_instancing(10, (0, 0, -1), (0, 0, 1))   # 10 copies along a segment
```

### Curve instancing

Place copies along a parametric curve, optionally aligning each copy to the curve's frame:

```python
def helix(t, R, H, freq):
    return np.asarray([R*np.cos(2*np.pi*freq*t),
                       R*np.sin(2*np.pi*freq*t),
                       H*t - H/2])

box = Box(0.3, 0.15, 0.1)

# Simple placement (no alignment)
box.curve_instancing(helix, (1.0, 2.0, 3.0), (0, 1, 20))

# Align the instance x-axis to the curve tangent
box.aligned_curve_instancing(helix, (1.0, 2.0, 3.0), (0, 1, 20))

# Align x/y/z to tangent/normal/binormal (full Frenet frame)
box.fully_aligned_curve_instancing(helix, (1.0, 2.0, 3.0), (0, 1, 20))
```

The second argument is passed as `*params` to the curve function; the third is `(t_start, t_end, n_instances)`.

---

## 14. Custom SDFs and GenericGeometry

Wrap any callable `f(co, *params) -> np.ndarray` as a first-class geometry object:

```python
from spomso.cores.geom import GenericGeometry


def my_sdf(co, radius, twist_rate):
    # co has shape (3, N)
    angle = twist_rate * co[2]
    x_rot = co[0] * np.cos(angle) - co[1] * np.sin(angle)
    y_rot = co[0] * np.sin(angle) + co[1] * np.cos(angle)
    r = np.sqrt(x_rot ** 2 + y_rot ** 2)
    return r - radius


shape = GenericGeometry(my_sdf, 0.5, np.pi)
shape.move((0, 0, 0.5))
sdf_values = shape.create(coor)
```

Write to a **copy** of `co` if you need to mutate it — writing in place corrupts the caller's grid.

This lets you define any SDF — procedural noise, imported data, physics-derived surfaces — and use it seamlessly with the rest of the API.

> `GenericGeometry2D` and `GenericGeometry3D` were **removed in 1.5.0**. They were unused duplicates; use `GenericGeometry` for both 2D and 3D.

### Wrapping an already-transformed object

Passing `obj.propagate` to `GenericGeometry` "freezes" everything applied so far, so new modifications run *after* the old transformations:

```python
box = Box(1, 1, 1)
box.rotate(np.pi/4, (0, 0, 1))

box_frozen = GenericGeometry(box.propagate)
box_frozen.mirror((0, -1, 0), (0, 1, 0))   # this mirror runs AFTER the rotation
```

### Functional SDFs take a size vector

If you call the raw SDF functions rather than the classes, note the 1.4.0 signature changes:

```python
from spomso.cores.sdf_3D import sdf_box
from spomso.cores.sdf_2D import sdf_box_2d, sdf_rounded_box_2d

sdf_box(co, (1, 2, 3))        # single vector of FULL side lengths
sdf_box_2d(co, (1, 2))        # size means full extent (was half-extent in 1.3.0)
```

As of 1.5.0 the SDF functions coerce their vector arguments, so tuples, lists, NumPy arrays, and JAX arrays are all accepted interchangeably — in both backends. Earlier versions rejected some combinations.

`sdf_braid` takes `(co, R, r, length, pitch)` — the order was aligned in 1.5.0 with `sdf_chainlink`, from which it derives. The `Braid` class is unaffected.

---

## 15. Order-of-Operations Rules

This is the most important rule to internalise:

> **Modifications always run before Euclidean transformations, regardless of call order.**

```python
box = Box(1, 1, 1)
box.rotate(np.pi/4, (0, 0, 1))    # called first, but applied second
box.mirror((-1, 0, 0), (1, 0, 0)) # called second, but applied first
```

The mirror is applied to the unrotated box, and *then* the result is rotated. For mirror, symmetry, repetition, and instancing operations — anything that depends on world-space position — the order relative to rotation changes the result. Modifications that only reshape the field locally (rounding, onion, twist, bend) are unaffected.

### Two ways to control the order

**Option A — `move_sdf` / `rotate_sdf` / `scale_sdf`.** These apply a transform at the *modification* stage, so it participates in the modification ordering:

```python
box = Box(1, 1, 1)
box.move_sdf((0.5, 0, 0))          # runs in modification order...
box.mirror((-1, 0, 0), (1, 0, 0))  # ...before this mirror
box.rotate(np.pi/4, (0, 0, 1))     # true transformation, still runs last
```

**Option B — freeze with `GenericGeometry`.** Wrap the object to bake in everything so far, then keep building:

```python
box_frozen = GenericGeometry(box.propagate)
box_frozen.mirror((0, -1, 0), (0, 1, 0))   # runs after box's transformations
```

Option A is usually cleaner for a single reorder; Option B is better when you need a genuine two-stage pipeline.

---

## 16. Inspecting Objects with `repr`

**New in 1.5.0:** every geometry object, `Points` object, and vector field prints its constructor arguments plus the modification and transformation chain. This makes debugging long pipelines much easier.

```python
>>> Box(2, 4, 6)
Box(a=2, b=4, c=6)

>>> t = Torus(1.0, 0.2)
>>> t.twist(0.5)
>>> t.rotate(np.pi/2, (1, 0, 0))
>>> t
Torus(primary_radius=1.0, secondary_radius=0.2)  Modifications: [twist]  Transformations: [rotate]
```

Long arrays and callables are abbreviated rather than dumped in full:

```python
>>> Points(np.zeros((3, 100)))
Points(points=ndarray(shape=(3, 100), dtype=float64))
```

Custom modifications registered with a name appear in the chain automatically.

One caveat: an object returned by `CombineGeometry.combine()` is a bare `GenericGeometry` wrapping an anonymous function, so its `repr` shows the wrapper rather than the shapes that went into it:

```python
>>> CombineGeometry("UNION").combine(seat, backrest)
GenericGeometry(geo_sdf=<new_geo_object>, geo_parameters=((),))
```

Print the individual components before combining if you need to inspect them.

---

## 17. Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Cannot reshape the pattern` | resolution length doesn't match the number of extents in `size` | pass `co_res_new[:2]` for a 2D-extent grid, or give the grid a z extent |
| Slices look transposed | grids are indexed `[x, y, z]` | `.T` the slice before `imshow` |
| Mirror/repetition in the wrong place | modifications run before transformations | use `move_sdf` or freeze with `GenericGeometry` |
| `Cone` sits lower than expected | it is placed by centre of mass, not bounding box | offset by `height - cone.height_offset` |
| A 2D primitive fills the whole z range | 3D-first: 2D shapes extend along z | apply `.extrusion(h)` |
| Torus in the wrong plane | the ring lies in **XY**, axis along z | rotate 90° about x for an XZ ring |
| `conv_averaging` raises a dimension error | it needs a grid, not a flat array | `smarter_reshape` first |
| Coordinate grid becomes corrupted | a custom SDF wrote into `co` in place | `co = co.copy()` first |
| Box read-back values look halved | you are on 1.3.0 | upgrade; `.a/.b/.c` return full extents from 1.4.0 |
| Elongation is half what you expect | 1.3.0 behaviour | upgrade; `elongation` uses full vector length from 1.4.0 |
| `sigmoid_falloff` raises `AttributeError` | 1.3.0 bug in the modification version | upgrade to 1.4.0+ |

---

## 18. Full Example – A Chair

A complete, self-contained script that builds a four-legged chair and outputs cross-section images plus a surface point cloud.

```python
import numpy as np
import matplotlib.pyplot as plt

from spomso.cores import generate_grid, smarter_reshape, CombineGeometry
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Box, Cylinder

# -- Grid ---------------------------------------------------------------------
CO_SIZE = (1.2, 1.2, 1.8)
CO_RESOLUTION = (80, 80, 120)
coor, co_res = generate_grid(CO_SIZE, CO_RESOLUTION)   # co_res is (81, 81, 121)

# -- Seat ---------------------------------------------------------------------
seat = Box(0.90, 0.90, 0.10)
seat.rounding(0.04)
seat.move((0.0, 0.0, -0.05))          # top surface at z = 0

# -- Back rest ----------------------------------------------------------------
backrest = Box(0.84, 0.06, 0.65)
backrest.rounding(0.04)
backrest.move((0.0, -0.42, 0.325))
backrest.rotate(np.deg2rad(5), (1, 0, 0))   # tilt back 5°

top_rail = Box(0.84, 0.08, 0.06)
top_rail.rounding(0.03)
top_rail.move((0.0, -0.42, 0.62))
top_rail.rotate(np.deg2rad(5), (1, 0, 0))

# -- Legs ---------------------------------------------------------------------
leg_positions = [(0.42, 0.42, -0.475), (0.42, -0.42, -0.475),
                 (-0.42, 0.42, -0.475), (-0.42, -0.42, -0.475)]
legs = []
for pos in leg_positions:
    leg = Cylinder(0.045, 0.75)
    leg.rounding(0.01)
    leg.move(pos)
    legs.append(leg)

# -- Support rails under the seat ---------------------------------------------
rails = []
for size, pos in [((0.80, 0.04, 0.12), (0.0, 0.42, -0.16)),
                  ((0.80, 0.04, 0.12), (0.0, -0.42, -0.16)),
                  ((0.04, 0.80, 0.12), (0.42, 0.0, -0.16)),
                  ((0.04, 0.80, 0.12), (-0.42, 0.0, -0.16))]:
    rail = Box(*size)
    rail.rounding(0.01)
    rail.move(pos)
    rails.append(rail)

# -- Union everything ---------------------------------------------------------
chair = CombineGeometry("UNION").combine(
    seat, backrest, top_rail, *legs, *rails
)

sdf = chair.create(coor)

# -- Binary field and cross-sections ------------------------------------------
binary_3d = smarter_reshape(hard_binarization(sdf, 0), co_res)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].imshow(binary_3d[:, co_res[1] // 2, :].T,
               cmap="Blues", origin="lower",
               extent=(-CO_SIZE[0] / 2, CO_SIZE[0] / 2,
                       -CO_SIZE[2] / 2, CO_SIZE[2] / 2))
axes[0].set_title("Side view (XZ)")

axes[1].imshow(binary_3d[co_res[0] // 2, :, :].T,
               cmap="Purples", origin="lower",
               extent=(-CO_SIZE[1] / 2, CO_SIZE[1] / 2,
                       -CO_SIZE[2] / 2, CO_SIZE[2] / 2))
axes[1].set_title("Front view (YZ)")

plt.tight_layout()
plt.show()

# -- Surface point cloud ------------------------------------------------------
dx = CO_SIZE[0] / co_res[0]
surface_pts = coor[:, np.abs(sdf) < dx * 1.5].T
np.save("chair_surface.npy", surface_pts)
print(f"Point cloud saved: {surface_pts.shape[0]:,} points")
```

---

## 19. Quick-Reference Cheat Sheet

### Imports

```python
from spomso.cores import generate_grid, smarter_reshape, CombineGeometry
from spomso.cores import GenericGeometry, Points, VectorField
from spomso.cores import Box, Sphere, Cylinder, Cone, Torus, SolidAngle, PointCloud3D
from spomso.cores import Circle, Rectangle
from spomso.cores.post_processing import hard_binarization, PostProcess
```

As of 1.5.0 essentially the whole public surface — including `VectorField`, `SolidAngle`, and `PointCloud3D` — is reachable straight from `spomso.cores`. The submodule paths (`spomso.cores.geom_3d`, `spomso.cores.geom_2d`, `spomso.cores.geom`) still work and are useful when you want to be explicit about where something comes from.

### Minimal boilerplate

```python
import numpy as np
from spomso.cores import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Sphere

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
| `.rotate(angle, axis)` or `.rotate(matrix)` | rotate (accumulates) |
| `.rotate_rotvec(angle, axis)` | rotate, explicit axis-angle |
| `.rotate_matrix(matrix)` | rotate, explicit matrix |
| `.set_rotation(angle, axis)` | set absolute rotation |
| `.rescale(f)` | multiply scale by f |
| `.set_scale(f)` | set absolute scale |

### Modification methods

| Method | Effect |
|---|---|
| `.rounding(r)` | round edges, thickens by r |
| `.rounding_cs(r, bb_size)` | round edges, stay inside bounding box |
| `.onion(t)` | hollow to a shell of thickness t |
| `.concentric(w)` | two surfaces w apart |
| `.boundary()` | absolute value of the SDF |
| `.elongation(v)` | stretch by the full length of v |
| `.extrusion(h)` | extrude 2D shape to height h |
| `.revolution(r)` | revolve 2D shape around the y-axis |
| `.axis_revolution(r, angle)` | revolve around a tilted axis |
| `.twist(pitch)` | twist around z, rad per unit length |
| `.bend(r, angle)` | bend around z |
| `.shear_xz(angle)` (and xy, yz, zy, yx, zx) | named shear |
| `.shear(angle, sheared_axis, fixed_axis)` | generic shear |
| `.mirror(a, b)` | mirror image at a, original at b |
| `.symmetry(axis)` | mirror across an axis plane |
| `.rotational_symmetry(n, radius, phase)` | n-fold circular pattern |
| `.infinite_repetition(d)` | tile all space with spacing d |
| `.finite_repetition(size, reps)` | tile within a bounding box |
| `.finite_repetition_rescaled(size, reps, instance_size, padding)` | tile and rescale |
| `.linear_instancing(n, a, b)` | n copies along a segment |
| `.curve_instancing(f, params, t_range)` | copies along a curve |
| `.aligned_curve_instancing(...)` | copies aligned to the tangent |
| `.fully_aligned_curve_instancing(...)` | copies aligned to the Frenet frame |
| `.displacement(fn, params)` | perturb the surface |
| `.signed(co_res)` | unsigned → signed distance field |
| `.invert()` / `.sign()` | flip / extract inside-outside |
| `.define_volume(fn, params)` / `.recover_volume(fn)` | manage the interior |
| `.move_sdf(v)` / `.rotate_sdf(R)` / `.scale_sdf(s)` | transform at modification stage |
| `.custom_modification(fn, params, name)` | your own modification |
| `.custom_post_process(fn, params, name)` | your own post-process |

### Combination operations

| String | Operation |
|---|---|
| `"UNION"` / `"UNION2"` | n-ary / binary union |
| `"INTERSECT"` / `"INTERSECT2"` | n-ary / binary intersection |
| `"SUBTRACT2"` | a minus b |
| `"SUM"` / `"DIFFERENCE"` | field addition / subtraction |
| `"SMOOTH_UNION2"` / `"SMOOTH_UNION2_2"` | smooth union, poly3 / poly2 |
| `"SMOOTH_INTERSECT2"` / `"…_BOLTZMANN"` | smooth intersection |
| `"SMOOTH_SUBTRACT2"` / `"…_BOLTZMANN"` | smooth subtraction |

Non-parametric → `.combine(*objects)`. Parametric → `.combine_parametric(*objects, parameters=width)`.
