![](Files/Images/comb_230623_2.png#gh-dark-mode-only)
![](Files/Images/comb_220224_0.png#gh-light-mode-only)

**Aegolius** is the public repository for the Python package **SPOMSO**.

**SPOMSO** is free and open-source software distributed under the [GNU LGPL](https://www.gnu.org/licenses/lgpl-3.0.html). This package is designed for the procedural construction of geometry and vector fields based on [Signed Distance Functions](https://en.wikipedia.org/wiki/Signed_distance_function) (SDFs).

## Key Features - SDFs

-   Support for both **2D** and **3D** geometry.
-   Object-oriented and functional approaches to defining geometry with pre-defined 2D and 3D objects.
-   Built-in [Euclidean Transformations](https://en.wikipedia.org/wiki/Rigid_transformation) (translation, rotation, scaling).
-   Seamless conversion between point clouds and SDFs.
-   Over 50 built-in SDF modifications, including:
    *   Extrusion, revolution, twist, bend, and elongation.
    *   Mirroring, symmetry, and rotational symmetry.
    *   Finite and infinite instancing (along lines, segmented lines, and parametric curves).
    *   Surface-to-volume and volume-to-surface operations.
    *   Various post-processing functions 
    *   Custom user-defined modifications.
-   13 distinct methods for combining geometric objects, including different implementations of:
    *   Boolean operations (union, intersection, subtraction).
    *   Smooth operations (smooth union, smooth intersection, smooth subtraction).

## Key Features - Vector Fields

-   Support for **2D** and **3D** vector fields.
-   Object-oriented and functional approaches with pre-defined vector fields. 
-   Ability to define vector fields from scalar fields/SDFs or custom functions.
-   Extensive modifications and transformations, including:
    *   Addition, subtraction, and rescaling.
    *   Element-wise rotations in polar and azimuthal directions.
    *   Element-wise rotations around arbitrary axes.
    *   Revolutions of 2D vector fields around principal axes.

## Key Features - Automatic Differentiation

A [JAX](https://jax.readthedocs.io/en/latest/) implementation is available for nearly all SDFs, modifications, combinations, and post-processing functions. This enables computationally demanding operations to be offloaded to the GPU, with full support for [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

**JAX** is an optional dependency and will not be installed by default.

## Examples

The repository contains a wide variety of use cases:
-   [2D Examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/scalar/2D) (27 scripts) and [3D Examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/scalar/3D) (17 scripts) demonstrating geometry construction.
-   [Vector Examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/vector) (5 scripts) for vector field manipulation.
-   [Automatic Differentiation Examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/autodiff) (7 scripts) showcasing JAX integration.

All examples are provided in both Python script (`.py`) and interactive Jupyter Notebook (`.ipynb`) formats.

There are also [LLM Examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/LLM) demonstrating how to use various (open-source) LLMs to generate complex geometry.

## Installation

The simplest method to install SPOMSO is via `pip install SPOMSO`.

For more information, please refer to `aegolius_install.ipynb` in the [Examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples) folder.

## Citing

If you use SPOMSO in your research, please cite the latest archived repository of Aegolius on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8090670.svg)](https://doi.org/10.5281/zenodo.8090670)

## Documentation

Access the official [Documentation here](https://aegolius.readthedocs.io/en/latest/).

## Acknowledgements

This work was supported by the [Faculty of Mathematics and Physics at the University of Ljubljana](https://www.fmf.uni-lj.si/en/) and the [Institute Jožef Stefan](https://ijs.si/ijsw/V001/JSI). Special thanks to my colleagues for their helpful discussions.