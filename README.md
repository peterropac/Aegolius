
![](Files/Images/comb_230623_2.png#gh-dark-mode-only)
![](Files/Images/comb_220224_0.png#gh-light-mode-only)

**Aegolius** is the public repository for the Python package **SPOMSO**.

**SPOMSO** is a **free and open-source software** under the [GNU LGPL](https://www.gnu.org/licenses/lgpl-3.0.html).
This python package is intended for procedural construction of geometry and vector fields on the foundation of [Signed Distance Functions](https://en.wikipedia.org/wiki/Signed_distance_function) (SDFs).

## Key Features - SDFs

-   Geometry can be defined in **2D** and **3D**.
-   Object-oriented and function-oriented approach to defining geometry, with pre-defined 2D and 3D objects.
-   Built-in [Euclidian Transformations](https://en.wikipedia.org/wiki/Rigid_transformation) (translation, rotation, scaling)
-   Point clouds can be converted into SDFs and vice versa.
-   Euclidian transformations for SDFs and point clouds.
-   In total 50 possible modifications of SDFs, including:
    * extrusion, revolution, twist, bend, elongation
    * mirror, symmetry, rotational symmetry
    * finite and infinite instancing
    * instancing along lines, segmented lines and parametric curves
    * surface-to-volume and volume-to-surface operations
    * various post-processing functions
    * custom user-defined modifications
-   In total 13 ways to combine different geometric objects together - different implementations of:
    * union, intersection, subtraction
    * smooth union, smooth intersection, smooth subtraction

## Key Features - Vector Fields

-   Support for **2D** and **3D** vector fields.
-   Object-oriented and function-oriented approach, with pre-defined vector fields. 
-   Vector fields can be defined from scalar fields/SDFs or with custom functions.
-   Modifications and transformations, including:
    * addition, subtraction, rescaling
    * element-wise rotations in the polar and azimuthal directions
    * element-wise rotations around arbitrary axes
    * revolutions of 2D vector fields around one of the principal axes

## Key Features - Automatic Differentiation
There is a [JAX](https://jax.readthedocs.io/en/latest/) implementation of almost all SDFs, modifications, combination functions and post-processing functions. 
Therefore, it is possible to run the computationally demanding part of the code on a GPU from within Python, with support for [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
**JAX** is an optional dependency, so if you don't need these features it isn't necessary to install, and it won't be installed when you install **SPOMSO**.

## Examples

There are 27 [2D examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/scalar/2D) and 17 [3D examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/scalar/3D) showing how to construct geometry and use many of the features included in SPOMSO.
There are 5 [Vector examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/vector) showing how to construct and manipulate vector fields.
There are 7 [Automatic differentiation examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/autodiff) showing how to use the JAX implemented functions.
For each of the examples there is both a *python script (.py)* version and an *interactive python notebook (.ipynb)* version.

## Install

See `aegolius_install.ipynb` in the [Examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples).

## Citing

I kindly request that you cite the latest archived repository of Aegolius (SPOMSO) on Zenodo in any published work for which you used SPOMSO.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8090670.svg)](https://doi.org/10.5281/zenodo.8090670)

## Documentation

Official website with the [Documentation](https://aegolius.readthedocs.io/en/latest/).

## Acknowledgements

I acknowledge the support of the [Faculty of Mathematics and Physics at University of Ljubljana](https://www.fmf.uni-lj.si/en/), and [Institute Jožef Stefan](https://ijs.si/ijsw/V001/JSI).
Special thanks to my colleagues for helpful discussions.






