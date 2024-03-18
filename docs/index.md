![](images/comb_220224_0.png)

**SPOMSO** is a **free and open-source software** under the [GNU LGPL](https://www.gnu.org/licenses/lgpl-3.0.html).
This python package is intended for procedural construction of geometry and vector fields on the foundation of [Signed Distance Functions](https://en.wikipedia.org/wiki/Signed_distance_function) (SDFs).

## Key Features - SDFs

-   Geometry can be defined in **2D** and **3D**.
-   Object-oriented and function-oriented approach to defining geometry, with pre-defined 2D and 3D objects.
-   Built-in [Euclidian Transformations](https://en.wikipedia.org/wiki/Rigid_transformation) (translation, rotation, scaling)
-   Point clouds can be converted into SDFs and vice-versa.
-   Euclidian transformations for SDFs and point clouds.
-   In total 40 possible modifications of SDFs, including:
    * extrusion, revolution, twist, bend, elongation
    * mirror, symmetry, rotational symmetry
    * finite and infinite instancing
    * instancing along lines, segmented lines and parametric curves
    * surface-to-volume and volume-to-surface operations
    * various post-processing functions
    * custom user-defined modifications
-   In total 11 ways to combine different geometric objects together - different implementations of:
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

## Examples

There are 24 [2D examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/scalar/2D) and 16 [3D examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/scalar/3D) showing how to construct geometry and use many of the features included in SPOMSO.
There are 5 [Vector examples](https://github.com/peterropac/Aegolius/tree/main/Code/examples/vector) showing how to construct and manipulate vector fields.
For each of the examples there is both a *python script (.py)* version and an *interactive python notebook (.ipynb)* version.

## Install

See the [Installation Guide](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/aegolius_install.ipynb).

## Citing

I kindly request that you cite the latest archived repository of Aegolius (SPOMSO) on Zenodo in any published work for which you used SPOMSO.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10470500.svg)](https://zenodo.org/badge/latestdoi/655622891)

## Acknowledgements

I acknowledge the support of the [Faculty of Mathematics and Physics at University of Ljubljana](https://www.fmf.uni-lj.si/en/), and [Institute Jožef Stefan](https://ijs.si/ijsw/V001/JSI).
Special thanks to my colleagues for helpful discussions.






