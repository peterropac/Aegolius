---
# Introduction
---

***Signed Distance Functions (SDFs)*** are scalar fields which at every point in metric space represent the shortest distance between the surface (boundary points) of an object and that point.
If the point is inside the object (enclosed by the surface) the **SDF** at that point has a negative sign.
These features, combined with manipulation of the underlying metric space (coordinate system) allow us to create procedural complex geometry from relatively simple mathematical functions.

SDFs allow us to create complex geometry with:

- ***Euclidian Transformations***
- ***Modifications***:
    - to the coordinate system
    - to the surface
- ***Boolean Operations*** between SDFs
- Changing the **metric** of the coordinate system

Please see the ***[Tutorials](SDF_Tutorials/Introduction.md)*** for more information.

SDFs also allow us to create complex procedural ***Vector Fields*** based on the **gradient** and the **value** of the SDF.

## Euclidian Transformations

Euclidian transformations to the SDFs are done by manipulating the coordinate system. 
For example ***translating*** the SDF by a ***vector*** is done by ***subtracting*** a ***vector*** from the ***position vectors*** in the coordinate system. 
To ***rotate*** the SDF the ***position vectors*** are rotated. To ***scale*** the SDF the ***position vectors*** are scaled.

$$ \mathbf{r_{ET}} = \mathbf{R} (\mathbf{r} - \mathbf{r_T})/S $$
$$ f_{ET}(r) = S \cdot f(\mathbf{r_{ET}}), $$

where $\mathbf{r}$ is the ***position vector***, $\mathbf{r_T}$ is the ***translation vector***, $\mathbf{R}$ is the ***rotation matrix***, $S$ in the ***scale factor***, 
***$f$*** is the SDF before the ***Euclidian transformation***, and ***$f_{ET}$*** is the transformed SDF.

***[Example](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/scalar/2D/basics_2D.ipynb)***

## Modifications

Geometry defined with SDFs can be modified in a similar fashion to Euclidian transforms - 
by modifying the underlying vector of position vectors or by applying a transformation to the values of SDF itself.
For example the ***Onion*** modification, which transforms a volume into a shell with some thickness is written as:

$$ f_{onion}( \mathbf{r} ) = | f(\mathbf{r}) | - w,$$

where $\mathbf{r}$ is the ***position vector***, $w$ is the ***shell thickness***, ***$f$*** is the SDF before the ***Onion modification***, 
and ***$f_{onion}$*** is the modified SDF.

***[Example](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/scalar/2D/olympic_rings_2D.ipynb)***

## Boolean Operations

An arbitrary geometry does not necessarily have an analytical function defining its SDF. 
The SDF can be computed using [Ray Marching](https://en.wikipedia.org/wiki/Ray_marching) or a similar method, which can be computationally expensive. 
The alternative is to use ***Boolean*** or ***Smooth/Parametric Boolean*** ***operations*** with SDFs to form complex geometries.

***Boolean operations*** between two SDFs such as ***union***, ***intersection***, ***difference/subtraction*** can be written as:

$$ f_{\textrm{union}} = \textrm{min} ( f_1 (\mathbf{r} ), f_2 (\mathbf{r} ) ) $$
$$ f_{\textrm{intersection}} = \textrm{max} ( f_1 (\mathbf{r} ), f_2 (\mathbf{r} ) ) $$
$$ f_{\textrm{subtraction}} = \textrm{max} ( f_1 (\mathbf{r} ), -f_2 (\mathbf{r} ) ),$$

where $f_1$ and $f_2$ are the SDFs, and $\textrm{min}$ and $\textrm{max}$ functions find the minimum/maximum of both SDFs at $\mathbf{r}$.

***Smooth/Parametric Boolean operations*** allow us to combine SDFs so that there is a smooth transition between them. 
For example the functions for ***smooth union***, ***smooth intersection*** and ***smooth subtraction*** can be written as:

$$ f_{\textrm{union}} = smin( f_1 (\mathbf{r} ), f_2 (\mathbf{r} ), a)$$
$$ f_{\textrm{intersection}} = -smin( -f_1 (\mathbf{r} ), -f_2 (\mathbf{r}), a )$$
$$ f_{\textrm{subtraction}} = -smin( -f_1 (\mathbf{r} ), f_2 (\mathbf{r} ), a ),$$

where $\textrm{smin}$ is a ***[Smooth Minimum Function](https://iquilezles.org/articles/smin/)*** ([others](https://en.wikipedia.org/wiki/Smooth_maximum)),
***$a$*** is the ***smoothing parameter***, and $f_1$ and $f_2$ are the SDFs.

***[Boolean Example](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/scalar/2D/olympic_rings_2D.ipynb)***, 
***[Boolean and Smooth Boolean Example](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/scalar/3D/plate_3D.ipynb)***,
***[Smooth Boolean Example](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/scalar/3D/pawn_3D.ipynb)***

## Vector Fields

SDF is a spatially dependent scalar field but by calculating the gradient of the SDF we can convert it into a spatially dependent vector field where all the vectors are unit length.
In this manner we can create vector fields around complex shapes. Additionally, the SDF itself can be used to modify the orientation of the vectors based on the distance from the surface.
There are also several built-in vector fields available, which can also be further modified using SDFs.

***[SDF Vector Fields](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/vector/sdf_vector_field.ipynb)***, ***[Build-in Vector Fields](https://nbviewer.org/github/peterropac/Aegolius/blob/main/Code/examples/vector/buildin_vector_fields.ipynb)*** 

## References

- [Signed Distance Functions Wiki](https://en.wikipedia.org/wiki/Signed_distance_function)
- [Inigo Quilez](https://iquilezles.org/articles/distfunctions/)
- [Cambridge](https://www.cl.cam.ac.uk/teaching/1819/FGraphics/1.%20Ray%20Marching%20and%20Signed%20Distance%20Fields.pdf)



