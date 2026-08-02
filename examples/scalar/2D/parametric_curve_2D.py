import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import ParametricCurve

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS


def polar_curve(t, radius1, radius2, f1, f2, ):

    r = radius1 + radius2*np.cos(f2*t*2*np.pi)

    x = r * np.cos(f1 * t * 2 * np.pi)
    y = r * np.sin(f1 * t * 2 * np.pi)

    return np.asarray((x, y))


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 6, 6
# resolution of the volume
co_resolution = 600, 600

show = "FIELD" # BINARY, FIELD
show_midplane = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# create a shape from the closed parametric curve
shape = False

# create a parametric curve geometry from the parametric function defining the curve
# the parameters of the function are 2, 0.5, 1, and 3
# the curve is evaluated for 201 values of the parameter t in range from 0 to 1
curve = ParametricCurve(polar_curve, (2, 0.5, 1, 3), (0, 1, 201), closed=True)

# create a shape
# this only works if the line is closed and has zero thickness
if shape:
    curve.shape()

# to get the sign of an SDF:
# curve_sign = curve.sign(direct=True)
# the line can be recovered with:
# curve.boundary()

# the interior can be recovered with a function:
# curve.recover_volume(curve_sign)

# the interior can be redefined with a function:
# curve.define_volume(some_sign_function, some_sign_function_parameters)

# thicken the curve to a thickness of 0.1 to create a shape
if not shape:
    curve.rounding(0.1)

# evaluate the SDF of the curve to create a signed distance field 2D map
curve_pattern = curve.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(curve_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show=="BINARY":
    fig, ax = plt.subplots(1,1, figsize=(8.25, 8.25))
    ax.imshow(pattern_2d[:, :].T,
              cmap="binary_r",
              extent=(-co_size[0]/2, co_size[0]/2,
                      -co_size[1]/2, co_size[1]/2),
              origin="lower"
              )
    ax.grid()

    fig.tight_layout()
    plt.show()

if show_midplane and show == "FIELD":
    fig, ax = plt.subplots(1, 1, figsize=(8.25, 8.25))
    print(field.shape)
    ax.imshow(field[:, :].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    cs = ax.contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                    coor[1].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :],
                    cmap="plasma_r")
    ax.clabel(cs, inline=True, fontsize=10)
    ax.grid()

    fig.tight_layout()
    plt.show()











