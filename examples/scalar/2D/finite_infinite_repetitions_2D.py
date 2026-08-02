import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import Rectangle

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 4
# resolution of the volume
co_resolution = 400, 400

show = "FIELD" # BINARY, FIELD
show_midplane = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# type of repeating pattern
# INFINITE, FINITE, FINITE_RESCALED
repetition_type = "FINITE_RESCALED"

# create a quad by the defining the side lengths to be 1 and 0.5
quad = Rectangle(1.0, 0.5)

# the only parameter in infinite repetitions are the distances between (the centers of) the instances along each axis
# in this case the instances are separated by 1.2, 1.5, 2 along the x, y, and z-axis
if repetition_type=="INFINITE":
    quad.infinite_repetition((1.2, 1.5, 2))

# bounding box of the finite repetition is (2., 3., 1.) [first parameter]
# this means that all the instances will be located inside this bounding box
# there are 2, 3, and 1 repetition along the x, y and z-axis, respectively [second parameter]
if repetition_type=="FINITE":
    quad.finite_repetition((2., 3., 1.), (2, 3, 1))
# since the dimensions of the rectangle are 1 and 0.5, the size of the bounding box is (2., 3., 1.)
# and since there are 2 repetitions along the x-axis, the instance along the x-axis will be merged
# therefore, wider rectangles are created

# the finite repetition does not rescale the instances, only positions the centers of instances on a grid
# the distances between instances equal to bounding_box_size/number_of_repetitions along each axis

# to rescale the instances one can use the following:
if repetition_type=="FINITE_RESCALED":
    quad.finite_repetition_rescaled((2., 3., 1.), (2, 3, 1), (1, 0.5, 1), (0.2, 0.3, 0.0))
# bounding box of the finite repetition is (2., 3., 1.) [first parameter]
# there are 2, 3, and 1 repetition along the x, y and z-axis, respectively [second parameter]
# the size of (the bounding box) of each instance, and the padding along each axis must be provided
# in this case the size of the bounding box is 1, 0.5, and 1 [third parameter]
# and the padding is set to be 0.2, 0.3, and 0.0 [fourth parameter]
# the instances are rescaled so that they are as large as possible while still satisfying the padding constraints
# the proportions of the instances are the same as the proportions of the original geometry

# evaluate the SDF of the quad to create a signed distance field 2D map
quad_pattern = quad.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(quad_pattern, co_resolution)
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











