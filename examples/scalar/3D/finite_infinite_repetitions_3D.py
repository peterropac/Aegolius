import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Box

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 3, 3, 3
# resolution of the volume
co_resolution = 100, 100, 100

show = "BINARY" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# type of repeating pattern
# INFINITE, FINITE, FINITE_RESCALED
repetition_type = "FINITE_RESCALED"

# create the box by the defining the side lengths to be 1, 0.5, and 0.25
box = Box(1.0, 0.5, 0.25)

# infinite repetitions are only defined by the distances between (the centers of) instances along each axis
# in this case the instances are separated by 1.2, 1, 0.5 along the x, y, and z-axis
if repetition_type=="INFINITE":
    box.infinite_repetition((1.2, 1, 0.5))

# bounding box of the finite repetition is (2., 3., 2.) [first parameter]
# this means that all the instances will be located inside this bounding box
# there are 2, 3, and 4 repetition along the x, y and z-axis, respectively [second parameter]
if repetition_type=="FINITE":
    box.finite_repetition((2., 3., 2.), (2, 3, 4))
# since the dimensions of the box are (1, 0.5, 0.25), the size of the bounding box is (2., 3., 2.)
# and since there are 2 repetitions along the x-axis, the instances along the x-axis will be merged
# therefore, wider boxes are created

# the finite repetition does not rescale the instances only positions the centers of instances on a grid
# the distances between instances are bounding_box_size/number_of_repetitions along each axis

# to rescale the instances one can use the following:
if repetition_type=="FINITE_RESCALED":
    box.finite_repetition_rescaled((2., 3., 2.), (2, 3, 5), (1, 0.5, 0.25), (0.2, 0.3, 0.1))
# bounding box of the finite repetition is (2., 3., 2.) [first parameter]
# there are 2, 3, and 5 repetition along the x, y and z-axis, respectively [second parameter]
# the size of (the bounding box) of each instance, and the padding along each axis must be provided
# in this case the size of the bounding box is 1, 0.5, and 1 [third parameter]
# and the padding is set to be 0.2, 0.3, and 0.1 [fourth parameter]
# the instances are rescaled so that they are as large as possible while still satisfying the padding constraints
# the proportions of the instances is the same as the proportions of the original geometry

# evaluate the SDF of the box to create a signed distance field 3D map
box_pattern = box.create(coor)


end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(box_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(box_pattern, 0)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show=="BINARY":
    fig, ax = plt.subplots(1,1, figsize=(8.25, 8.25))
    ax.imshow(pattern_2d[:, :, co_resolution[2]//2].T,
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
    ax.imshow(field[:, :, co_resolution[2]//2].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    z_mask = coor[2] == 0
    cs = ax.contour(coor[0, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    coor[1, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :, co_resolution[2]//2],
                    cmap="plasma_r")
    ax.clabel(cs, inline=True, fontsize=10)
    ax.grid()

    fig.tight_layout()
    plt.show()


# Isosurfaces plot
if show_3d and show=="BINARY":
    fig = go.Figure(data=go.Volume(
        x=coor[0],
        y=coor[1],
        z=coor[2],
        value=pattern,
        isomin=0.1,
        isomax=1,
        opacity=0.1,
        surface_count=2,
    ))
    fig.show()

if show_3d and show=="FIELD":
    fig = go.Figure(data=go.Volume(
        x=coor[0],
        y=coor[1],
        z=coor[2],
        value=box_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()









