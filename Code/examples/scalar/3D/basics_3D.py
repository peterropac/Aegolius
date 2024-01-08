import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Box
from spomso.cores.geom import GenericGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 4, 2
# resolution of the volume
co_resolution = 100, 100, 100

show = "FIELD" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# create a box with side lengths 1, 0.5 and 0.25
box = Box(1, 0.5, 0.25)

# LOCATION
# move the center of the geometry (box) by the vector (0.1, 1, -0.25)
box.move((0.1, 1, -0.25))
# since the box is originally centered at zero the center of box is moved to (0.1, 1, -0.25)

# geometry can be moved to a specific location with the following function (in this case to (1, 1, 0.5))
box.set_location((1, 1, 0.5))
# set_location overrides all previous set_location and move calls

# move the box back to the origin
box.move((-1, -1, -0.5))

# SCALE
# rescale the object by a factor of 2
# the box will now have side lengths of 2, 1 and 0,5
box.rescale(2)
# rescale the box once again by a factor of 1.5
box.rescale(1.5)
# now the side lengths of the box are original_side_lengths*2*1.5

# the rescaling factor can be set to a specific value with:
box.set_scale(1.5)
# set_scale overrides all previous rescale and set_scale calls
# the side lengths are now 1.5, 0.75, and 0.375

# rescale the side lengths back to 1, 0.5, and 0.25
box.rescale(2/3)


# ROTATION
# rotate the box around the (1, 1, 0) axis for 90°
box.rotate(np.pi/2, (1, 1, 0))
# the rotate method can also accept a rotation matrix as an input

# rotate the box around the (-1, 1, 0) axis for -45°
box.rotate(-np.pi/4, (-1, 1, 0))
# rotate the box around the z-axis for 45°
box.rotate(np.pi/4, (0, 0, 1))

# set the rotation to a specific angle and axis
# 30° around the z-axis
box.set_rotation(np.pi/6, (0, 0, 1))
# set_rotation overrides all previous rotate or set_rotation calls

# rotate the box back to its original orientation
box.rotate(-np.pi/6, (0, 0, 1))

# to see the applied euclidian transformations use:
applied_transformations = box.transformations
print("Euclidian Transformations:", applied_transformations)

# to see the location of the geometry use:
center = box.center
print("Position of the box:", center)

# to see the scaling factor applied to the geometry use:
scale = box.scale
print("Scaling of the box:", scale)

# to see the rotation matrix applied to the geometry use:
rotation_matrix = box.rotation_matrix
print("Rotation matrix of the box:", rotation_matrix)


# NOTE:
# the Euclidean transformations are always applied after modifications
# for example:
# apply a rotation of 45° around the z-axis
box.rotate(np.pi/4, (0, 0, 1))
# apply mirroring on the x-axis where the original box is located at (1, 0, 0) and its mirror image at (-1, 0, 0)
box.mirror((-1, 0, 0), (1, 0, 0))
# the rotation was called before the mirror but the rotation is applied to the geometry after the mirroring

# to apply the modifications after the transformations a new object has to be crated:
box = GenericGeometry(box.propagate)
# apply the mirror operation once more, but this time along the y-axis
box.mirror((0, -0.5, 0), (0, 0.5, 0))
# in certain cases it does not matter if the Euclidian transformations are applied after the modifications
# this is true for modifications where the location of the geometry does not change
# for mirror, repetitions, and instancing modifications the order does matter

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









