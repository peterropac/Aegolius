import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Box

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS


def spiral(t, radius, height, freq):

    x = radius*np.cos(2*np.pi*freq*t)
    y = radius*np.sin(2*np.pi*freq*t)
    z = height*t - height/2

    return np.asarray((x, y, z))


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

# type of instancing along the parametric curve: SIMPLE, ALIGNED, FULLY_ALIGNED
instancing_type = "FULLY_ALIGNED"

# create a box with side lengths 0.5, 0.2, 0.3
box = Box(0.5, 0.2, 0.3)

# instance the box on the spiral 21 times
# the radius of the spiral is set to 1, height is set to 1 and the frequency to 2
if instancing_type == "SIMPLE":
    box.curve_instancing(spiral, (1, 2, 2), (0, 1, 21))

# its is possible to align the box to tangent vectors on the curve.
if instancing_type == "ALIGNED":
    box.aligned_curve_instancing(spiral, (1, 2, 2), (0, 1, 21))

# it is possible to align the box to tangent, normal and binormal vectors on the curve.
if instancing_type == "FULLY_ALIGNED":
    box.fully_aligned_curve_instancing(spiral, (1, 2, 2), (0, 1, 21))

# evaluate the SDF of the instances to create a signed distance field 3D map
spiral_pattern = box.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(spiral_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(spiral_pattern, 0)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show=="BINARY":
    fig, ax = plt.subplots(1,1, figsize=(8.25, 8.25))
    ax.imshow(pattern_2d[:, co_resolution[1]//2, :].T,
              cmap="binary_r",
              extent=(-co_size[0]/2, co_size[0]/2,
                      -co_size[2]/2, co_size[2]/2),
              origin="lower"
              )
    ax.grid()

    fig.tight_layout()
    plt.show()

if show_midplane and show == "FIELD":
    fig, ax = plt.subplots(1, 1, figsize=(8.25, 8.25))
    ax.imshow(field[:, co_resolution[1]//2, :].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[2] / 2, co_size[2] / 2),
              origin="lower"
              )
    y_mask = coor[1] == 0
    cs = ax.contour(coor[0, y_mask].reshape(co_res_new[0], co_res_new[2]),
                    coor[2, y_mask].reshape(co_res_new[0], co_res_new[2]),
                    field[:, co_resolution[1]//2, :],
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
        value=spiral_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()









