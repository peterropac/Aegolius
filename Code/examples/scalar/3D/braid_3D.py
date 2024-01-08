import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom import GenericGeometry
from spomso.cores.geom_3d import Torus

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 1.5, 1.5, 4
# resolution of the volume
co_resolution = 150, 150, 200

show = "BINARY" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# create a torus with both radii
torus = Torus(0.25, 0.2)
# elongate the torus along the x-axis to create a chin link
torus.elongation((4., 0., 0.0))
# rotate the torus by 90° around the y-axis
torus.rotate(np.pi/2, (0, 1, 0))
# print the modifications applied to the torus
print("Torus modifications:", torus.modifications)
# print the transformations applied to the torus
print("Torus transformations:", torus.transformations)


# extract the SDF of the modified torus
modified_torus_sdf = torus.propagate
# create new geometry from the extracted SDF
braid = GenericGeometry(modified_torus_sdf, ())
# twist the new geometry (chain link) with a pitch of pi/(unit length) to create a braid
braid.twist(np.pi)
# print the modifications applied to the chain link object
print("Braid modifications:", braid.modifications)
# print the transformations applied to the chain link object
print("Braid transformations:", braid.transformations)

# evaluate the SDF of the braid to create a signed distance field 3D map
braid_pattern = braid.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(braid_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(braid_pattern, 0)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show=="BINARY":
    fig, ax = plt.subplots(1,1, figsize=(8.25, 8.25))
    ax.imshow(pattern_2d[:, :, co_resolution[-1]//2].T,
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
    ax.imshow(field[:, :, co_resolution[-1] // 2].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    z_mask = coor[2] == 0
    cs = ax.contour(coor[0, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    coor[1, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :, co_resolution[-1] // 2],
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
        value=braid_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()










