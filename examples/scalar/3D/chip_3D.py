import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Cylinder
from spomso.cores.geom import GenericGeometry

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS


def displace(co_, A_, P_):
    return A_*np.sin(co_[1]*P_*np.pi*2)


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 3, 3, 2
# resolution of the volume
co_resolution = 150, 150, 100

show = "BINARY" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# create a cylinder with a radius of 1 and thickness of 0.1
cy = Cylinder(1, 0.05)
# rotate the cylinder so it is aligned with the y-axis
cy.rotate(np.pi/2, (1, 0, 0))

# extract the SDF of the cylinder
modified_cylinder_sdf = cy.propagate
# create new geometry from the extracted SDF
chip_s1 = GenericGeometry(modified_cylinder_sdf, ())
# bend the new geometry with the bending radius of 1.75 and bending radius of PI
chip_s1.bend(1.75, np.pi)
# reorient the new geometry for the second bend
chip_s1.rotate(np.pi/2, (0, 1, 0))

# extract the SDF from the previous stage
chip_s1_sdf = chip_s1.propagate
# create new geometry from the extracted SDF
chip_s2 = GenericGeometry(chip_s1_sdf, ())
# bend the new geometry with the bending radius of 1.75 and bending radius of PI
chip_s2.bend(1.75, np.pi)
# reorient the new geometry
chip_s2.rotate(np.pi/2, (1, 0, 0))


# extract the SDF from the previous stage
chip_s2_sdf = chip_s2.propagate
# create new geometry from the extracted SDF
chip = GenericGeometry(chip_s2_sdf, ())
# add displacement to the surface
chip.displacement(displace, (0.02, 10))

# evaluate the SDF of the chip to create a signed distance field 3D map
chip_pattern = chip.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(chip_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(chip_pattern, 0)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show=="BINARY":
    fig, ax = plt.subplots(1,1, figsize=(8.25, 8.25))
    ax.imshow(pattern_2d[co_resolution[0]//2, :, :].T,
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
    print(field.shape)
    ax.imshow(field[co_resolution[0]//2, :, :].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[2] / 2, co_size[2] / 2),
              origin="lower"
              )
    x_mask = coor[1] == 0
    cs = ax.contour(coor[0, x_mask].reshape(co_res_new[0], co_res_new[2]),
                    coor[2, x_mask].reshape(co_res_new[0], co_res_new[2]),
                    field[co_resolution[0]//2, :, :],
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
        value=chip_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()










