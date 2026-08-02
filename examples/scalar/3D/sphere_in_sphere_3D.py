import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Sphere
from spomso.cores.geom_2d import Arc
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 2.3, 2.3, 2.3
# resolution of the volume
co_resolution = 150, 150, 150

show = "BINARY" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# create the inner sphere with a radius of 0.25
inner_sphere = Sphere(0.25)
# rescale the sphere for a factor of 2: radius of the sphere will be 0.25*2
inner_sphere.rescale(2)
# set scaling of the sphere to 1.5: radius of the sphere will be 0.25*1.5,
# since set_scale overrides the previous rescale and set_scale commands
inner_sphere.set_scale(1.5)

# create the 2d arc which will become the outer hollow sphere
outer_sphere = Arc(0.75, -np.pi/4, np.pi/3)
# add 0.1 of thickness to the arc
outer_sphere.rounding(0.1)
# revolve the arc around the y-axis to create the hollow sphere with a cutout
outer_sphere.revolution(0)
# print the modifications applied to the outer sphere
print("Outer sphere modifications:", outer_sphere.modifications)
# print the transformations applied to the outer sphere
print("Outer sphere transformations:", outer_sphere.transformations)


# Combine the inner and the outer sphere into a single geometry
union = CombineGeometry("UNION2")
combined = union.combine(inner_sphere, outer_sphere)
# rotate the combined geometry by 45° around the z-axis
combined.rotate(np.pi/4, (0,0,1))
# scale the combined geometry by a factor of 1.25: the new size of the geometry will be size*1.25.
# The radius of the inner sphere will be 0.25*1.5*1.25
combined.rescale(1.25)
# print the modifications applied to the combined geometry
print("Combined geometry modifications:", combined.modifications)
# print the transformations applied to the combined geometry
print("Combined geometry transformations:", combined.transformations)
# evaluate the SDF of the combined geometry to create a signed distance field 3d map
combined_pattern = combined.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(combined_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(combined_pattern, 0)

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
        value=combined_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()










