import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import Circle, Segment
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 2.4, 2.4, 1
# resolution of the volume
co_resolution = 200, 200, 100

show = "BINARY" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# total radius of the dinner plate
outer_plate_radius = 1
# radius of the flat part of the dinner plate
inner_plate_radius = 0.5
# height of the outer rim of the dinner plate
plate_rim_height = 0.12
# thickness of the plate
plate_thickness = 0.015
# thickness of the rim at the bottom
bottom_rim_radius = 0.01

# define the flat part of the dinner plate as a line segment between the origin and the inner radius
inner = Segment((0, 0, 0), 
                (inner_plate_radius, 0, 0))
# add thickness to the part
inner.rounding(plate_thickness)

# define the sloped part of the dinner plate as a line segment between the inner and outer radius
outer = Segment((inner_plate_radius, 0, 0),
                (outer_plate_radius, plate_rim_height, 0))
# adding thickness to the part
outer.rounding(plate_thickness)

# define the bottom rim of the dinner plate
bottom_rim = Circle(bottom_rim_radius)
bottom_rim.move((inner_plate_radius, -bottom_rim_radius*2, 0))

# combining the flat and the sloped part of the dinner plate with a union
union = CombineGeometry("UNION2")
top = union.combine(inner, outer)

# combining the bottom rim to the top with a smooth union with a smoothing width of 0.05
smooth_union = CombineGeometry("SMOOTH_UNION2")
plate = smooth_union.combine_parametric(top, bottom_rim, parameters=0.05)

# up until this point the shape was the cross-section of the dinner plate
# revolve the cross-section around the z-axis to create the dinner plate
plate.revolution(0)
# position the dinner plate in space
plate.rotate(np.pi/2, (1, 0, 0))
plate.move((0, 0, -plate_rim_height/2))

# evaluate the SDF of the plate to create a signed distance field 3D map
plate_pattern = plate.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(plate_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(plate_pattern, 0)

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
        value=plate_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()










