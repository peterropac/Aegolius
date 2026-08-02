import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import NGon
from spomso.cores.geom_3d import Box, Cylinder, Arc3D
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 3, 3, 1.6
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

# create a box with side lengths 3, 1 and 0.5
box = Box(3, 1, 0.5)

# create a hexagon with radius of 0.3
hexagon = NGon(0.3, 6)
# convert the shape into a line
hexagon.boundary()
# create two concentric lines with a separation of 0.2
hexagon.concentric(0.2)
# give the lines a thickness of 0.05
hexagon.rounding(0.05)
# extrude the shape along the z-axis for 2 units 
hexagon.extrusion(2)
# position the hexagon
hexagon.move((0.5, 0, 0))

# create a cylinder with radius of 0.4 and height of 1
cy = Cylinder(0.4, 1)
# position the cylinder
cy.move((-0.5, 0, 0))
# rotate the cylinder for 30° around the x-axis
cy.rotate(np.pi/6, (0, 1, 0))

# create an arc in 3D with the radius of 1, thickness of 0.1
# the endpoints are at angles 45° and 315°.
arc = Arc3D(1, 0.2, np.pi/4, 7*np.pi/4)

# create a cylinder with radius of 1.2 and height of 1
cya = Cylinder(1.2, 1)

# define union, subtract and intersect operations
union = CombineGeometry("UNION")
subtract = CombineGeometry("SUBTRACT2")
intersect = CombineGeometry("INTERSECT2")

# merge the box and the arc into geometry s1 and merge the hexagon and the cylinder into geometry s2
# the s1 geometry will be the base from which s2 geometry is subtracted
s1 = union.combine(box, arc)
s2 = union.combine(hexagon, cy)

# subtract geometry s2 from geometry s1
s3 = subtract.combine(s1, s2)
# by intersecting the geometries of s3 and the large cylinder the edges of the box are rounded
s4 = intersect.combine(s3, cya)

# evaluate the SDF of the combined geometry to create a signed distance field 3D map
combined_pattern = s4.create(coor)

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
    ax.imshow(pattern_2d[:, :,  co_resolution[2]//2].T,
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
    ax.imshow(field[:, :,  co_resolution[2]//2].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    z_mask = coor[2] == 0
    cs = ax.contour(coor[0, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    coor[1, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :,  co_resolution[2]//2],
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









