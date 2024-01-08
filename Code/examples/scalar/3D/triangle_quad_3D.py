import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Triangle3D, Quad
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 2.5, 1.5, 2.5
# resolution of the volume
co_resolution = 200, 100, 200

show = "FIELD" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# define the points of the triangle
point_A = (-1, 0.2, -1)
point_B = (1, -0.2, 0)
point_C = (0.1, 0.5, 1)

# create the triangle from the three points
trig = Triangle3D(point_A, point_B, point_C)
# add thickness to the triangle
trig.rounding(0.05)

# define the points of the quadrilateral
# the four points must be in plane
point_Q_A = (-1, 0.2, 0)
point_Q_B = (1, -0.2, 0)
point_Q_C = (0.1, 0.5, 0)
point_Q_D = (-0.5, 0.5, 0)

# create the quadrilateral from the four points
quad = Quad(point_Q_A, point_Q_B, point_Q_C, point_Q_D)
# only the quadrilateral can be seen in the 2D cross-section

# add thickness to the quadrilateral
quad.rounding(0.05)

# create a union of both objects
union = CombineGeometry("UNION")
combined = union.combine(quad, trig)

# evaluate the SDF of the combined to create a signed distance field 3D map
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
        value=combined_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()









