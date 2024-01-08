import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_3d import Cylinder, Sphere, Cone
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 1.4, 1.4, 2.2
# resolution of the volume
co_resolution = 100, 100, 200

show = "BINARY" # BINARY, FIELD
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# define the torso of a chess pawn as a cone with a height of 1.2 and the slope angle of PI/10
torso = Cone(1.2, np.pi/10)
# position the torso in space
torso.move((0, 0, -0.55))
# round the torso with the rounding radius of 0.1 while preserving its maximum radius of 0.4
torso.rounding_cs(0.1, 0.4)

# define the base of the pawn as a cylinder with a radius of 0.4 and a height of 0.07
base = Cylinder(0.4, 0.07)
# round the cylinder with the rounding radius of 0.1 and position it in space
base.rounding(0.1)
base.move((0, 0, -0.95))

# define the head as a sphere with a radius of 0.25 and position it at the top of the torso
head = Sphere(0.25)
head.move((0, 0, 0.6))

# define the collar (rim below the head of the pawn) as a cylinder with a radius of 0.3 and height of 0.05
collar = Cylinder(0.3, 0.05)
# position the collar below the head
collar.move((0, 0, 0.25))

# define the union of two objects as an UNION2 operator
union = CombineGeometry("UNION2")
# apply the union operation to the base and the torso
p1 = union.combine(base, torso)
# apply the union once again on the result of the previous union and the head
statue = union.combine(p1, head)
# NOTE: union could also be defined as an UNION operator which can take multiple objects as an input.
# here the UNION2 was applied twice just to demonstrate that the operator (once defined) can be reused.

# define the SMOOTH_UNION2_2 operator with which we will join the collar and the rest of the pawn.
# there will be a smooth transition between the collar and the rest of the pawn.
smooth_union = CombineGeometry("SMOOTH_UNION2_2")
# smooth union with the smoothing width of 0.2
pawn = smooth_union.combine_parametric(statue, collar, parameters=0.2)
# position the pawn in space
pawn.move((0, 0, 0.2))

# evaluate the SDF of the pawn to create a signed distance field 3d map
pawn_pattern = pawn.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(pawn_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(pawn_pattern, 0)

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
        value=pawn_pattern,
        isomin=-0,
        isomax=0.5,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()









