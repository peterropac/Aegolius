import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import Sector

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 2, 2
# resolution of the volume
co_resolution = 200, 200

show = "FIELD" # BINARY, FIELD
show_midplane = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# define the angles constraining the sector
angle_1 = np.pi/3
angle_2 = -np.pi/4
# and the radius of the sector
radius = 1

# create the sector/pie from the angles and the radius
# the sector will span along the shortest path
pie = Sector(radius, angle_1, angle_2)

# center the sector at the origin
mid_angle = (angle_2 + angle_1)/2
pie_center = radius*np.asarray((np.cos(mid_angle), np.sin(mid_angle)))/2
pie.set_location(-pie_center)

# evaluate the SDF of the pie to create a signed distance field 2D map
pie_pattern = pie.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(pie_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show=="BINARY":
    fig, ax = plt.subplots(1,1, figsize=(8.25, 8.25))
    ax.imshow(pattern_2d[:, :].T,
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
    print(field.shape)
    ax.imshow(field[:, :].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    cs = ax.contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                    coor[1].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :],
                    cmap="plasma_r")
    ax.clabel(cs, inline=True, fontsize=10)
    ax.grid()

    fig.tight_layout()
    plt.show()











