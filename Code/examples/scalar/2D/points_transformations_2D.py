import numpy as np
import matplotlib.pyplot as plt
from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import PointCloud2D
from spomso.cores.geom import Points

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 3, 3
# resolution of the volume
co_resolution = 400, 400

show = "FIELD" # BINARY, FIELD
show_midplane = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# define 4 points in 3D space
coordinates = [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]
coordinates = np.asarray(coordinates).T

# create a point cloud object
points = Points(coordinates)

# rotate the points around the z-axis by 30˘
points.rotate(np.pi/6, (0, 0, 1))

# scale the points by 0.5 along the x-axis and 0.75 along the y-axis
points.rescale((0.5, 0.75, 1))

# move the points by a vector (0.2, 0.1, 0)
points.move((0.2, 0.1, 0))

# the order of transformations is always (no matter the order in the code): rescale, rotate, translate
# to change the order one can create a new points object for each transform
# to create a new points object from the existing one, call:
# new_points = Points(points.cloud)

# get the new coordinates of points
cloud = points.cloud

# create an SDF from the point cloud
final = PointCloud2D(cloud)

# transform the points into circles with width 0.1
final.onion(0.1)

# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = final.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# Convert the distance field to a binary voxel map
# where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(final_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show=="BINARY":
    fig, ax = plt.subplots(1, 1, figsize=(8.25, 8.25))
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
                    cmap="plasma_r",
                    linewidths=2)
    ax.clabel(cs, inline=True, fontsize=10)
    ax.grid()
    fig.tight_layout()
    plt.show()


