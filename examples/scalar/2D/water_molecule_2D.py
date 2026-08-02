import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import Circle
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 5, 3
# resolution of the volume
co_resolution = 500, 300

show = "FIELD" # BINARY, FIELD
show_midplane = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# angle between the H-atoms
angle = 104.5
# distance between the H and O atoms
d = 0.0957
# diameter of the H atom (not realistic!)
h_size = 0.075
# diameter of the O atom (not realistic!)
o_size = h_size*1.3

# smoothing width for the smooth union operation
smoothing_width = 0.45

# the positions of the H atoms
x_sep = 10*d*np.cos(np.deg2rad((180-angle)/2))
y_sep = 10*d*np.sin(np.deg2rad((180-angle)/2))

# create the H atom
h = Circle(10*h_size/2)
# instance the H atom twice (2) along a line
h.linear_instancing(2, (-x_sep, 0, 0), (x_sep, 0, 0))
# position the atoms
h.move((0, -y_sep, 0))

# create the O atom
o = Circle(10*o_size/2)


# combine the atoms together
combine = CombineGeometry("")
# see which non-parametric operations are available
operations = combine.available_operations
# see which parametric operations available
parametric_operations = combine.available_parametric_operations
# set the operation type as SMOOTH_UNION2, which creates a smooth union of two objects
combine.operation_type = "SMOOTH_UNION2"
# combine the three atoms
h2o = combine.combine_parametric(h, o, parameters=smoothing_width)

# evaluate the SDF of the "water molecule" to create a signed distance field 2D map
h2o_pattern = h2o.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(h2o_pattern, co_resolution)
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











