import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom import GenericGeometry
from spomso.cores.geom_2d import Rectangle

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 4
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

# create a rectangle with side lengths of 1 and 0.5
rectangle = Rectangle(1, 0.5)

# LOCATION
# move the center of the geometry (rectangle) by the vector (0.1, 1, 0)
rectangle.move((0.1, 1, 0))
# since the rectangle is originally centered at zero the center of rectangle is moved to (0.1, 1, 0)

# geometry can be moved to a specific location with the following function (in this case to (1, 1, 0))
rectangle.set_location((1, 1, 0))
# set_location overrides all previous set_location and move calls

# move the rectangle back to the origin
rectangle.move((-1, -1, 0))

# SCALE
# rescale the object by a factor of 2
# the rectangle will now have side lengths of 2 and 1
rectangle.rescale(2)
# rescale the rectangle once again by a factor of 1.5
rectangle.rescale(1.5)
# now the side lengths of the rectangle are original_side_lengths*2*1.5

# the rescaling factor can be set to a specific value with:
rectangle.set_scale(1.5)
# set_scale overrides all previous rescale and set_scale calls
# the side lengths are now 1.5 and 0.75

# rescale the side lengths back to 1 and 0.5
rectangle.rescale(2/3)


# ROTATION
# rotate the rectangle around the z-axis for 45°
rectangle.rotate(np.pi/4, (0, 0, 1))
# the rotate method can also accept a rotation matrix as an input

# set the rotation to a specific angle and axis
# 30° around the z-axis
rectangle.set_rotation(np.pi/6, (0, 0, 1))
# set_rotation overrides all previous rotate or set_rotation calls

# rotate the rectangle back to its original orientation
rectangle.rotate(-np.pi/6, (0, 0, 1))


# to see the applied euclidian transformations use:
applied_transformations = rectangle.transformations
print("Euclidian Transformations:", applied_transformations)

# to see the location of the geometry use:
center = rectangle.center
print("Position of the rectangle:", center)

# to see the scaling factor applied to the geometry use:
scale = rectangle.scale
print("Scaling of the rectangle:", scale)

# to see the rotation matrix applied to the geometry use:
rotation_matrix = rectangle.rotation_matrix
print("Rotation matrix of the rectangle:", rotation_matrix)


# NOTE:
# the Euclidean transformations are always applied after modifications
# for example:
# apply a rotation of 45° around the z-axis
rectangle.rotate(np.pi/4, (0, 0, 1))
# apply mirroring on the x-axis
# the original rectangle will be located at (1, 0, 0) and its mirror image at (-1, 0, 0)
rectangle.mirror((-1, 0, 0), (1, 0, 0))
# the rotation was called before the mirror modification but the rotation is applied to the geometry after the mirroring

# to apply the modifications after the transformations a new object has to be crated:
rectangle = GenericGeometry(rectangle.propagate)
# apply the mirror operation once more, but this time along the y-axis
rectangle.mirror((0, -0.5, 0), (0, 0.5, 0))
# in certain cases it does not matter if the Euclidian transformations are applied after the modifications
# this is true for modifications where the location of the geometry does not change
# for mirror, repetitions, and instancing modifications the order does matter

# evaluate the SDF of the rectangle to create a signed distance field 2D map
rectangle_pattern = rectangle.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# convert the distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(rectangle_pattern, co_resolution)
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

