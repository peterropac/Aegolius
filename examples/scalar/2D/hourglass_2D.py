
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import SegmentedParametricCurve, SegmentedLine

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 6
# resolution of the volume
co_resolution = 400, 600

show = "FIELD" # BINARY, FIELD
show_midplane = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# there are two ways how to evaluate a segmented line
# the parametric approach allows us to calculate the SDF for only a section of the segmented line
# with the non-parametric approach the SDF will be calculated for the whole segmented line
parametric_evaluate = True

# create a polygon from the closed segmented line
polygon = True

points = [[-1, -2, 0], [1, 2, 0], [-1, 2, 0],  [1, -2, 0]]

# create a segmented line from the points. The line is closed on itself by setting closed=True.
if parametric_evaluate:
    spc = SegmentedParametricCurve(points, (0, 4, 200), closed=True)
# or
else:
    spc = SegmentedLine(points, closed=True)

# create a polygon
# this only works if the line is closed and has zero thickness
if polygon:
    spc.polygon()

# to get the sign of an SDF:
# spc_sign = spc.sign(direct=True)
# the line can be recovered with:
# spc.boundary()

# the interior can be recovered with a function:
# spc.recover_volume(spc_sign)

# the interior can be redefined with a function:
# spc.define_volume(some_sign_function, some_sign_function_parameters)

# thicken the line to a thickness of 0.1 to create a shape
if not polygon:
    spc.rounding(0.1)

# evaluate the SDF of the segmented line to create a signed distance field 2d map
segmented_line_pattern = spc.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(segmented_line_pattern, co_resolution)
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











