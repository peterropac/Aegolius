import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import Circle, SegmentedLine
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4.5, 4.5
# resolution of the volume
co_resolution = 400, 400

show = "FIELD" # BINARY, FIELD
show_midplane = True

export = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()


c1 = Circle(0.8)
c1.boundary()
c1.mirror((0.9, 0, 0), (-0.9, 0, 0))
c1.move((0, 0.25, 0))

points = [[0, -0.4, 0], [0.3, -0.65, 0], [0, -1.4, 0], [-0.3, -0.65, 0]]
line = SegmentedLine(points, closed=True)

# union = CombineGeometry("UNION")
# final = union.combine(c1, line)

union = CombineGeometry("SMOOTH_UNION2")
final = union.combine_parametric(c1, line, parameters=0.3)

logo_pattern = final.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(logo_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(logo_pattern, 0)

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
    # ax.imshow(field[:, :].T,
    #           cmap="binary_r",
    #           extent=(-co_size[0] / 2, co_size[0] / 2,
    #                   -co_size[1] / 2, co_size[1] / 2),
    #           origin="lower",
    #           vmin=0,
    #           vmax=0.5
    #           )
    cs = ax.contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                    coor[1].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :],
                    cmap="binary_r",
                    levels=np.linspace(-0, 0.5, 9),
                    linewidths=2)
    # ax.clabel(cs, inline=True, fontsize=10)
    # ax.grid()
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    if export:
        fig.savefig(r"F:\ULJ FMF\2022-2023\SPOMSO\Files\Images\project_logo_220224_1.png",
                    dpi=300,
                    transparent=True)


