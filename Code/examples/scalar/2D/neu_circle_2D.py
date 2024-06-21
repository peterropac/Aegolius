import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.combine import CombineGeometry
from spomso.cores.geom_2d import NEUCircle

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 5, 5
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

# create non-euclidian circles with radius of 1 and  different order parameters
# order 0.5
non_euclidian_circle_1 = NEUCircle(1, 0.5)
# order 1
non_euclidian_circle_2 = NEUCircle(1, 1)
# order 2
non_euclidian_circle_3 = NEUCircle(1, 2)
# order 5
non_euclidian_circle_4 = NEUCircle(1, 5)

# move the non-euclidian circles
non_euclidian_circle_1.move((-1, 1, 0))
non_euclidian_circle_2.move((1, 1, 0))
non_euclidian_circle_3.move((1, -1, 0))
non_euclidian_circle_4.move((-1, -1, 0))

# combine geometries
union = CombineGeometry("UNION")
circles = union.combine(non_euclidian_circle_1,
                        non_euclidian_circle_2,
                        non_euclidian_circle_3,
                        non_euclidian_circle_4)

# evaluate the SDF of the non-euclidian circle to create a signed distance field 2D map
pattern = circles.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# convert the distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(pattern, co_resolution)
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

