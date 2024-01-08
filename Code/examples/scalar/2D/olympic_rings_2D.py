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

# create circles with radius 0.5, thickness 0.05, and position them
radius = 0.5
thickness = 0.05
x_sep = 1.2
y_sep = 0.5

c1 = Circle(radius)
c1.onion(thickness)
c1.move((-x_sep, y_sep/2, 0))

c2 = Circle(radius)
c2.onion(thickness)
c2.move((0, y_sep/2, 0))

c3 = Circle(radius)
c3.onion(thickness)
c3.move((x_sep, y_sep/2, 0))

c4 = Circle(radius)
c4.onion(thickness)
c4.move((-x_sep/2, -y_sep/2, 0))

c5 = Circle(radius)
c5.onion(thickness)
c5.move((x_sep/2, -y_sep/2, 0))

# combine the circles together
union = CombineGeometry("UNION")
olympic_rings = union.combine(c1, c2, c3, c4, c5)

# evaluate the SDF of the olympic rings to create a signed distance field 2D map
olympic_rings_pattern = olympic_rings.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(olympic_rings_pattern, co_resolution)
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











