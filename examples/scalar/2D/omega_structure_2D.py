import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from spomso.cores.geom import GenericGeometry
from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import Segment, Arc
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 280, 400
# resolution of the volume
co_resolution = 400, 400

show = "BINARY" # BINARY, FIELD
show_midplane = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# define parameters
phi = 56
R = 100
n = 1
w = 5

# calculate the positions of key points
phi = np.deg2rad(phi)
period = 4*R*np.cos(phi)
a = R*np.sin(phi)

ixs = np.linspace(0, 2*n + 1, 2*n + 1, endpoint=False)

xs = period*ixs/2
ys = -a*np.cos(2*np.pi*xs/period) + a + R

# offset positions of key points
x0 = -n*period/2
y0 = -R*(1 + np.sin(phi))
xs = xs + x0
ys = ys + y0
print("structure offset:", xs, ys)

# calculate limiting angles for of the arcs:
phi_start = np.ones(ixs.size)
phi_start[1::2] = 0
phi_end = phi + np.pi*(phi_start + 1)
phi_start = np.pi*phi_start - phi
phi_start[0] = 3*np.pi/2
phi_end[-1] = 3*np.pi/2

# create the omega structure, inlet, and outlet
sdfs = []
for i in range(ixs.size):
    s = Arc(R, phi_start[i], phi_end[i])
    s.move((xs[i], ys[i], 0))
    s2 = GenericGeometry(s.propagate, ())
    sdfs.append(s2)

inlet = Segment((-co_size[0]/2, y0, 0), (x0, y0, 0))
outlet = Segment( (x0 + n*period, y0, 0), (co_size[0]/2, y0, 0))

sdfs.append(inlet)
sdfs.append(outlet)
sdfs = tuple(sdfs)

# combine the geometry and give it some thickness
union = CombineGeometry("UNION")
final = union.combine(*sdfs)
final.rounding(w/2)

# evaluate the SDF of the omega structure to create a signed distance field 2D map
final_pattern = final.create(coor)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# distance field to a binary voxel map, where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
    field = smarter_reshape(final_pattern, co_resolution)
    if show=="BINARY":
        pattern_2d = hard_binarization(field, 0)

if show=="BINARY":
    pattern = hard_binarization(final_pattern, 0)

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











