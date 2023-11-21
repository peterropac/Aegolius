import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape, vector_smarter_reshape

from spomso.cores.geom_2d import Segment, Arc
from spomso.cores.geom_3d import Z
from spomso.cores.combine import CombineGeometry

from spomso.cores.geom_vector_special import LCWG3Dm1, LCWG2D

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS


# geometry parameters
u = 30
w = 30
d = 5.5

radius = 30
runup_x = 5
runup_y = 5
padding = 5, 5, 0

# size of the volume
co_size = radius + 2*padding[0] + u/2 + runup_x, radius + 2*padding[1] + u/2 + runup_y, d
# resolution of the volume
co_resolution = 100, 100, 50

# show 2D cross-section along the z-axis
show_midplane = True
# index of the cross-section along the z-axis
depth_index = 0
# 1/decimate of total vectors are shown in the cross-section
decimate = 4

# type vector field defining the waveguide: 2D, 3D
vector_field_type = "3D"

# show a 3D plot
show_3d = True
show_field = True
show_field_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# define the inlet and outlet by specifying the end points
inlet_s = (-co_size[0]/2,
           co_size[1]/2 - w/2 - padding[1],
           -d/2)
inlet_e = (-co_size[1]/2 + padding[0] + runup_x,
           co_size[1]/2 - w/2 - padding[1],
           -d/2)

outlet_s = (co_size[0]/2 - padding[0] - w/2,
            -co_size[1]/2,
            -d/2)
outlet_e = (co_size[0]/2 - padding[0] - w/2,
            -co_size[0]/2 + padding[1] + runup_y,
            -d/2)

# define the inlet and outlet
inlet = Segment(inlet_s, inlet_e)
outlet = Segment(outlet_s, outlet_e)

# define the arc connecting the inlet and outlet
connection = Arc(radius, 0, np.pi/2)
connection.set_location((inlet_e[0], outlet_e[1], -d/2))

# combine the parts of the waveguide together
combine = CombineGeometry("UNION")
# combine the parts
wg = combine.combine(inlet, connection, outlet)

# evaluate the SDF of the waveguide to create a signed distance field 3D map
wg_pattern = wg.create(coor)
print("pattern", wg_pattern.shape)


# ----------------------------------------------------------------------------------------------------------------------
# PLOT the SDF field of the waveguide

if show_midplane and show_field:
    field = smarter_reshape(wg_pattern, co_resolution)
    fig, ax = plt.subplots(1, 1, figsize=(8.25, 8.25))
    print(field.shape)
    ax.imshow(field[:, :, depth_index].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    z_mask = coor[2] == -d/2
    cs = ax.contour(coor[0, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    coor[1, z_mask].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :, depth_index],
                    cmap="plasma_r")
    ax.clabel(cs, inline=True, fontsize=10)
    ax.grid()

    fig.tight_layout()
    plt.show()

if show_field_3d and show_field:
    fig = go.Figure(data=go.Volume(
        x=coor[0],
        y=coor[1],
        z=coor[2],
        value=wg_pattern,
        isomin=-0,
        isomax=20,
        opacity=0.1,
        surface_count=5,
    ))
    fig.show()

# ----------------------------------------------------------------------------------------------------------------------
# CREATE VECTOR FIELDS

if vector_field_type == "2D":
    final = LCWG2D(w, co_resolution)
    coordinates = wg_pattern

if vector_field_type == "3D":
    vertical = Z(-d/2).create(coor)
    final = LCWG3Dm1((w, d), co_resolution)
    coordinates = (wg_pattern, vertical)


# evaluate the vector field functions to create a map of the vector field
final_field = final.create(coordinates)

# extract the x, y, and z components of the vector field
x = final.x(coordinates)
y = final.y(coordinates)
z = final.z(coordinates)

# extract the phi (azimuthal), theta (polar) angles and vector lengths
phi = final.phi(coordinates)
theta = final.theta(coordinates)
length = final.length(coordinates)

# convert the field maps into grids
field = vector_smarter_reshape(final_field, co_resolution)
x = smarter_reshape(x, co_resolution)
y = smarter_reshape(y, co_resolution)
z = smarter_reshape(z, co_resolution)
phi = smarter_reshape(phi, co_resolution)
theta = smarter_reshape(theta, co_resolution)
length = smarter_reshape(length, co_resolution)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))
# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
if show_midplane:
    # XY
    fig, axs = plt.subplots(2, 3, figsize=(8.25, 2*8.25/3), sharex="col", sharey="row")

    patterns = ((x, y, z), (length, phi, theta))
    titles = (("X component", "Y component", "Z component"), ("Length", r"$\phi$", r"$\vartheta$"))
    mins = ((-co_size[0]/2, -co_size[1]/2, -co_size[2]/2), (0, -np.pi, 0))
    maxs = ((co_size[0]/2, co_size[1]/2, co_size[2]/2), (1, np.pi, np.pi))

    for i in range(2):
        for j in range(3):
            ax = axs[i, j]

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            f = ax.imshow(patterns[i][j][:, :, depth_index].T,
                      cmap="bwr",
                      extent=(-co_size[0]/2, co_size[0]/2,
                              -co_size[1]/2, co_size[1]/2),
                      origin="lower",
                          vmin=mins[i][j],
                          vmax=maxs[i][j]
                      )

            cbar = fig.colorbar(f, cax=cax)
            cbar.set_ticks(np.linspace(mins[i][j], maxs[i][j], 3))
            cbar.set_ticklabels(np.round(np.linspace(mins[i][j], maxs[i][j], 3), 2))

            ax.set_xticks(np.linspace(-co_size[0]/2, co_size[0]/2, 3))
            ax.set_yticks(np.linspace(-co_size[1]/2, co_size[1]/2, 3))
            ax.grid()

            ax.quiver(smarter_reshape(coor[0], co_resolution)[::decimate, ::decimate, depth_index],
                      smarter_reshape(coor[1], co_resolution)[::decimate, ::decimate, depth_index],
                      field[0, ::decimate, ::decimate, depth_index],
                      field[1, ::decimate, ::decimate, depth_index])

            if i == 1:
                ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("y")

            ax.set_title(titles[i][j])

    fig.tight_layout()
    plt.show()

    # XZ
    fig, axs = plt.subplots(2, 3, figsize=(8.25, 2*8.25/3), sharex="col", sharey="row")

    patterns = ((x, y, z), (length, phi, theta))
    titles = (("X component", "Y component", "Z component"), ("Length", r"$\phi$", r"$\vartheta$"))
    mins = ((-co_size[0]/2, -co_size[1]/2, -co_size[2]/2), (0, -np.pi, 0))
    maxs = ((co_size[0]/2, co_size[1]/2, co_size[2]/2), (1, np.pi, np.pi))

    for i in range(2):
        for j in range(3):
            ax = axs[i, j]

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            f = ax.imshow(patterns[i][j][:, depth_index, :].T,
                      cmap="bwr",
                      extent=(-co_size[0]/2, co_size[0]/2,
                              -co_size[2]/2, co_size[2]/2),
                      origin="lower",
                          vmin=mins[i][j],
                          vmax=maxs[i][j]
                      )

            cbar = fig.colorbar(f, cax=cax)
            cbar.set_ticks(np.linspace(mins[i][j], maxs[i][j], 3))
            cbar.set_ticklabels(np.round(np.linspace(mins[i][j], maxs[i][j], 3), 2))

            ax.set_xticks(np.linspace(-co_size[0]/2, co_size[0]/2, 3))
            ax.set_yticks(np.linspace(-co_size[2]/2, co_size[2]/2, 3))
            ax.grid()

            ax.quiver(smarter_reshape(coor[0], co_resolution)[::decimate, depth_index, ::decimate],
                      smarter_reshape(coor[2], co_resolution)[::decimate, depth_index, ::decimate],
                      field[0, ::decimate, depth_index, ::decimate],
                      field[2, ::decimate, depth_index, ::decimate])

            if i == 1:
                ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("z")

            ax.set_title(titles[i][j])

    fig.tight_layout()
    plt.show()

if show_3d:

    xx = smarter_reshape(coor[0], co_resolution)[::decimate, ::decimate, :].flatten()
    yy = smarter_reshape(coor[1], co_resolution)[::decimate, ::decimate, :].flatten()
    zz = smarter_reshape(coor[2], co_resolution)[::decimate, ::decimate, :].flatten()
    vx = field[0, ::decimate, ::decimate, :].flatten()
    vy = field[1, ::decimate, ::decimate, :].flatten()
    vz = field[2, ::decimate, ::decimate, :].flatten()
    fig = go.Figure(
        data=go.Cone(
            x=xx,
            y=yy,
            z=zz,
            u=vx,
            v=vy,
            w=vz,
            colorscale='Blues',
            sizemode="absolute",
            sizeref=1
        )
    )

    fig.show()











