import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go

from time import process_time
from PIL import Image
from pathlib import Path

from spomso.cores.helper_functions import generate_grid, smarter_reshape, vector_smarter_reshape

from spomso.cores.geom import Points
from spomso.cores.geom_2d import PointCloud2D
from spomso.cores.geom_3d import Z
from spomso.cores.geom_vector_special import LCWG3Dm1, LCWG2D
from spomso.cores.post_processing import conv_averaging

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 100, 70, 5.5
# waveguide width
w = 30
# resolution of the volume
co_resolution = 100, 100, 11

# show 2D cross-section along the z-axis
show_midplane = True
# index of the cross-section along the z-axis
depth_index = 0
# 1/decimate of total vectors are shown in the cross-section
decimate = 4

# type of the vector field defining the waveguide: 2D, 3D
vector_field_type = "3D"

# how the sign of the vector field is determined: MASK, AUTO
sign_compute = "MASK"

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

# import an image from a directory and convert the image to greyscale
image_file_name = "waveguide_corrected.jpg"
spomso_dir = Path(os.getcwd()).resolve().parents[1]
image_path = os.path.join(spomso_dir, "files", "test_images", image_file_name)
image = Image.open(image_path).convert("L")
image = np.flipud(np.asarray(image).T)

# import the sign mask as an image
if sign_compute=="MASK":
    mask_image_file_name = "waveguide_corrected_sign.jpg"
    mask_image_path = os.path.join(spomso_dir, "files", "test_images", mask_image_file_name)
    mask_image_ = Image.open(mask_image_path).convert("L")
    mask_image = np.flipud(np.asarray(mask_image_).T)

# display the greyscale image
plt.imshow(image, cmap="binary_r")
plt.show()

# display the mask
if sign_compute=="MASK":
    plt.imshow(mask_image, cmap="binary_r")
    plt.show()

# create a point cloud object
points = Points([])
# calculate aspect ratio of the image
aspect_ratio = image.shape[0]/image.shape[1]
# extract the point cloud from the greyscale image (first parameter)
# all the pixels in the image with a brightness value below the binary threshold (third parameter)
# are included in the point cloud
# and the positions of the points are calculated from the specified image size (second parameter)
points.from_image(image, (co_size[0], co_size[0]*aspect_ratio), binary_threshold=0.5)
cloud = points.cloud

# create an SDF from the point cloud
wg = PointCloud2D(cloud)

# evaluate the SDF of the waveguide to create a signed distance field 3D map
wg_pattern = wg.create(coor)

# smooth the SDF to avoid artifacts in the final vector field
wg_pattern_smooth = smarter_reshape(wg_pattern, co_resolution)
wg_pattern_smooth = conv_averaging(wg_pattern_smooth, (5, 5, 1), 1)
wg_pattern = wg_pattern_smooth.reshape(wg_pattern.shape)

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
    z_mask = coor[2] == -co_size[2]/2
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

    fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=co_size[1]/co_size[0], z=co_size[2]/co_size[0])
        )
    )

    fig.show()

# ----------------------------------------------------------------------------------------------------------------------
# CREATE VECTOR FIELDS

if sign_compute == "MASK":
    # calculate the image aspect ratio
    mask_aspect_ratio = mask_image.shape[0]/mask_image.shape[1]
    # convert image into a point cloud
    mask_points = Points([])
    mask_points.from_image(mask_image,
                           (co_size[0], co_size[0] * mask_aspect_ratio),
                           binary_threshold=0.5)
    # covert point cloud to a mask
    mask_ = mask_points.to_image(co_size, co_resolution, extend=("-Z", "+Z", "-Y"))
    mask_ = 1 - mask_
    # non-zero values represent a positive sign and zero values represent a negative sign
    sign_ = 2.0*(mask_>0) - 1.0
    sign_ = sign_.flatten()
elif sign_compute == "AUTO":
    sign_ = None
else:
    sign_ = None

if vector_field_type == "2D":
    final = LCWG2D(w, co_resolution, sign_)
    coordinates = wg_pattern

if vector_field_type == "3D":
    vertical = Z(-co_size[2]/2).create(coor)
    final = LCWG3Dm1((w, co_size[2]), co_resolution, sign_)
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
    fig, axs = plt.subplots(2, 3, figsize=(8.25, 1.5*8.25/3), sharex="col", sharey="row")

    patterns = ((x, y, z), (length, phi, theta))
    titles = (("X component", "Y component", "Z component"), ("Length", r"$\phi$", r"$\vartheta$"))
    mins = ((-1, -1, -1), (0, -np.pi, 0))
    maxs = ((1, 1, 1), (1, np.pi, np.pi))

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
    fig.subplots_adjust(left=0.112,
                        bottom=0.05,
                        right=0.936,
                        top=1,
                        wspace=0.336,
                        hspace=0.0)
    plt.show()

    # XZ
    fig, axs = plt.subplots(2, 3, figsize=(8.25*2, 0.7*8.25/3), sharex="col", sharey="row")

    patterns = ((x, y, z), (length, phi, theta))
    titles = (("X component", "Y component", "Z component"), ("Length", r"$\phi$", r"$\vartheta$"))
    mins = ((-1, -1, -1), (0, -np.pi, 0))
    maxs = ((1, 1, 1), (1, np.pi, np.pi))

    for i in range(2):
        for j in range(3):
            ax = axs[i, j]

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            depth_index_xz = patterns[i][j].shape[1]//2
            f = ax.imshow(patterns[i][j][:, depth_index_xz, :].T,
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
    fig.subplots_adjust(left=0.112,
                        bottom=0.05,
                        right=0.936,
                        top=1,
                        wspace=0.336,
                        hspace=0.0)
    plt.show()

if show_3d:

    xx = smarter_reshape(coor[0], co_resolution)[::decimate, ::decimate, :].flatten()
    yy = smarter_reshape(coor[1], co_resolution)[::decimate, ::decimate, :].flatten()
    zz = smarter_reshape(coor[2], co_resolution)[::decimate, ::decimate, :].flatten()
    vx = field[0, ::decimate, ::decimate, :].flatten()
    vy = field[1, ::decimate, ::decimate, :].flatten()
    vz = field[2, ::decimate, ::decimate, :].flatten()
    phi_ = phi[::decimate, ::decimate, :].flatten()

    fig = go.Figure()

    fig = go.Figure(
        data=go.Cone(
            x=xx,
            y=yy,
            z=zz,
            u=vx,
            v=vy,
            w=vz,
            colorscale='Blues_r',
            sizemode="absolute",
            sizeref=5
        )
    )

    fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=co_size[1]/co_size[0], z=2*co_size[2]/co_size[0])
        )
    )

    fig.show()

