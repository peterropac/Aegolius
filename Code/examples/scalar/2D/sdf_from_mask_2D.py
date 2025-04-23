import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import PointCloud2D
from spomso.cores.geom import Points, GenericGeometry
from spomso.cores.combine import CombineGeometry

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS


def alpha_greyscale_combine(image_):
    image_ = np.asarray(image_)
    print(image_.shape)

    image_ = image_/255.0
    out = 1-image_[:, :, 1]

    out = np.maximum(out, image_[:,:,0])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 8, 6
# resolution of the volume
co_resolution = 800, 600

show = "FIELD" # BINARY, FIELD
show_midplane = True

# set the morphology operation: ERODE or DILATE
morphology = "NOTHING"

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# import an image from a directory and convert the image to greyscale
image_file_name = "owl_logo.png"
spomso_dir = Path(os.getcwd()).resolve().parents[3]
image_path = os.path.join(spomso_dir, "Files", "test_images", image_file_name)
# if an image contains an alpha channel it must be converted as such:
image = Image.open(image_path).convert("LA")
# combine greyscale and alpha channels
image = alpha_greyscale_combine(image)

# display the greyscale image
plt.imshow(image, cmap="binary_r")
plt.show()

# create a point cloud object
points = Points([])
# extract the point cloud from the greyscale image [first parameter] (make sure values are between 0 and 1)
# the binary-threshold determines which pixels are included and which are not.
# all the pixels in the image with a brightness value below the binary threshold [third parameter]
# are included in the point cloud
# and the positions of the points are calculated from the specified image size [second parameter]
# first calculate the exterior SDF
points.from_image(image, (9, 16), binary_threshold=0.0)
exterior = PointCloud2D(points.cloud)
# repeat the same process for the interior, but this time invert the pixel values
points.from_image(1 - image, (9, 16), binary_threshold=0.0)
interior = PointCloud2D(points.cloud)

# combine both SDFs
union = CombineGeometry("DIFFERENCE")
final = union.combine(exterior, interior)

# dilate for 0.01
if morphology=="DILATE":
    final.rounding(0.5)
# or erode for 0.01
elif morphology=="ERODE":
    final.rounding(-0.5)
else:
    pass

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
                    cmap="BrBG",
                    linewidths=2,
                    vmin=-1, vmax=1,
                    levels=11)
    ax.clabel(cs, inline=True, fontsize=10)
    ax.grid()
    fig.tight_layout()
    plt.show()


