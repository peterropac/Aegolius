import os

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.post_processing import hard_binarization
from spomso.cores.geom_2d import PointCloud2D
from spomso.cores.geom import Points

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

# import an image from a directory and convert the image to greyscale
image_file_name = "lines_test_handdrawn.png"
spomso_dir = Path(os.getcwd()).resolve().parents[3]
image_path = os.path.join(spomso_dir, "Files", "test_images", image_file_name)
image = Image.open(image_path).convert("L")

# display the greyscale image
plt.imshow(image, cmap="binary_r")
plt.show()

# create a point cloud object
points = Points([])
# extract the point cloud from the greyscale image [first parameter]
# all the pixels in the image with a brightness value below the binary threshold [third parameter]
# are included in the point cloud
# and the positions of the points are calculated from the specified image size [second parameter]
points.from_image(image, (3, 1.5), binary_threshold=0.5)
cloud = points.cloud

# create an SDF from the point cloud
final = PointCloud2D(cloud)
# give the lines a thickness of 0.01
final.onion(0.01)

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
                    cmap="plasma_r",
                    linewidths=2)
    ax.clabel(cs, inline=True, fontsize=10)
    ax.grid()
    fig.tight_layout()
    plt.show()


