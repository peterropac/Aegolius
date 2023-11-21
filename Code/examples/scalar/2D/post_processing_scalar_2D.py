import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from time import process_time

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.geom_2d import Circle

from spomso.cores.post_processing import hard_binarization, capped_exponential, relu, gaussian_boundary
from spomso.cores.post_processing import linear_falloff, sigmoid_falloff, gaussian_falloff

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 4
# resolution of the volume
co_resolution = 400, 400

show = "POSTPROCESSING" # BINARY, FIELD, POSTPROCESSING
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# create some geometry
final = Circle(1)

# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = final.create(coor)

# convert the final pattern into a 2D image
field = smarter_reshape(final_pattern, co_resolution)


# the SDF field can be transformed into other scalar fields with various operations.
# one of the possibilities is to apply a post-processing function on top of the SDF field.
# there are 7 such predefined post-processing functions included in the package:

# CAPPED EXPONENTIAL
# set the amplitude of the capped exponential post-processing function to 1 and the width to 0.5
ce = capped_exponential(field, 1.0, 0.5)

# RELU
# set the width parameters of the relu postprocessing function to 1
rl = relu(field, 1.0)

# GAUSSIAN BOUNDARY
# set the amplitude of the gaussian boundary post-processing function to 1 and the width to 0.5
gb = gaussian_boundary(field, 1.0, 0.5)

# LINEAR FALLOFF
# set the amplitude of the linear falloff post-processing function to 1 and the width to 0.5
lf = linear_falloff(field, 1.0, 0.5)

# SIGMOID FALLOFF
# set the amplitude of the sigmoid falloff post-processing function to 1 and the width to 0.5
sf = sigmoid_falloff(field, 1.0, 0.5)

# GAUSSIAN FALLOFF
# set the amplitude of the gaussian falloff post-processing function to 1 and the width to 0.5
gf = gaussian_falloff(field, 1.0, 0.5)

# HARD BINARIZATION
# set the threshold parameter of the hard binarization postprocessing function to 0
hb = hard_binarization(field, 0)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# BINARIZATION
# Convert the distance field to a binary voxel map
# where 1 corresponds to the interior and 0 to the exterior of the geometry.

if show_midplane:
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


if show_midplane and show=="POSTPROCESSING":
    fig, axs = plt.subplots(2, 4, figsize=(8.25, 8.25/2))

    patterns = ((field, rl, lf, hb), (gb, sf, gf, ce))
    titles = (("SDF", "ReLU", "Linear Falloff", "Hard Binarization"),
              ("Gaussian Boundary", "Gaussian Falloff", "Sigmoid Falloff", "Capped exponential",))
    for i in range(2):
        for j in range(4):
            ax = axs[i, j]
            ax.imshow(patterns[i][j][:, :].T,
                      cmap="binary_r",
                      extent=(-co_size[0]/2, co_size[0]/2,
                              -co_size[1]/2, co_size[1]/2),
                      origin="lower"
                      )
            cs = ax.contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                            coor[1].reshape(co_res_new[0], co_res_new[1]),
                            patterns[i][j][:, :],
                            cmap="plasma_r",
                            linewidths=2)
            ax.clabel(cs, inline=True, fontsize=10)
            ax.set_xticks(np.linspace(-co_size[0]/2, co_size[0]/2, 3))
            ax.set_yticks(np.linspace(-co_size[1]/2, co_size[1]/2, 3))
            ax.grid()
            ax.set_title(titles[i][j])

    fig.tight_layout()
    plt.show()


if show_3d:
    patterns = ((field, rl, lf, hb), (gb, sf, gf, ce))
    titles = ("SDF", "ReLU", "Linear Falloff", "Hard Binarization",
              "Gaussian Boundary", "Gaussian Falloff", "Sigmoid Falloff", "Capped exponential")

    fig = make_subplots(rows=2, cols=4,
                        specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                               [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=titles
                        )

    for i in range(2):
        for j in range(4):
            fig.add_trace(go.Surface(x=coor[0].reshape(co_res_new[0], co_res_new[1]),
                                     y=coor[1].reshape(co_res_new[0], co_res_new[1]),
                                     z=patterns[i][j][:, :],
                                     contours={
                                         "z": {"show": True, "start": 0.0, "end": 1, "size": 0.1}
                                     },
                                     cmin=0,
                                     cmax=1),
                          1+i, 1+j,
                          )

            fig.update_traces(contours_z=dict(show=True, usecolormap=False,
                                              highlightcolor="limegreen", project_z=True))


    fig.show()












