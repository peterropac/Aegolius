import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from time import process_time

from spomso.cores.geom import GenericGeometry

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.geom_2d import Circle

from spomso.cores.post_processing import hard_binarization
from spomso.cores.post_processing import PostProcess

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
# there are a few such predefined post-processing functions included in the package, here 7 of them are shown:

# CAPPED EXPONENTIAL
# set the amplitude of the capped exponential post-processing function to 1 and the width to 0.5
post_process = PostProcess(final.propagate)
post_process.capped_exponential(1, 0.5)
ce_geo = GenericGeometry(post_process.processed_geo_object, ())
ce_geo_field = ce_geo.create(coor)
ce = smarter_reshape(ce_geo_field, co_resolution)
print("Post-processing type:", post_process.post_processing_operations)

# RELU
# set the width parameters of the relu postprocessing function to 1
post_process = PostProcess(final.propagate)
post_process.relu(1.0)
rl_geo = GenericGeometry(post_process.processed_geo_object, ())
rl_geo_field = rl_geo.create(coor)
rl = smarter_reshape(rl_geo_field, co_resolution)
print("Post-processing type:", post_process.post_processing_operations)

# GAUSSIAN BOUNDARY
# set the amplitude of the gaussian boundary post-processing function to 1 and the width to 0.5
post_process = PostProcess(final.propagate)
post_process.gaussian_boundary(1.0, 0.5)
gb_geo = GenericGeometry(post_process.processed_geo_object, ())
gb_geo_field = gb_geo.create(coor)
gb = smarter_reshape(gb_geo_field, co_resolution)
print("Post-processing type:", post_process.post_processing_operations)

# LINEAR FALLOFF
# set the amplitude of the linear falloff post-processing function to 1 and the width to 0.5
post_process = PostProcess(final.propagate)
post_process.linear_falloff(1.0, 0.5)
lf_geo = GenericGeometry(post_process.processed_geo_object, ())
lf_geo_field = lf_geo.create(coor)
lf = smarter_reshape(lf_geo_field, co_resolution)
print("Post-processing type:", post_process.post_processing_operations)

# SIGMOID FALLOFF
# set the amplitude of the sigmoid falloff post-processing function to 1 and the width to 0.5
post_process = PostProcess(final.propagate)
post_process.sigmoid_falloff(1.0, 0.5)
sf_geo = GenericGeometry(post_process.processed_geo_object, ())
sf_geo_field = sf_geo.create(coor)
sf = smarter_reshape(sf_geo_field, co_resolution)
print("Post-processing type:", post_process.post_processing_operations)

# GAUSSIAN FALLOFF
# set the amplitude of the gaussian falloff post-processing function to 1 and the width to 0.5
post_process = PostProcess(final.propagate)
post_process.gaussian_falloff(1.0, 0.5)
gf_geo = GenericGeometry(post_process.processed_geo_object, ())
gf_geo_field = gf_geo.create(coor)
gf = smarter_reshape(gf_geo_field, co_resolution)
print("Post-processing type:", post_process.post_processing_operations)

# HARD BINARIZATION
# set the threshold parameter of the hard binarization postprocessing function to 0
post_process = PostProcess(final.propagate)
post_process.hard_binarization(0)
hb_geo = GenericGeometry(post_process.processed_geo_object, ())
hb_geo_field = hb_geo.create(coor)
hb = smarter_reshape(hb_geo_field, co_resolution)
print("Post-processing type:", post_process.post_processing_operations)

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












