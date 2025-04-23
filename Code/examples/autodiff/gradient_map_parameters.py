import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from time import process_time

from jax import jvp, jacfwd

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.jax_cores.sdf_2D_jax import sdf_circle

from spomso.jax_cores.post_processing_jax import hard_binarization_jax
from spomso.jax_cores.post_processing_jax import relu_jax, linear_falloff_jax
from spomso.jax_cores.post_processing_jax import gaussian_boundary_jax, gaussian_falloff_jax
from spomso.jax_cores.post_processing_jax import capped_exponential_jax, sigmoid_falloff_jax

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 4
# resolution of the volume
co_resolution = 400, 400

show = "POSTPROCESSING"  # FIELD, POSTPROCESSING
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()

# define the radius of the circle
radius = 1.

# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = sdf_circle(coor, radius)

# calculate the Jacobian with respect to the radius (argnums=1)
# it should be -1 everywhere per definition of the SDF of the circle
gradient_pattern = jacfwd(sdf_circle, argnums=1)(coor, radius)

# convert the final pattern and the gradient pattern into 2D images
field = smarter_reshape(final_pattern, co_resolution)
gradient_field = smarter_reshape(gradient_pattern, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# POST-PROCESSING

# the SDF field can be transformed into other scalar fields with various operations
# one of the possibilities is to apply a post-processing function on top of the SDF field

# above we calculated the Jacobian of the SDF of a circle,
# now using the jax.jvp function we can apply the post-processing and calculate the gradients with respect to the radius
# in the primals argument we use parameters of the post-processing function
# in the tangents arguments we use the Jacobian we calculated above,
# the tangents for the post-processing function parameters we set to zero
# since we are not interested in the JVP with respect to the post-processing function parameters

# CAPPED EXPONENTIAL
# set the amplitude of the capped exponential post-processing function to 1 and the width to 0.5
ce, grad_ce = jvp(capped_exponential_jax, (final_pattern, 1.0, 0.5), (gradient_pattern, 0., 0.))
ce, grad_ce = smarter_reshape(ce, co_resolution), smarter_reshape(grad_ce, co_resolution)

# RELU
# set the width parameters of the relu postprocessing function to 1
rl, grad_rl = jvp(relu_jax, (final_pattern, 1.0), (gradient_pattern, 0.))
rl, grad_rl = smarter_reshape(rl, co_resolution), smarter_reshape(grad_rl, co_resolution)

# GAUSSIAN BOUNDARY
# set the amplitude of the gaussian boundary post-processing function to 1 and the width to 0.5
gb, grad_gb = jvp(gaussian_boundary_jax, (final_pattern, 1.0, 0.5), (gradient_pattern, 0., 0.))
gb, grad_gb = smarter_reshape(gb, co_resolution), smarter_reshape(grad_gb, co_resolution)

# LINEAR FALLOFF
# set the amplitude of the linear falloff post-processing function to 1 and the width to 0.5
lf, grad_lf = jvp(linear_falloff_jax, (final_pattern, 1.0, 0.5), (gradient_pattern, 0., 0.))
lf, grad_lf = smarter_reshape(lf, co_resolution), smarter_reshape(grad_lf, co_resolution)

# SIGMOID FALLOFF
# set the amplitude of the sigmoid falloff post-processing function to 1 and the width to 0.5
sf, grad_sf = jvp(sigmoid_falloff_jax, (final_pattern, 1.0, 0.5), (gradient_pattern, 0., 0.))
sf, grad_sf = smarter_reshape(sf, co_resolution), smarter_reshape(grad_sf, co_resolution)

# GAUSSIAN FALLOFF
# set the amplitude of the gaussian falloff post-processing function to 1 and the width to 0.5
gf, grad_gf = jvp(gaussian_falloff_jax, (final_pattern, 1.0, 0.5), (gradient_pattern, 0., 0.))
gf, grad_gf = smarter_reshape(gf, co_resolution), smarter_reshape(grad_gf, co_resolution)

# HARD BINARIZATION
# set the threshold parameter of the hard binarization postprocessing function to 0
hb, grad_hb = jvp(hard_binarization_jax, (final_pattern, 0.0), (gradient_pattern, 0.))
hb, grad_hb = smarter_reshape(hb, co_resolution), smarter_reshape(grad_hb, co_resolution)
grad_hb = np.zeros(grad_hb.shape) if grad_hb.dtype is not gradient_field.dtype else grad_hb

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time - start_time))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

print("Drawing results...")
# Mid-plane cross-section plot
if show_midplane and show == "FIELD":
    fig, axs = plt.subplots(1, 2, figsize=(2*8.25, 8.25))

    # FIELD
    axs[0].imshow(field[:, :].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    cs1 = axs[0].contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                    coor[1].reshape(co_res_new[0], co_res_new[1]),
                    field[:, :],
                    cmap="plasma_r",
                    linewidths=2)
    axs[0].clabel(cs1, inline=True, fontsize=10)
    axs[0].grid()

    # GRADIENT
    axs[1].imshow(gradient_field[:, :].T,
                  cmap="binary_r",
                  extent=(-co_size[0] / 2, co_size[0] / 2,
                          -co_size[1] / 2, co_size[1] / 2),
                  origin="lower"
                  )
    cs2 = axs[1].contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                        coor[1].reshape(co_res_new[0], co_res_new[1]),
                        gradient_field[:, :],
                        cmap="plasma_r",
                        linewidths=2)
    axs[1].clabel(cs2, inline=True, fontsize=10)
    axs[1].grid()

    for ax in axs:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    fig.tight_layout()
    plt.show()

if show_midplane and show == "POSTPROCESSING":

    # FIELD
    fig, axs = plt.subplots(2, 4, figsize=(8.25, 8.25 / 2))

    patterns = ((field, rl, lf, hb), (gb, sf, gf, ce))
    titles = (("SDF", "ReLU", "Linear Falloff", "Hard Binarization"),
              ("Gaussian Boundary", "Gaussian Falloff", "Sigmoid Falloff", "Capped exponential",))
    for i in range(2):
        for j in range(4):
            ax = axs[i, j]
            ax.imshow(patterns[i][j][:, :].T,
                      cmap="binary_r",
                      extent=(-co_size[0] / 2, co_size[0] / 2,
                              -co_size[1] / 2, co_size[1] / 2),
                      origin="lower"
                      )
            cs = ax.contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                            coor[1].reshape(co_res_new[0], co_res_new[1]),
                            patterns[i][j][:, :],
                            cmap="plasma_r",
                            linewidths=2)
            ax.clabel(cs, inline=True, fontsize=10)
            ax.set_xticks(np.linspace(-co_size[0] / 2, co_size[0] / 2, 3))
            ax.set_yticks(np.linspace(-co_size[1] / 2, co_size[1] / 2, 3))
            ax.grid()
            ax.set_title(titles[i][j])

    fig.tight_layout()
    plt.show()

    # GRADIENT
    fig, axs = plt.subplots(2, 4, figsize=(8.25, 8.25 / 2))

    patterns = ((gradient_field, grad_rl, grad_lf, grad_hb), (grad_gb, grad_sf, grad_gf, grad_ce))
    titles = (("SDF", "ReLU", "Linear Falloff", "Hard Binarization"),
              ("Gaussian Boundary", "Gaussian Falloff", "Sigmoid Falloff", "Capped exponential",))
    for i in range(2):
        for j in range(4):
            ax = axs[i, j]
            ax.imshow(patterns[i][j][:, :].T,
                      cmap="binary_r",
                      extent=(-co_size[0] / 2, co_size[0] / 2,
                              -co_size[1] / 2, co_size[1] / 2),
                      origin="lower"
                      )
            cs = ax.contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                            coor[1].reshape(co_res_new[0], co_res_new[1]),
                            patterns[i][j][:, :],
                            cmap="plasma_r",
                            linewidths=2)
            ax.clabel(cs, inline=True, fontsize=10)
            ax.set_xticks(np.linspace(-co_size[0] / 2, co_size[0] / 2, 3))
            ax.set_yticks(np.linspace(-co_size[1] / 2, co_size[1] / 2, 3))
            ax.grid()
            ax.set_title(titles[i][j])

    fig.tight_layout()
    plt.show()

if show_3d:
    # FIELD
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
                          1 + i, 1 + j,
                          )

            fig.update_traces(contours_z=dict(show=True, usecolormap=False,
                                              highlightcolor="limegreen", project_z=True))

    fig.show()

    # GRADIENT
    patterns = ((gradient_field, grad_rl, grad_lf, grad_hb), (grad_gb, grad_sf, grad_gf, grad_ce))
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
                                         "z": {"show": True,
                                               "start": np.floor(np.amin(patterns[i][j][:, :]*10))/10,
                                               "end": np.ceil(np.amax(patterns[i][j][:, :]*10))/10,
                                               "size": 0.1}
                                     },
                                     cmin=0,
                                     cmax=1),
                          1 + i, 1 + j,
                          )

            fig.update_traces(contours_z=dict(show=True, usecolormap=False,
                                              highlightcolor="limegreen", project_z=True))

    fig.show()
