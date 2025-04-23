import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from time import process_time

from jax import jvp, jacfwd
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.jax_cores.sdf_2D_jax import sdf_circle

from spomso.jax_cores.combine_jax import combine_2_sdfs, combine_multiple_sdfs, parametric_combine_2_sdfs
from spomso.jax_cores.combine_jax import union, union2, smooth_union2_3o

from spomso.jax_cores.modifications_jax import onion
from spomso.jax_cores.transformations_jax import compound_euclidian_transform_sdf
from spomso.jax_cores.post_processing_jax import gaussian_falloff_jax

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 4
# resolution of the volume
co_resolution = 400, 400

# FIELD, PROJECTED
show = "PROJECTED"
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
# the distance between the circles
distance = 1.
# the width of the circles
width = 0.2
# the smoothing distance
smooth = 0.2


# define a new SDF where the radius, distance, width, and smoothing distance are the parameters
def combined_circle(r, d, w, s):
    sdf_1 = sdf_circle
    sdf_1 = compound_euclidian_transform_sdf(sdf_1, jnp.eye(3), jnp.asarray([-d/2, 0, 0]), 1.)

    sdf_2 = sdf_circle
    sdf_2 = compound_euclidian_transform_sdf(sdf_2, jnp.eye(3), jnp.asarray([d/2, 0, 0]), 1.)

    # sdf = combine_2_sdfs(sdf_1, sdf_2, (radius,), (radius,), union2)
    # sdf = combine_multiple_sdfs((sdf_1, sdf_2), ((radius,), (radius,)), union)
    sdf = parametric_combine_2_sdfs(sdf_1, sdf_2, (radius,), (radius,), smooth_union2_3o, s)

    onioned_circle = onion(sdf, w)

    vec = jnp.asarray([0, 0, 0])
    rot_mat = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    scale = 1.
    f_euclidean = compound_euclidian_transform_sdf(onioned_circle, rot_mat, vec, scale)

    out = f_euclidean(coor, r)
    return out


# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = combined_circle(radius, distance, width, smooth)
# calculate the Jacobian with respect to the radius (argnums=0), distance (argnums=1), width (argnums=2), or smoothing distance (argnums=3)
gradient_pattern = jacfwd(combined_circle, argnums=1)(radius, distance, width, smooth)

# convert the final pattern and the gradient pattern into 2D images
field = smarter_reshape(final_pattern, co_resolution)
gradient_field = smarter_reshape(gradient_pattern, co_resolution)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time - start_time))

# ----------------------------------------------------------------------------------------------------------------------
# PROJECTION

if show == "PROJECTED":
    gf, grad_gf = jvp(gaussian_falloff_jax, (final_pattern, 1.0, 0.5), (gradient_pattern, 0., 0.))
    field, gradient_field = smarter_reshape(gf, co_resolution), smarter_reshape(grad_gf, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show_midplane:
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


if show_3d:
    patterns = ((field, gradient_field),)
    titles = ("Field", "Gradient")

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=titles
                        )

    for i in range(1):
        for j in range(2):
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


