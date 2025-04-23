import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from time import process_time

from jax import jvp, jacfwd, value_and_grad, config
import jax.numpy as jnp
import optax

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.jax_cores.sdf_2D_jax import sdf_circle
from spomso.jax_cores.transformations_jax import compound_euclidian_transform_sdf
from spomso.jax_cores.modifications_jax import gaussian_falloff

from spomso.jax_cores.combine_jax import combine_2_sdfs, parametric_combine_2_sdfs
from spomso.jax_cores.combine_jax import union2, smooth_union2_3o


config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 8, 8
# resolution of the volume
co_resolution = 400, 400

show = "FIELD"  # FIELD, TARGET, OBJECTIVE
show_midplane = True
show_3d = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM

coor, co_res_new = generate_grid(co_size, co_resolution)

# ----------------------------------------------------------------------------------------------------------------------
# SDF PARAMETERS

# define the radii of the circles
radius = 1., 1., 1.

# define the initial positions of the circles
x = 0.0, 0.1, -1.
y = 0.0, 0.2, 0.5

# define the target positions of the circles
x_target = 2.2, 0.4, -2.
y_target = -1.0, -0.2, 1.1

# the optimization can also be done using SDFs - without any additional post-processing functions
pure_sdf = True

# the circles in the dynamic field can be combined using a smooth union function
smooth_combine = False

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()


# define a target field where the input are the target positions of the circles
def target_sdf(x0, y0):

    vec = jnp.asarray([x0[0], y0[0], 0])
    sdf = compound_euclidian_transform_sdf(sdf_circle, jnp.eye(3), vec, 1.)

    vect = jnp.asarray([x0[1], y0[1], 0])
    sdf_t = compound_euclidian_transform_sdf(sdf_circle, jnp.eye(3), vect, 1.)

    sdf = parametric_combine_2_sdfs(sdf, sdf_t, (radius[0],), (radius[1],), smooth_union2_3o, 0.75)

    if len(x0) > 2:
        for i in range(2, len(x0)):
            vect = jnp.asarray([x0[i], y0[i], 0])
            sdf_t = compound_euclidian_transform_sdf(sdf_circle, jnp.eye(3), vect, 1.)
            sdf = parametric_combine_2_sdfs(sdf, sdf_t, (), (radius[i],), smooth_union2_3o, 0.75)

    if not pure_sdf:
        sdf = gaussian_falloff(sdf, 1., 0.5)
    out = sdf(coor)

    return out


# define the field of a circle where the input is the pointcloud of coordinates,
# and the position and radius of the circle
def s_circle(co, vec, r):
    co = jnp.subtract(co.T, vec).T + 0.001
    out = sdf_circle(co, r)
    return out


# define the dynamic field where the input are the positions of the circles
def combined_dynamic_field(vecs):

    if smooth_combine:
        sdf = parametric_combine_2_sdfs(s_circle, s_circle,
                                        (vecs[:, 0], radius[0]), (vecs[:, 1], radius[1]),
                                        smooth_union2_3o, 0.75)
    else:
        sdf = combine_2_sdfs(s_circle, s_circle,
                                        (vecs[:, 0], radius[0]), (vecs[:, 1], radius[1]),
                                        union2)
    if vecs.shape[1] > 2:
        for i in range(2, vecs.shape[1]):
            if smooth_combine:
                sdf = parametric_combine_2_sdfs(sdf, s_circle,
                                                (), (vecs[:, i], radius[i]),
                                                smooth_union2_3o, 0.75)
            else:
                sdf = combine_2_sdfs(sdf, s_circle,
                                                (), (vecs[:, i], radius[i]),
                                                union2)

    if not pure_sdf:
        sdf = gaussian_falloff(sdf, 1., 0.5)
    out = sdf(coor)
    return out


# define the objective/cost function, where the inputs are the dynamic and target fields
def objective_function(f_, t_):
    m = (f_ - t_)**2
    return m


# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

# compute the target field
target_pattern = target_sdf(x_target, y_target)


# define the worker for the optax optimizer
def worker(p):
    dynamic_pattern = combined_dynamic_field(p)
    out = objective_function(dynamic_pattern, target_pattern)
    out = jnp.sum(out)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# PLOT TARGET FIELD

if show_midplane and (show=="TARGET" or show=="FIELD"):
    target_field = smarter_reshape(target_pattern, co_resolution)
    fig, ax = plt.subplots(1, 1, figsize=(8.25, 8.25))
    ax.imshow(target_field[:, :].T,
              cmap="binary_r",
              extent=(-co_size[0] / 2, co_size[0] / 2,
                      -co_size[1] / 2, co_size[1] / 2),
              origin="lower"
              )
    cs1 = ax.contour(coor[0].reshape(co_res_new[0], co_res_new[1]),
                    coor[1].reshape(co_res_new[0], co_res_new[1]),
                    target_field[:, :],
                    cmap="plasma_r",
                    linewidths=2)
    ax.clabel(cs1, inline=True, fontsize=10)
    for i in range(len(x_target)):
        ax.axvline(x=x_target[i], color='green', linestyle='--', linewidth=1)
        ax.axhline(y=y_target[i], color='green', linestyle='--', linewidth=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()
    fig.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# OPTIMIZATION PARAMETERS

# maximum number of optimization iterations
max_iterations = 1000
# the optimizer will stop when the relative tolerance of the cost function is less than this number
relative_cost_difference = 1e-6
# the optimizer will stop when the cost function is less than this number
stop_cost_value = 1e-6
# start learning rate of the optimizer
start_learning_rate = 0.01

# ----------------------------------------------------------------------------------------------------------------------
# FIND SOLUTION

# initialize parameters of the model and optimizer
params = jnp.array([x, y, jnp.zeros(len(x))])
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(params)

# optimization/update loop
prev_value = 1e16
best_value, best_params = 1e16, params.copy()
for i in range(max_iterations):
    value, grads = value_and_grad(worker)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    print(f"Iteration: {i + 1}", "Value:", value, "Parameters:", params, "Gradient:", grads)

    if value < stop_cost_value:
        break

    rtol = jnp.abs(value / prev_value - 1)
    if rtol < relative_cost_difference:
        break
    else:
        prev_value = value.copy()
        print("Relative tolerance:", rtol)

    if value < best_value:
        best_value, best_params = value, params.copy()

    params = optax.apply_updates(params, updates)

# get final results
value, grads = value_and_grad(worker)(params)
if value > best_value:
    value, params = best_value, best_params.copy()

print("\nFinal results:", "\nValue:", value, "\nParameters:", params, "\nGradient:", grads, "\n")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE SOLUTION

x_solution, y_solution, z_solution = tuple(params)

if show == "OBJECTIVE":
    def shown_field(x_, y_):
        v = jnp.asarray([x_, y_, jnp.zeros(len(x_))])
        return objective_function(combined_dynamic_field(v), target_sdf(x_target, y_target))
else:
    def shown_field(x_, y_):
        v = jnp.asarray([x_, y_, jnp.zeros(len(x_))])
        return combined_dynamic_field(v)

# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = shown_field(x_solution, y_solution)

# calculate the Jacobian with respect to the x-position (argnums=0), or y-position (argnums=1)
# of the first point (point_index = 0).
# note thst the first point in the solution does not necessarily represent the first target point.
point_index = 0
gradient_pattern = jacfwd(shown_field, argnums=0)(x_solution, y_solution)[:, point_index]

# convert the final pattern and the gradient pattern into 2D images
field = smarter_reshape(final_pattern, co_resolution)
gradient_field = smarter_reshape(gradient_pattern, co_resolution)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time - start_time))

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
    for i in range(len(x_target)):
        axs[0].axvline(x=x_target[i], color='green', linestyle='--', linewidth=2)
        axs[0].axhline(y=y_target[i], color='green', linestyle='--', linewidth=2)
    for i in range(len(x_solution)):
        axs[0].axvline(x=x_solution[i], color='grey', linestyle='--', linewidth=1)
        axs[0].axhline(y=y_solution[i], color='grey', linestyle='--', linewidth=1)

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
    for i in range(len(x_target)):
        axs[0].axvline(x=x_target[i], color='green', linestyle='--', linewidth=2)
        axs[0].axhline(y=y_target[i], color='green', linestyle='--', linewidth=2)
    for i in range(len(x_solution)):
        axs[0].axvline(x=x_solution[i], color='grey', linestyle='--', linewidth=1)
        axs[0].axhline(y=y_solution[i], color='grey', linestyle='--', linewidth=1)

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
