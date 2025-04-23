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

# define the radius of the smoothed circle
radius = 1.

# define the initial position of the smoothed circle
x = 0.0
y = 0.0

# define the target position of the smoothed circle
x_target = 2.5
y_target = -1.0

# the optimization can also be done using SDFs - without any additional post-processing functions
pure_sdf = False

# ----------------------------------------------------------------------------------------------------------------------
# CREATE SDFs

start_time = process_time()


# define a target field where the input is the target position of the smoothed circle
def target_circle(x0, y0):
    vec = jnp.asarray([x0, y0, 0])

    if pure_sdf:
        out = compound_euclidian_transform_sdf(sdf_circle, jnp.eye(3), vec, 1.)(coor, radius)
    else:
        circle = gaussian_falloff(sdf_circle, 1., 0.5)
        circle = compound_euclidian_transform_sdf(circle, jnp.eye(3), vec, 1.)
        out = circle(coor, radius)

    return out


# define the dynamic field of the smoothed circle where the input is its position
def smoothed_circle(vec):
    co = jnp.subtract(coor.T, vec).T + 0.001

    if pure_sdf:
        out = sdf_circle(co, radius)
    else:
        circle = gaussian_falloff(sdf_circle, 1., 0.5)
        out = circle(co, radius)

    return out


# define the objective/cost function, where the inputs are the smoothed circle field and the target field
def objective_function(f_, t_):
    m = (f_ - t_)**2
    return m


# ----------------------------------------------------------------------------------------------------------------------
# PRECOMPUTE

# compute the target field
target_pattern = target_circle(x_target, y_target)


# define the worker for the optax optimizer
def worker(p):
    dynamic_pattern = smoothed_circle(p)
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
    ax.axvline(x=x_target, color='green', linestyle='--', linewidth=1)
    ax.axhline(y=y_target, color='green', linestyle='--', linewidth=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()
    fig.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# OPTIMIZATION PARAMETERS

# maximum number of optimization iterations
max_iterations = 500
# the optimizer will stop when the relative tolerance of the cost function is less than this number
relative_cost_difference = 1e-6
# the optimizer will stop when the cost function is less than this number
stop_cost_value = 1e-6
# start learning rate of the optimizer
start_learning_rate = 0.05

# ----------------------------------------------------------------------------------------------------------------------
# FIND SOLUTION

# initialize parameters of the model and optimizer
params = jnp.array([x, y, 0])
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(params)

# optimization/update loop
prev_value = 1e16
best_value, best_params = 1e16, jnp.array([x, y, 0])
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
    def shown_field(x_, y_): return objective_function(smoothed_circle(jnp.asarray([x_, y_, 0])),
                                                       target_circle(x_target, y_target))
else:
    def shown_field(x_, y_): return smoothed_circle(jnp.asarray([x_, y_, 0]))

# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = shown_field(x_solution, y_solution)

# calculate the Jacobian with respect to the x-position (argnums=0), y-position (argnums=1)
gradient_pattern = jacfwd(shown_field, argnums=0)(x_solution, y_solution)

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
    axs[0].axvline(x=x_target, color='green', linestyle='--', linewidth=2)
    axs[0].axhline(y=y_target, color='green', linestyle='--', linewidth=2)
    axs[0].axvline(x=x_solution, color='grey', linestyle='--', linewidth=1)
    axs[0].axhline(y=y_solution, color='grey', linestyle='--', linewidth=1)

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
    axs[1].axvline(x=x_target, color='green', linestyle='--', linewidth=2)
    axs[1].axhline(y=y_target, color='green', linestyle='--', linewidth=2)
    axs[1].axvline(x=x_solution, color='grey', linestyle='--', linewidth=1)
    axs[1].axhline(y=y_solution, color='grey', linestyle='--', linewidth=1)

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
