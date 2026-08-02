import numpy as np
import matplotlib.pyplot as plt

from time import process_time

from jax import value_and_grad, config
import jax.numpy as jnp
import optax

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.jax_cores.vector_functions_jax import vortex_vector_field_cylindrical
from spomso.jax_cores.sdf_2D_jax import sdf_circle
from spomso.jax_cores.post_processing_jax import gaussian_boundary_jax

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the 2D domain
co_size = 8, 8
co_resolution = 400, 400

# radius of the Gaussian envlope of each vortex
radius = 1.0
# sigma of the Gaussian envelope on each vortex
sigma = 2.0

# particle handedness
handedness_1 = 1
handedness_2 = -1

# monitor line
line_y = 0.0
line_extent = 6.0
n_line_samples = 500

# random seed for the initial particle positions
seed = 24072026

# FIELD (vector field with quiver overlay), PROJECTION (x-component of the field)
show = "FIELD"
show_midplane = True
show_projection = True

# ----------------------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEMS

coor, co_res_new = generate_grid(co_size, co_resolution)
line_x = jnp.linspace(-line_extent/2, line_extent/2, n_line_samples)
line_coor = jnp.stack([line_x, jnp.full_like(line_x, line_y), jnp.zeros_like(line_x)])

# ----------------------------------------------------------------------------------------------------------------------
# VECTOR FIELD

# vortex vector field attenuated by a potential
def vortex(coor_, x0, y0, handedness):
    p = jnp.subtract(coor_.T, jnp.asarray([x0, y0, 0])).T
    vf = vortex_vector_field_cylindrical(p)
    sdf = sdf_circle(p, radius)
    f = gaussian_boundary_jax(sdf, 1, sigma)
    out = vf * f * handedness
    return out[:2]

# define the total field, which the sum of both vortex vector fields
def total_field(coor_, params):
    x1, y1, x2, y2 = params
    return vortex(coor_, x1, y1, handedness_1) + vortex(coor_, x2, y2, handedness_2)

# define the worker for the optax optimizer
def worker(p):
    field_on_line = total_field(line_coor, p)
    projection = field_on_line[0]
    return -jnp.max(projection)


# ----------------------------------------------------------------------------------------------------------------------
# INITIAL CONDITION

# Random initial positions. To avoid the wrong local minimum where both particles
# end up on the same side of the line, the initial position of one particle is above and
# the initial position of the other particle is below the line.
rng = np.random.default_rng(seed)
x1_init = float(rng.uniform(-co_size[0]/2, co_size[0]/2))
x2_init = float(rng.uniform(-co_size[0]/2, co_size[0]/2))
y1_init = float(rng.uniform(0.3, co_size[1]/2))
y2_init = float(-rng.uniform(0.3, co_size[1]/2))
init_params = jnp.asarray([x1_init, y1_init, x2_init, y2_init])
print(f"Initial parameters: p1 = ({x1_init:.3f}, {y1_init:.3f}), "
      f"p2 = ({x2_init:.3f}, {y2_init:.3f})")

# ----------------------------------------------------------------------------------------------------------------------
# OPTIMIZATION PARAMETERS

# maximum number of optimization iterations
max_iterations = 1000
# the optimizer will stop when the relative tolerance of the cost function is less than this number
relative_cost_difference = 1e-10
# the optimizer will stop when the cost function is less than this number
stop_cost_value = -1e12
# start learning rate of the optimizer
start_learning_rate = 0.1

# ----------------------------------------------------------------------------------------------------------------------
# FIND SOLUTION

start_time = process_time()

# initialize parameters of the model and optimizer
params = init_params
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(params)

# trajectory storage for plotting
trajectory = [np.asarray(params)]
cost_history = []

# optimization/update loop
prev_value = 1e16
best_value, best_params = 1e16, params
for i in range(max_iterations):
    value, grads = value_and_grad(worker)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    print(f"Iteration: {i + 1}  Value: {value:.4f}  Parameters: {params}")

    cost_history.append(float(value))

    if value < stop_cost_value:
        break

    rtol = jnp.abs(value / prev_value - 1)
    if rtol < relative_cost_difference:
        break
    else:
        prev_value = value.copy()
        print("Relative tolerance:", rtol)

    params = optax.apply_updates(params, updates)
    if value < best_value:
        best_value, best_params = value, params.copy()

    trajectory.append(np.asarray(params))

# get final results
value, _ = value_and_grad(worker)(params)
if value > best_value:
    value, params = best_value, best_params

end_time = process_time()
print(f"\nOptimization completed in {end_time - start_time:.2f} s")
print(f"Final value: {value:.4f}")
print(f"Final parameters: p1 = ({params[0]:.3f}, {params[1]:.3f}), "
      f"p2 = ({params[2]:.3f}, {params[3]:.3f})")
print(f"Gradients: {grads}")

x1_sol, y1_sol, x2_sol, y2_sol = tuple(np.asarray(params))
trajectory = np.asarray(trajectory)

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE FINAL FIELD ON THE GRID

field_flat = total_field(coor, params)

if show == "PROJECTION":
    displayed = field_flat[0]
    displayed_label = "field projection onto monitor line direction"
else:
    displayed = jnp.linalg.norm(field_flat, axis=0)
    displayed_label = "field magnitude"

displayed_grid = smarter_reshape(displayed, co_resolution)
field = np.asarray(displayed_grid)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show_midplane:

    fig, ax = plt.subplots(1, 1, figsize=(8.25, 8.25))

    ax.imshow(
        displayed_grid.T,
        cmap="plasma",
        extent=(-co_size[0] / 2, co_size[0] / 2, -co_size[1] / 2, co_size[1] / 2),
        origin="lower",
    )

    # quiver overlay: subsample coordinates
    step_x = co_resolution[0] // 30
    step_y = co_resolution[1] // 30
    res2d = (co_res_new[0], co_res_new[1])
    X = coor[0].reshape(res2d)[::step_x, ::step_y]
    Y = coor[1].reshape(res2d)[::step_x, ::step_y]
    U = np.asarray(field_flat[0]).reshape(res2d)[::step_x, ::step_y]
    V = np.asarray(field_flat[1]).reshape(res2d)[::step_x, ::step_y]
    ax.quiver(X, Y, U, V, color="white", scale_units="xy", scale=3.0, width=0.003, alpha=0.85)

    # monitor line
    ax.hlines(y=line_y,
               xmin=-line_extent/2, xmax=line_extent/2,
               color="red", linestyle="--", linewidth=2, label="monitor line")

    # trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="cyan", linewidth=1, alpha=0.7)
    ax.plot(trajectory[:, 2], trajectory[:, 3], color="magenta", linewidth=1, alpha=0.7)

    # initial and final positions
    ax.plot(x1_init, y1_init, "o", color="cyan", markersize=8, markeredgecolor=None,
                label="p1 initial")
    ax.plot(x2_init, y2_init, "o", color="magenta", markersize=8, markeredgecolor=None,
                label="p2 initial")
    ax.plot(x1_sol, y1_sol, "o", color="cyan", markersize=8, markeredgecolor=None,
                label="p1 final")
    ax.plot(x2_sol, y2_sol, "o", color="magenta", markersize=8, markeredgecolor=None,
                label="p2 final")

    ax.set_xlim(-co_size[0] / 2, co_size[0] / 2)
    ax.set_ylim(-co_size[1] / 2, co_size[1] / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{displayed_label} with vector field overlay")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# PROJECTION ALONG THE MONITOR LINE, initial vs final

if show_projection:
    fig, ax = plt.subplots(1, 1, figsize=(8.25, 4))

    init_field_on_line = total_field(line_coor, init_params)[0]
    final_field_on_line = total_field(line_coor, params)[0]
    ax.plot(np.asarray(line_x), np.asarray(init_field_on_line),
            color="grey", linestyle="--", label="initial")
    ax.plot(np.asarray(line_x), np.asarray(final_field_on_line),
            color="black", linewidth=2, label="final")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("x (along the monitor line, y = %.1f)" % line_y)
    ax.set_ylabel("field_x on the line")
    ax.set_title("projection along the monitor line")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()
