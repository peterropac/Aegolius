import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from time import process_time

from spomso.cores.geom import GenericGeometry

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.geom_2d import Circle

from spomso.cores.post_processing import gaussian_boundary
from spomso.cores.post_processing import PostProcess

# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS

# size of the volume
co_size = 4, 4
# resolution of the volume
co_resolution = 400, 400

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

# the SDF field can be transformed into other scalar fields with various operations.
# This can be done using 3 different approaches:
# MODIFICATION: applying a modification to the geometry.
# FUNCTION: applying a function to the signed distance field 2D map.
# OOP: using the PostProcess class to apply the post-processing to the SDF.
# All the approaches are shown below:

# there are a few such predefined post-processing functions included in the package.
# In the examples below the **Gaussian Boundary** post-processing function is used.

# FUNCTION
# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = final.create(coor)
# convert the final pattern into a 2D image
field = smarter_reshape(final_pattern, co_resolution)
# set the amplitude of the gaussian boundary post-processing function to 1 and the width to 0.5
gb_function = gaussian_boundary(field, 1.0, 0.5)

# OOP
# create the PostProcess class instance and pass the SDF of the geometry to the instance
post_process = PostProcess(final.propagate)
# set the amplitude of the gaussian boundary post-processing function to 1 and the width to 0.5
post_process.gaussian_boundary(1.0, 0.5)
# create a new geometry from the post-processed SDF
gb_geo = GenericGeometry(post_process.processed_geo_object, ())
# evaluate the SDF of the geometry to create a signed distance field 2D map
gb_geo_field = gb_geo.create(coor)
# convert the map into a 2D image
gb_oop = smarter_reshape(gb_geo_field, co_resolution)
# print the post-processing type
print("Post-processing type:", post_process.post_processing_operations)

# MODIFICATION
# set the amplitude of the gaussian boundary post-processing function to 1 and the width to 0.5
final.gaussian_boundary(1.0, 0.5)
# evaluate the SDF of the geometry to create a signed distance field 2D map
gb_modification_pattern = final.create(coor)
# convert the final pattern into a 2D image
gb_modification = smarter_reshape(gb_modification_pattern, co_resolution)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show_midplane:
    fig, axs = plt.subplots(2, 2, figsize=(8.25, 8.25))

    patterns = ((field, gb_modification), (gb_function, gb_oop))
    titles = (("SDF", "Gaussian Boundary\nModification"), ("Gaussian Boundary\nFunction", "Gaussian Boundary\nOOP"))
    for i in range(2):
        for j in range(2):
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
    patterns = ((field, gb_modification), (gb_function, gb_oop))
    titles = ("SDF", "Gaussian Boundary - Modification", "Gaussian Boundary - Function", "Gaussian Boundary - OOP")

    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'is_3d': True}, {'is_3d': True}],
                               [{'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=titles
                        )

    for i in range(2):
        for j in range(2):
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

