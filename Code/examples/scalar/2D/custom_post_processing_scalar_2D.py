import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from time import process_time

from spomso.cores.geom import GenericGeometry

from spomso.cores.helper_functions import generate_grid, smarter_reshape
from spomso.cores.geom_2d import Circle

from spomso.cores.post_processing import custom_post_process
from spomso.cores.post_processing import PostProcess

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS


def sinc(u, amplitude, width):
    return amplitude*np.sinc(u/width)


post_processing_function_name = "Sinc"

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

# the SDF field can be transformed into other scalar fields with various operations -
# including user specified functions.
# This can be done using 3 different approaches:
# MODIFICATION: applying a modification to the geometry.
# FUNCTION: applying a function to the signed distance field 2D map.
# OOP: using the PostProcess class to apply the post-processing to the SDF.
# All the approaches are shown below:

# FUNCTION
# evaluate the SDF of the geometry to create a signed distance field 2D map
final_pattern = final.create(coor)
# convert the final pattern into a 2D image
field = smarter_reshape(final_pattern, co_resolution)
# set the amplitude of the custom sinc post-processing function to 1 and the width to 0.5
custom_function = custom_post_process(field, sinc, (1.0, 0.5))

# OOP
# create the PostProcess class instance and pass the SDF of the geometry to the instance
post_process = PostProcess(final.propagate)
# set the amplitude of the custom sinc post-processing function to 1 and the width to 0.5
post_process.custom_post_process(sinc, (1.0, 0.5),
                                 post_process_name=post_processing_function_name)
# create a new geometry from the post-processed SDF
custom_geo = GenericGeometry(post_process.processed_geo_object, ())
# evaluate the SDF of the geometry to create a signed distance field 2D map
custom_geo_field = custom_geo.create(coor)
# convert the map into a 2D image
custom_oop = smarter_reshape(custom_geo_field, co_resolution)
# print the post-processing type
print("Post-processing type:", post_process.post_processing_operations)

# MODIFICATION
# set the amplitude of the custom sinc post-processing function to 1 and the width to 0.5
final.custom_post_process(sinc, (1.0, 0.5),
                          post_process_name=post_processing_function_name)
# evaluate the SDF of the geometry to create a signed distance field 2D map
custom_modification_pattern = final.create(coor)
# convert the final pattern into a 2D image
custom_modification = smarter_reshape(custom_modification_pattern, co_resolution)
# print the modifications
print("Modifications:", final.modifications)

end_time = process_time()
print("Evaluation Completed in {:.2f} seconds".format(end_time-start_time))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT

if show_midplane:
    fig, axs = plt.subplots(2, 2, figsize=(8.25, 8.25))

    patterns = ((field, custom_modification), (custom_function, custom_oop))
    titles = (("SDF", f"{post_processing_function_name}\nModification"),
              (f"{post_processing_function_name}\nFunction", f"{post_processing_function_name}\nOOP"))
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
    patterns = ((field, custom_modification), (custom_function, custom_oop))
    titles = ("SDF",
              f"{post_processing_function_name} - Modification",
              f"{post_processing_function_name} - Function",
              f"{post_processing_function_name} - OOP")

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

