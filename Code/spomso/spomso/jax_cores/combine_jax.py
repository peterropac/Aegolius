# Copyright (C) 2025 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.


import jax.numpy as jnp
import numpy as np
import jax
from typing import Callable

array_like_type = jnp.ndarray | np.ndarray | list | tuple
scalar_like_type = float | int
function_like_type = Callable[[array_like_type, tuple], jnp.ndarray]
combine_function_type = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
combine_multiple_function_type = Callable[[jnp.ndarray], jnp.ndarray]
parametric_combine_function_type = Callable[[jnp.ndarray, jnp.ndarray, scalar_like_type], jnp.ndarray]


@jax.jit
def union2(obj1: jnp.ndarray, obj2: jnp.ndarray) -> jnp.ndarray:
    """
    Union boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return jnp.minimum(obj1, obj2)


@jax.jit
def union(objs: array_like_type) -> jnp.ndarray:
    """
    Union boolean operation between multiple SDFs.

    Args:
        objs: Array of SDFs, evaluated at the coordinate points.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    objs = jnp.asarray(objs)
    return jnp.amin(objs, axis=0)


@jax.jit
def subtract2(obj1: array_like_type, obj2: array_like_type) -> jnp.ndarray:
    """
    Subtract boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return jnp.maximum(obj1, -obj2)


@jax.jit
def intersect2(obj1: array_like_type, obj2: array_like_type) -> jnp.ndarray:
    """
    Intersect boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return jnp.maximum(obj1, obj2)


@jax.jit
def intersect(objs: array_like_type) -> jnp.ndarray:
    """
    Intersect boolean operation between multiple SDFs.

    Args:
        objs: Array of SDFs, evaluated at the coordinate points.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    objs = jnp.asarray(objs)
    return jnp.amax(objs, axis=0)

@jax.jit
def add(obj1: array_like_type, obj2: array_like_type) -> jnp.ndarray:
    """
    Computes the sum of two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return jnp.add(obj1, obj2)

@jax.jit
def difference(obj1: array_like_type, obj2: array_like_type) -> jnp.ndarray:
    """
    Computes the difference between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return jnp.subtract(obj1, obj2)


@jax.jit
def smoothmin_poly2(x: array_like_type | scalar_like_type, y: array_like_type | scalar_like_type,
                    a: scalar_like_type) -> array_like_type | scalar_like_type:
    """
    Computes the second order smooth minimum.
    https://iquilezles.org/articles/smin/

    Args:
        x: First value.
        y: Second value.
        a: Smoothing parameter.
    Returns:
        Smooth minimum between x and y, based on parameter a.
    """
    a = jnp.abs(a) + 0.001
    h = jnp.maximum(a - jnp.abs(x - y), 0.0)/a
    return jnp.minimum(x, y) - h * h * a/4.0


@jax.jit
def smoothmin_poly3(x: array_like_type | scalar_like_type, y: array_like_type | scalar_like_type,
                    a: scalar_like_type) -> array_like_type | scalar_like_type:
    """
    Computes the third order smooth minimum.
    https://iquilezles.org/articles/smin/

    Args:
        x: First value.
        y: Second value.
        a: Smoothing parameter.
    Returns:
        Smooth minimum between x and y, based on parameter a.
    """
    a = jnp.abs(a) + 0.001
    h = jnp.maximum(a - jnp.abs(x - y), 0.0) / a
    return jnp.minimum(x, y) - h * h * h * a / 6.0


@jax.jit
def smoothmax_boltz(x: array_like_type | scalar_like_type, y: array_like_type | scalar_like_type,
                    a: scalar_like_type) -> array_like_type | scalar_like_type:
    """
    Computes the smooth maximum.
    https://en.wikipedia.org/wiki/Smooth_maximum

    Args:
        x: First value.
        y: Second value.
        a: Smoothing parameter.
    Returns:
        Smooth maximum between x and y, based on parameter a.
    """
    exp1 = jnp.exp(x/a)
    exp2 = jnp.exp(y/a)

    return (x*exp1 + y*exp2)/(exp1 + exp2)


@jax.jit
def smooth_union2_2o(obj1: array_like_type, obj2: array_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Second order parametric union boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
        width: Smoothing parameter.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return smoothmin_poly2(obj1, obj2, width)


@jax.jit
def smooth_union2_3o(obj1: array_like_type, obj2: array_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Third order parametric union boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
        width: Smoothing parameter.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return smoothmin_poly3(obj1, obj2, width)


@jax.jit
def smooth_intersect2_2o(obj1: array_like_type, obj2: array_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Second order parametric intersect boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
        width: Smoothing parameter.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return -smoothmin_poly2(-obj1, -obj2, width)


@jax.jit
def smooth_intersect2_3o(obj1: array_like_type, obj2: array_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Third order parametric intersect boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
        width: Smoothing parameter.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return -smoothmin_poly3(-obj1, -obj2, width)


@jax.jit
def smooth_subtract2_2o(obj1: array_like_type, obj2: array_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Second order parametric subtract boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
        width: Smoothing parameter.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return -smoothmin_poly2(-obj1, obj2, width)


@jax.jit
def smooth_subtract2_3o(obj1: array_like_type, obj2: array_like_type, width: scalar_like_type) -> jnp.ndarray:
    """
    Third order parametric subtract boolean operation between two SDFs.

    Args:
        obj1: First SDF, evaluated at the coordinate points.
        obj2: Second SDF, evaluated at the coordinate points.
        width: Smoothing parameter.
    Returns:
        Combined SDF, evaluated at the coordinate points.
    """
    return -smoothmin_poly3(-obj1, obj2, width)


def combine_2_sdfs(function_1: function_like_type, function_2: function_like_type,
                   params_1: tuple, params_2: tuple,
                   combine_function: combine_function_type) -> function_like_type:
    """
    Function that combines two SDFs, based on the parameters and the 'combine_function', and returns the combined SDF.

    Args:
        function_1: Function defining the first SDF.
        function_2: Function defining the second SDF.
        params_1: Parameters for the first SDF.
        params_2: Parameters for the second SDF.
        combine_function: Parameter-less function that combines two SDFs.
    Returns:
        Combined SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return combine_function(function_1(co, *params_1), function_2(co, *params_2))

    return new_geo_object


def combine_multiple_sdfs(functions_array: array_like_type,
                          params_array: array_like_type,
                          combine_function: combine_multiple_function_type) -> function_like_type:
    """
    Function that combines two SDFs, based on the parameters and the 'combine_function', and returns the combined SDF.

    Args:
        functions_array: Array of functions defining the SDFs to be combined.
        params_array: Array of parameters of the SDFs in functions_array.
        combine_function: Parameter-less function that combines multiple SDFs.
    Returns:
        Combined SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        f_array = jnp.zeros((len(functions_array), co.shape[1]))
        for i in range(len(functions_array)):
            f_array = f_array.at[i, ...].set(functions_array[i](co, *params_array[i]))
        return combine_function(f_array)

    return new_geo_object


def parametric_combine_2_sdfs(function_1: function_like_type, function_2: function_like_type,
                              params_1: tuple, params_2: tuple,
                              combine_function: parametric_combine_function_type,
                              combine_parameter: scalar_like_type) -> function_like_type:
    """
    Function that combines two SDFs, based on the parameters and the parametric 'combine_function', and returns the combined SDF.

    Args:
        function_1: Function defining the first SDF.
        function_2: Function defining the second SDF.
        params_1: Parameters for the first SDF.
        params_2: Parameters for the second SDF.
        combine_function: Parametric function that combines two SDFs.
        combine_parameter: Parameter of the 'combine_function'.
    Returns:
        Combined SDF.
    """

    @jax.jit
    def new_geo_object(co, *params):
        return combine_function(function_1(co, *params_1),
                                function_2(co, *params_2),
                                combine_parameter)

    return new_geo_object

