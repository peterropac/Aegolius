# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.ndimage import convolve
from typing import Callable
from spomso.cores.helper_functions import smarter_reshape


class PostProcess:
    """Class containing all the possible post-processing operations which can be applied to a scalar field.

    Attributes:
        unprocessed_geo_object: Original scalar field.
        processed_geo_object: Modified scalar field.

    Args:
        geo_object: Scalar field.
    """

    def __init__(self, geo_object: Callable[[np.ndarray, tuple], np.ndarray]):
        self._pmod = []
        self.unprocessed_geo_object = geo_object
        self.processed_geo_object = geo_object

    @property
    def post_processing_operations(self) -> list:
        """
        All the post-processing operations which were applied to the geometry in chronological order.
        
        Returns:
             List of post-processing operations.
        """
        return self._pmod

    @property
    def processed_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        SDF of the modified geometry.
        
        Returns:
             SDF of the modified geometry.
        """
        return self.processed_geo_object

    @property
    def unprocessed_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        SDF of the unmodified geometry.
        
        Returns:
             SDF of the unmodified geometry.
        """
        return self.unprocessed_geo_object

    def sigmoid_falloff(self, amplitude: float | int, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies a sigmoid to the scalar (Signed Distance Function) field.
        
        Args:
            amplitude: Maximum value of the transformed scalar field.
            width: Width of the sigmoid.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("sigmoid_falloff")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return sigmoid_falloff(u, amplitude, width)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def positive_sigmoid_falloff(self,
                                 amplitude: float | int,
                                 width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies a sigmoid, shifted to the positive velues by the value of the width parameter,
        to the scalar (Signed Distance Function) field.
        
        Args:
            amplitude: Maximum value of the transformed scalar field.
            width: Width of the sigmoid.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("positive_sigmoid_falloff")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return positive_sigmoid_falloff(u, amplitude, width)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def capped_exponential(self,
                           amplitude: float | int,
                           width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies a decreasing exponential functon to the scalar (Signed Distance Function) field.
        to the scalar (Signed Distance Function) field.
        
        Args:
            amplitude: Maximum value of the transformed scalar field.
            width: Range at which the value of the transformed scalar field drops to almost zero.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("capped_exponential")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return capped_exponential(u, amplitude, width)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def hard_binarization(self, threshold: float) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Binarizes the Signed Distance field/pattern based on a threshold.
        Values below the threshold are 1 and values above are 0.
        
        Args:
            threshold: Binarization threshold.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("hard_binarization")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return hard_binarization(u, threshold)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def linear_falloff(self, amplitude: float | int, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies a decreasing linear function to the scalar (Signed Distance Function) field.
        
        Args:
            amplitude: Maximum value of the transformed scalar field.
            width: Range at which the value of the transformed scalar field drops to zero.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("linear_falloff")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return linear_falloff(u, amplitude, width)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def relu(self, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies the ReLU function to the scalar (Signed Distance Function) field.
        
        Args:
            width: Range at which the value of the transformed field reaches one.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("relu")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return relu(u, width)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def smooth_relu(self, smooth_width: float | int,
                    width: float | int = 1, threshold: int | float = 0.01) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies the "squareplus" function to the scalar (Signed Distance Function) field.
        https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        
        Args:
            smooth_width: Distance from the origin at which the Smooth ReLU function
                is greater than ReLU for less than the value of the threshold parameter.
            width: Range at which the value of the transformed field reaches one.
            threshold: At smooth_width distance from the origin the value of the Smooth ReLU function is greater
                than ReLU for the value of the threshold parameter.
                at smooth_width distance from the origin.
        
        Returns:
            Transformed scalar field.
        """
        
        self._pmod.append("smooth_relu")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return smooth_relu(u, smooth_width, width, threshold)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def slowstart(self, smooth_width: float | int,
                  width: float | int = 1,
                  threshold: int | float = 0.01,
                  ground: bool = True) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies the SlowStart function to the scalar (Signed Distance Function) field.
        
        Args:
            smooth_width: Distance from the origin at which the SlowStart function
                is greater than ReLU for less than the value of the threshold parameter.
            width: Range at which the value of the transformed field reaches one.
            threshold: At smooth_width distance from the origin the value of the SlowStart function is greater
                than ReLU for the value of the threshold parameter.
            ground: if True the value of the function is zero at zero.
        
        Returns:
            Transformed scalar field.
        """

        self._pmod.append("slowstart")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return slowstart(u, smooth_width, width, threshold, ground=ground)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def gaussian_boundary(self, amplitude: float | int, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies the Gaussian to the scalar (Signed Distance Function) field.
        
        Args:
            amplitude: Maximum value of the transformed scalar field.
            width: Range at which the value of the transformed scalar field drops to almost zero.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("gaussian_boundary")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return gaussian_boundary(u, amplitude, width)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def gaussian_falloff(self, amplitude: float | int, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies the Gaussian to the positive values of the scalar (Signed Distance Function) field.
        
        Args:
            amplitude: Maximum value of the transformed scalar field (and points at which the scalar field was < 0).
            width: Range at which the value of the transformed scalar field drops to almost zero.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("gaussian_falloff")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return gaussian_falloff(u, amplitude, width)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def conv_averaging(self,
                       kernel_size: int | tuple | list | np.ndarray,
                       iterations: int,
                       co_resolution: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Averages the field using an averaging convolutional kernel of the specified size.
        
        Args:
            kernel_size: Size of the averaging kernel. Must be an integer or a tuple/array of the
                same dimension as the scaler field.
            iterations: Number of times the convolutional averaging is applied to the input scalar field.
            co_resolution: Resolution of the coordinate system.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("conv_averaging")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            u = smarter_reshape(u, co_resolution)
            out = conv_averaging(u, kernel_size, iterations)
            return out.flatten()

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def conv_edge_detection(self,
                            co_resolution: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Edge detection based on a 3x3 convolutional kernel.
        
        Args:
            co_resolution: Resolution of the coordinate system.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append("conv_edge_detection")
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            u = smarter_reshape(u, co_resolution)
            return conv_edge_detection(u)

        self.processed_geo_object = new_geo_object
        return new_geo_object

    def custom_post_process(self,
                            function: Callable[[np.ndarray, tuple], np.ndarray],
                            parameters: tuple,
                            post_process_name: str = "custom") -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies a custom user-specified post-processing function to a scalar (Signed Distance Function) field.
        
        Args:
            function: A custom post-processing function which takes the SDF as a first argument
                and the parameters of the function as the next arguments.
            parameters: Parameters of the custom post-processing function.
            post_process_name: Name of the custom post-processing function.
        
        Returns:
             Modified Scalar Field.
        """
        
        self._pmod.append(post_process_name)
        geo_object = self.processed_geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return function(u, *parameters)

        self.processed_geo_object = new_geo_object
        return new_geo_object


# ----------------------------------------------------------------------------------------------------------------------
# SCALAR POSTPROCESSING FUNCTIONS

def sigmoid_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a sigmoid to the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Width of the sigmoid.
    
    Returns:
        Transformed scalar field.
    """
    e = np.exp(4 * u / width)

    return amplitude*(1/(1 + e))


def positive_sigmoid_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a sigmoid, shifted to the positive velues by the value of the width parameter,
    to the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Width of the sigmoid.
    
    Returns:
        Transformed scalar field.
    """
    e = np.exp(4 * (u - width) / width)

    return amplitude*(1/(1 + e))


def capped_exponential(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a decreasing exponential functon to the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to almost zero.
    
    Returns:
        Transformed scalar field.
    """
    e = np.exp(- 4 * u / width)

    return amplitude * np.minimum(e, 1)


def hard_binarization(u: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binarizes the Signed Distance field/pattern based on a threshold.
    Values below the threshold are 1 and values above are 0.
    
    Args:
        u: Signed Distance field or any scalar field.
        threshold: Binarization threshold.
    
    Returns:
        Binarized scalar field.
    """
    out = u <= threshold
    out = out.astype(float)
    return out


def linear_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies a decreasing linear function to the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to zero.
    
    Returns:
        Transformed scalar field.
    """
    out = 1 - u/width

    return np.clip(out, 0, 1)*amplitude


def relu(u: np.ndarray, width: float | int = 1) -> np.ndarray:
    """
    Applies the ReLU function to the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        width: Range at which the value of the transformed field reaches one.
    
    Returns:
        Transformed scalar field.
    """
    return np.maximum(u/width, 0)


def smooth_relu(u: np.ndarray, smooth_width: float | int,
                width: float | int = 1, threshold: int | float = 0.01) -> np.ndarray:
    """
    Applies the "squareplus" function to the scalar (Signed Distance Function) field.
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    
    Args:
        u: Signed Distance field or any scalar field.
        smooth_width: Distance from the origin at which the Smooth ReLU function
            is greater than ReLU for less than the value of the threshold parameter.
        width: Range at which the value of the transformed field reaches one.
        threshold: At smooth_width distance from the origin the value of the Smooth ReLU function is greater
            than ReLU for the value of the threshold parameter.
            at smooth_width distance from the origin.
    
    Returns:
        Transformed scalar field.
    """
    b = (smooth_width + threshold)*4*threshold
    v = u/width
    return (v + np.sqrt(v**2 + b))/2


def slowstart(u: np.ndarray,
              smooth_width: float | int,
              width: float | int = 1, threshold: int | float = 0.01,
              ground: bool = True) -> np.ndarray:
    """
    Applies the SlowStart function to the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        smooth_width: Distance from the origin at which the SlowStart function
            is greater than ReLU for less than the value of the threshold parameter.
        width: Range at which the value of the transformed field reaches one.
        threshold: At smooth_width distance from the origin the value of the SlowStart function is greater
            than ReLU for the value of the threshold parameter.
        ground: if True the value of the function is zero at zero.
    
    Returns:
        Transformed scalar field.
    """
    b = (2*smooth_width + threshold)*threshold
    return np.sqrt(np.maximum(u / width, 0)**2 + b/width) - np.sqrt(b/width)*ground


def gaussian_boundary(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies the Gaussian to the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field.
        width: Range at which the value of the transformed scalar field drops to almost zero.
    
    Returns:
        Transformed scalar field.
    """
    out = np.exp(-4*(u/width)**2)

    return amplitude*out


def gaussian_falloff(u: np.ndarray, amplitude: float | int, width: float | int) -> np.ndarray:
    """
    Applies the Gaussian to the positive values of the scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        amplitude: Maximum value of the transformed scalar field (and points at which the scalar field was < 0).
        width: Range at which the value of the transformed scalar field drops to almost zero.
    
    Returns:
        Transformed scalar field.
    """
    u = np.maximum(u, 0)
    out = np.exp(-4*(u/width)**2)

    return amplitude*out


def conv_averaging(u: np.ndarray, kernel_size: int | tuple | list | np.ndarray, iterations: int) -> np.ndarray:
    """
    Averages the field using an averaging convolutional kernel of the specified size.
    
    Args:
        u: Signed Distance field or any scalar field.
        kernel_size: Size of the averaging kernel. Must be an integer or a tuple/array of the
            same dimension as the scaler field.
        iterations: Number of times the convolutional averaging is applied to the input scalar field.
    
    Returns:
        Transformed scalar field.
    """
    if iterations == 0:
        return u

    shape_u = u.shape

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)*len(shape_u)
    if isinstance(kernel_size, np.ndarray):
        kernel_size = tuple(kernel_size)
    if isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)

    if not len(kernel_size) == len(shape_u):
        raise ValueError("Dimension of the kernel and the field must match!")

    if len(kernel_size)==2:
        norm_ = kernel_size[0]*kernel_size[1]
    if len(kernel_size) == 3:
        norm_ = kernel_size[0] * kernel_size[1] * kernel_size[2]

    filter_ = np.ones(kernel_size) / norm_

    new = convolve(u, filter_)
    for i in range(iterations):
        new = convolve(new, filter_)

    return new


def conv_edge_detection(u: np.ndarray) -> np.ndarray:
    """
    Edge detection based on a 3x3 convolutional kernel.
    
    Args:
        u: Signed Distance field or any scalar field.
    
    Returns:
        Transformed scalar field.
    """

    if len(u.shape) == 2:
        filter_ = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    if len(u.shape) == 3:
        f = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filter_ = np.zeros((3,3,1))
        filter_[:,:,:] = np.asarray([f]).T

    new = convolve(u, filter_)

    return new


def custom_post_process(u: np.ndarray,
                        function: Callable[[np.ndarray, tuple], np.ndarray],
                        parameters: tuple) -> np.ndarray:
    """
    Applies a custom user-specified post-processing function to a scalar (Signed Distance Function) field.
    
    Args:
        u: Signed Distance field or any scalar field.
        function: A custom post-processing function which takes the SDF as a first argument
            and the parameters of the function as the next arguments.
        parameters: Parameters of the custom post-processing function.
    
    Returns:
        Modified Scalar Field.
    """

    return function(u, *parameters)

