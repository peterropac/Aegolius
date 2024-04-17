# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from spomso.cores.transformations import EuclideanTransform, EuclideanTransformPoints
from spomso.cores.modifications import ModifyObject, ModifyVectorObject
from spomso.cores.helper_functions import resolution_conversion
from typing import Callable


class GenericGeometry(EuclideanTransform, ModifyObject):
    """Constructs geometry based on an SDF.

        Args:
            geo_sdf: SDF of a geometry - geo_sdf(co, *geo_parameters).
            geo_parameters: Parameters of the SDF.
    """

    def __init__(self, geo_sdf: Callable[[np.ndarray, tuple], np.ndarray], *geo_parameters: object):
        EuclideanTransform.__init__(self)
        ModifyObject.__init__(self, geo_sdf)
        self._geo_parameters = geo_parameters
        self._sdf = geo_sdf

    def create(self, co: np.ndarray) -> np.ndarray:
        """
        Applies the modifications and transformations (in that order) to the SDF and returns the map of the
        Signed Distance Field.

        Args:
            co: Point cloud of coordinates (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.

        Returns:
            Signed Distance Field of shape (N,).
        """
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, self._geo_parameters)

    def propagate(self, co: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications and transformations (in that order) to the SDF and returns the map of the
        Signed Distance Field. Similar to create but is meant to be used as input to construct new geometry.

        Args:
            co: Point cloud of coordinates (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty

        Returns:
            Signed Distance Field of shape (N,).
        """
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, self._geo_parameters)

    def point_cloud(self, co: np.ndarray) -> np.ndarray:
        """Creates a point cloud from the interior points of the SDF.

        Args:
            co: Point cloud of coordinates (D, N);
        """
        data_ = self.create(co)
        mask_ = data_ <= 0

        _points = np.zeros((3, np.count_nonzero(mask_)))
        _points[:2, :] = co[:2, mask_]

        return _points


class Points(EuclideanTransformPoints):
    """Constructs a Points object from a point cloud.

    Args:
        points: The point cloud of shape (D, N), where D is the spatial dimension and N is the number of points.
    """

    def __init__(self, points: np.ndarray | list | tuple):
        EuclideanTransformPoints.__init__(self)
        self._points = np.asarray(points)
        if self._points.size > 0:
            if self._points.shape[1] < self._points.shape[0]:
                self._points = self._points.T

    @property
    def cloud(self) -> np.ndarray:
        """Transformed point cloud.

        Returns:
            Transformed point cloud of shape (D, N).
        """
        return self.apply(self._points)

    def from_image(self,
                   greyscale_image: np.ndarray,
                   image_dimensions: tuple | list | np.ndarray,
                   binary_threshold: float = 0.5):
        """
        Creates a point cloud from a greyscale image.
        Positions of the points in the point cloud are calculated
        from the positions of the pixels with brightness below the binary threshold.

        Args:
            greyscale_image: The greyscale image.
            image_dimensions: A 2vector defining the size of the image.
            binary_threshold: Threshold used to filter which points belong in the point cloud.
        """
        data_ = np.asarray(greyscale_image)
        data_max = np.amax(data_)
        data_ = data_.astype(float)/data_max

        data_ = np.flip(data_, axis=0).T
        mask_ = data_ <= binary_threshold

        coi = np.asarray(
            np.meshgrid(np.linspace(-image_dimensions[0] / 2, image_dimensions[0] / 2, data_.shape[0]),
                        np.linspace(-image_dimensions[1] / 2, image_dimensions[1] / 2, data_.shape[1]),
                        indexing="ij")
        )

        _points = np.zeros((3, np.count_nonzero(mask_)))
        _points[:2, :] = coi[:, mask_]

        self._points = _points

    def to_image(self, co_size: tuple | np.ndarray,
                 co_resolution: tuple | np.ndarray,
                 extend: tuple) -> np.ndarray:
        """
        Converts a point cloud into a black and white image.
        Black where there are points and white where there are none.

        Args:
            co_size: Size of the 3D coordinate system (size_x, size_y, size_z).
            co_resolution: Resolution of the grid along all three dimensions (res_x, res_y, res_z).
            extend: Along which directions is the image extended (-X, +X, -Y, +Y, -Z, +Z).

        Returns:
            A grid of voxels with the shape (res_x, res_y, res_z) and values of 0 or 1.
        """

        cloud = self.cloud

        new_res = (resolution_conversion(co_resolution[0]),
                   resolution_conversion(co_resolution[1]),
                   resolution_conversion(co_resolution[2]))

        out, edges = np.histogramdd(cloud.T,
                                    bins=new_res,
                                    range=((-co_size[0] / 2, co_size[0] / 2),
                                           (-co_size[1] / 2, co_size[1] / 2),
                                           (-co_size[2] / 2, co_size[2] / 2),)
                                    )
        out = out > 0
        out = out.astype(float)

        for ex in extend:

            if ex=="-Z" or ex=="+Z":
                p = np.amax(out, axis=(0,1))
                pp = p > 0
                zix = np.arange(0, new_res[2], dtype=int)
            if ex == "-Z":
                zminus_mask = zix < zix[pp][0]
                out[:, :, zminus_mask] = np.repeat(out[:, :, zix[pp][0]][:, :, np.newaxis],
                                                   np.count_nonzero(zminus_mask),
                                                   axis=2)
            if ex == "+Z":
                zplus_mask = zix > zix[pp][-1]
                out[:, :, zplus_mask] = np.repeat(out[:, :, zix[pp][-1]][:, :, np.newaxis],
                                                  np.count_nonzero(zplus_mask),
                                                  axis=2)

            if ex=="-X" or ex=="+X":
                p = np.amax(out, axis=(1, 2))
                pp = p > 0
                xix = np.arange(0, new_res[0], dtype=int)
            if ex == "-X":
                xminus_mask = xix < xix[pp][0]
                out[xminus_mask, :, :] = np.repeat(out[xix[pp][0], :, :][np.newaxis, :, :],
                                                   np.count_nonzero(xminus_mask),
                                                   axis=0)
            if ex == "+X":
                xplus_mask = xix > xix[pp][-1]
                out[xplus_mask, :, :] = np.repeat(out[xix[pp][-1], :, :][np.newaxis, :, :],
                                                  np.count_nonzero(xplus_mask),
                                                  axis=0)

            if ex=="-Y" or ex=="+Y":
                p = np.amax(out, axis=(0, 2))
                pp = p > 0
                yix = np.arange(0, new_res[1], dtype=int)
            if ex == "-Y":
                yminus_mask = yix < yix[pp][0]
                out[:, yminus_mask, :] = np.repeat(out[:, yix[pp][0], :][:, np.newaxis, :],
                                                   np.count_nonzero(yminus_mask),
                                                   axis=1)
            if ex == "+Y":
                yplus_mask = yix > yix[pp][-1]
                out[:, yplus_mask, :] = np.repeat(out[:, yix[pp][-1], :][:, np.newaxis, :],
                                                  np.count_nonzero(yplus_mask),
                                                  axis=1)

        return out


class VectorField(ModifyVectorObject):
    """Constructs a VectorField object from a vector field function.

        Args:
            vf: The vector field function - vf(p, *vf_parameters).
            vf_parameters: The parameters of the vector field function.
    """

    def __init__(self, vf: Callable[[np.ndarray, tuple], np.ndarray], *vf_parameters: tuple):
        ModifyVectorObject.__init__(self, vf)
        self._vf_parameters = vf_parameters
        self._vf = vf

    def create(self, p: np.ndarray) -> np.ndarray:
        """Applies the modifications to the vector field and returns the map of the vector field.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
        Returns:
             Vector field of shape (D, N).
        """
        self._vf = self.vf
        return self._vf(p, *self._vf_parameters)

    def propagate(self, p: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications to the vector field and returns the map of the vector field.
        Similar to create but is meant to be used as input to construct new vector fields.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty.
        
        Returns:
            Vector field of shape (D, N).
        """
        self._vf = self.vf
        return self._vf(p, *self._vf_parameters)

    def x(self, p: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications to the vector field and returns the x component of the vector field.
        Similar to create but is meant to be used as input to construct new vector fields.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty.
        
        Returns:
            Scalar field of shape (N,).
        """
        self._vf = self.vf
        return self._vf(p, *self._vf_parameters)[0]

    def y(self, p: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications to the vector field and returns the x component of the vector field.
        Similar to create but is meant to be used as input to construct new vector fields.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty.
        
        Returns:
            Scalar field of shape (N,).
        """
        self._vf = self.vf
        return self._vf(p, *self._vf_parameters)[1]

    def z(self, p: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications to the vector field and returns the x component of the vector field.
        Similar to create but is meant to be used as input to construct new vector fields.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty.
        
        Returns:
            Scalar field of shape (N,).
        """
        self._vf = self.vf
        return self._vf(p, *self._vf_parameters)[2]

    def phi(self, p: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications and returns the azimuthal angle for each vector in the vector field.
        Similar to create but is meant to be used as input to construct new vector fields.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty.
        
        Returns:
            Scalar field of shape (N,).
        """
        self._vf = self.vf
        vec = self._vf(p, *self._vf_parameters)

        return np.arctan2(vec[1], vec[0])

    def theta(self, p: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications and returns the polar angle for each vector in the vector field.
        Similar to create but is meant to be used as input to construct new vector fields.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty.
        
        Returns:
            Scalar field of shape (N,).
        """
        self._vf = self.vf
        vec = self._vf(p, *self._vf_parameters)

        return np.arccos(vec[2])

    def length(self, p: np.ndarray, *parameters_: tuple) -> np.ndarray:
        """
        Applies the modifications and returns the lengths of vectors in the vector field.
        Similar to create but is meant to be used as input to construct new vector fields.
        
        Args:
            p: Point cloud of field components (D, N);
                D - number of dimensions (2,3);
                N - number of points in the point cloud.
            parameters_: can be empty.
        
        Returns:
            Scalar field of shape (N,).
        """
        self._vf = self.vf
        vec = self._vf(p, *self._vf_parameters)

        return np.linalg.norm(vec, axis=0)





