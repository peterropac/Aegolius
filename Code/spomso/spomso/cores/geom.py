# Copyright (C) 2023 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from spomso.cores.transformations import EuclideanTransform
from spomso.cores.modifications import ModifyObject


class GenericGeometry(EuclideanTransform, ModifyObject):

    def __init__(self, geo_sdf, *geo_parameters):
        """
        Constructs all the necessary attributes for the GenericGeometry object.
        :param geo_sdf: the SDF function of the geometry - geo_sdf(co, *geo_parameters)
        :param geo_parameters: the parameters of the SDF function
        """
        EuclideanTransform.__init__(self)
        ModifyObject.__init__(self, geo_sdf)
        self.geo_parameters = geo_parameters
        self._sdf = geo_sdf

    def create(self, co: np.ndarray) -> np.ndarray:
        """
        Applies the modifications and transformations (in that order) to the SDF and return the map of the
        Signed Distance field.
        :param co: Point cloud of coordinates (D, N);
        D - number of dimensions (2,3);
        N - number of points in the point cloud.
        :return: Signed Distance field as a numpy.ndarray of shape (N,)
        """
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, self.geo_parameters)

    def propagate(self, co: np.ndarray, *parameters_: object) -> np.ndarray:
        """
        Applies the modifications and transformations (in that order) to the SDF and return the map of the
        Signed Distance field. Similar to create but is meant to be used as input to construct new geometry.
        :param co: Point cloud of coordinates (D, N);
        D - number of dimensions (2,3);
        N - number of points in the point cloud.
        :param parameters_: can be empty
        :return: Signed Distance field as a numpy.ndarray of shape (N,)
        """
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, self.geo_parameters)