# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.


from spomso.cores.geom import VectorField
from spomso.cores.vector_functions_special import lcwg1_2d, lcwg1_m1, lcwg1_p1


class LCWG2D(VectorField):

    def __init__(self,  parameters, co_resolution, sign):
        """
        Vector field of a Liquid Crystal waveguide.
        The vector field is generated based on the distance field and size parameters of the waveguide.
        Point cloud specifying the values of the Signed Distance Function is taken as the input.
        The vector field is independent of the z coordinate (cartesian coordinates).
        The components of the output vector field are cartesian (x, y, z).
        :param parameters: Total width of the waveguide.
        :param co_resolution: Number of points along each axis in the grid on which the SDF is evaluated.
        :param sign: array, float, int or None determining the sign of the normal coordinates in parts of the cartesian
         coordinate system.
         None - sign is calculated automatically,
         float - the value is used a threshold for automatic calculation of the sign (same as None)
         int (+1 or -1) - the same sign everywhere
         array - array containing values of +1 or -1, specifying the sign for each point in the point cloud.
        """
        self._parameters = parameters
        VectorField.__init__(self, lcwg1_2d,  parameters, co_resolution, sign)

    @property
    def parameters(self):
        return self._parameters


class LCWG3Dm1(VectorField):

    def __init__(self,  parameters, co_resolution, sign):
        """
        M1 type vector field of a Liquid Crystal waveguide. Winding number in the yz plane is -1.
        The vector field is generated based on the distance field and size parameters of the waveguide.
        Point cloud specifying the values of the Signed Distance Function is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        :param parameters: Total width of the waveguide, thickness of the waveguide.
        :param co_resolution: Number of points along each axis in the grid on which the SDF is evaluated.
        :param sign: array, float, int or None determining the sign of the normal coordinates in parts of the cartesian
         coordinate system.
         None - sign is calculated automatically,
         float - the value is used a threshold for automatic calculation of the sign (same as None)
         int (+1 or -1) - the same sign everywhere
         array - array containing values of +1 or -1, specifying the sign for each point in the point cloud.
        """
        self._parameters = parameters
        VectorField.__init__(self, lcwg1_m1, parameters, co_resolution, sign)

    @property
    def parameters(self):
        return self._parameters


class LCWG3Dp1(VectorField):

    def __init__(self,  parameters, co_resolution, sign):
        """
        P1 type vector field of a Liquid Crystal waveguide. Winding number in the yz plane is +1.
        The vector field is generated based on the distance field and size parameters of the waveguide.
        Point cloud specifying the values of the Signed Distance Function is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        :param parameters: Total width of the waveguide, thickness of the waveguide.
        :param co_resolution: Number of points along each axis in the grid on which the SDF is evaluated.
        :param sign: array, float, int or None determining the sign of the normal coordinates in parts of the cartesian
         coordinate system.
         None - sign is calculated automatically,
         float - the value is used a threshold for automatic calculation of the sign (same as None)
         int (+1 or -1) - the same sign everywhere
         array - array containing values of +1 or -1, specifying the sign for each point in the point cloud.
        """
        self._parameters = parameters
        VectorField.__init__(self, lcwg1_p1, parameters, co_resolution, sign)

    @property
    def parameters(self):
        return self._parameters

