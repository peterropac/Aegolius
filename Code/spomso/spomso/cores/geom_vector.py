# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from spomso.cores.geom import VectorField
from spomso.cores.vector_functions import cartesian_define, cylindrical_define, spherical_define
from spomso.cores.vector_functions import radial_vector_field_cylindrical, radial_vector_field_spherical
from spomso.cores.vector_functions import hyperbolic_vector_field_cylindrical, awn_vector_field_cylindrical
from spomso.cores.vector_functions import aar_vector_field_cylindrical
from spomso.cores.vector_functions import vortex_vector_field_cylindrical, aav_vector_field_cylindrical
from spomso.cores.vector_functions import x_vector_field, y_vector_field, z_vector_field
from spomso.cores.vector_functions import from_sdf


class CartesianVectorField(VectorField):

    def __init__(self):
        """
        Vector field defined by its components in the cartesian coordinate system (x, y, z).
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, cartesian_define)


class CylindricalVectorField(VectorField):

    def __init__(self):
        """
        Vector field defined by its components in the cylindrical coordinate system (r, phi, z).
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, cylindrical_define)


class SphericalVectorField(VectorField):

    def __init__(self):
        """
        Vector field defined by its components in the spherical coordinate system (r, phi, theta).
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, spherical_define)


class RadialSphericalVectorField(VectorField):

    def __init__(self):
        """
        Vector field where all the vectors are pointing radially outwards from the origin.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, radial_vector_field_spherical)


class RadialCylindricalVectorField(VectorField):

    def __init__(self):
        """
        Vector field where all the vectors are pointing radially outwards from the line x=0, y=0.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, radial_vector_field_cylindrical)


class HyperbolicCylindricalVectorField(VectorField):

    def __init__(self):
        """
        Hyperbolic vector field centered at the line x=0, y=0.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, hyperbolic_vector_field_cylindrical)


class WindingCylindricalVectorField(VectorField):

    def __init__(self, gamma: float | int | np.ndarray):
        """
        Vector field where the vectors rotate around the line x=0, y=0, based on the winding number.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        :param gamma: winding number.
        """
        self._gamma = gamma
        VectorField.__init__(self, awn_vector_field_cylindrical, (gamma, ))

    @property
    def gamma(self):
        return self._gamma


class AngledRadialCylindricalVectorField(VectorField):

    def __init__(self, alpha: float | int | np.ndarray):
        """
        Modified radial vector field centered at the line x=0, y=0.
        The vectors are at the specified angle with respect to the lines pointing outward from the line x=0, y=0
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        :param alpha: Angle between the vectors and the equidistant surfaces from the line x=0, y=0.
        """
        self._alpha = alpha
        VectorField.__init__(self, aar_vector_field_cylindrical, (alpha, ))

    @property
    def alpha(self):
        return self._alpha


class VortexCylindricalVectorField(VectorField):

    def __init__(self):
        """
        Vortex vector field centered at the line x=0, y=0.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, vortex_vector_field_cylindrical)


class AngledVortexCylindricalVectorField(VectorField):

    def __init__(self, alpha: float | int | np.ndarray):
        """
        Modified vortex vector field centered at the line x=0, y=0.
        The vectors are at the specified angle with respect to the equidistant surfaces from the line x=0, y=0.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        :param alpha: Angle between the vectors and the equidistant surfaces from the line x=0, y=0.
        """
        self._alpha = alpha
        VectorField.__init__(self, aav_vector_field_cylindrical, (alpha, ))

    @property
    def alpha(self):
        return self._alpha


class XVectorField(VectorField):

    def __init__(self):
        """
        Vector field where only the X component (cartesian coordinates) is non-zero.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, x_vector_field)


class YVectorField(VectorField):

    def __init__(self):
        """
        Vector field where only the Y component (cartesian coordinates) is non-zero.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, y_vector_field)


class ZVectorField(VectorField):

    def __init__(self):
        """
        Vector field where only the Z component (cartesian coordinates) is non-zero.
        Point cloud specifying the positions of points at which the vector field is evaluated is taken as the input.
        The components of the output vector field are cartesian (x, y, z).
        """
        VectorField.__init__(self, z_vector_field)


class VectorFieldFromSDF(VectorField):

    def __init__(self, grid_resolution):
        """
        Vector field constructed from an SDF.
        Point cloud specifying the value of the SDF is taken as an input.
        The components of the output vector field are cartesian (x, y, z).
        :param grid_resolution: Number of points along each axis in the grid on which the SDF is evaluated.
        """
        VectorField.__init__(self, from_sdf,  grid_resolution)





