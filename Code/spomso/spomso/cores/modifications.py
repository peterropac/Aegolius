# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from typing import Callable

from spomso.cores.vector_modification_functions import add_vectors, subtract_vectors, rescale_vectors
from spomso.cores.vector_modification_functions import rotate_vectors_phi, rotate_vectors_theta
from spomso.cores.vector_modification_functions import revolve_field_x, revolve_field_y, revolve_field_z
from spomso.cores.vector_modification_functions import rotate_vectors_x_axis
from spomso.cores.vector_modification_functions import rotate_vectors_y_axis
from spomso.cores.vector_modification_functions import rotate_vectors_z_axis
from spomso.cores.vector_modification_functions import rotate_vectors_axis
from spomso.cores.vector_modification_functions import batch_normalize

from spomso.cores.post_processing import sigmoid_falloff, positive_sigmoid_falloff, capped_exponential
from spomso.cores.post_processing import linear_falloff
from spomso.cores.post_processing import relu, smooth_relu, slowstart
from spomso.cores.post_processing import hard_binarization
from spomso.cores.post_processing import gaussian_boundary, gaussian_falloff
from spomso.cores.post_processing import conv_averaging, conv_edge_detection

from spomso.cores.helper_functions import smarter_reshape, vector_smarter_reshape


class ModifyObject:
    """Class containing all the possible modifications which can be applied to a scalar field.

    Attributes:
        original_geo_object: Original SDF.
        geo_object: Modified SDF.

    Args:
        geo_object: SDF of a geometry.
    """

    def __init__(self, geo_object: Callable[[np.ndarray, tuple], np.ndarray]):
        self._mod: list = []
        self.original_geo_object: Callable[[np.ndarray, tuple], np.ndarray] = geo_object
        self.geo_object: Callable[[np.ndarray, tuple], np.ndarray] = geo_object

    @property
    def modifications(self) -> list:
        """All the modifications which were applied to the geometry in chronological order.

        Returns:
            List of modifications.
        """
        return self._mod

    @property
    def modified_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        SDF of the modified geometry.

        Returns:
            SDF of the modified geometry.
        """
        return self.geo_object

    @property
    def original_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        SDF of the unmodified geometry.

        Returns:
            SDF of the unmodified geometry.
        """
        return self.original_geo_object

    def elongation(self, elon_vector: np.ndarray | tuple | list) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Elongates the geometry along a certain vector by the length of the vector in each respective direction.

        Args:
            elon_vector: 3vector defining the direction and distance of the elongation.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("elongation")
        elon_vector = np.asarray(elon_vector)/2

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            qo0 = co[0] - np.clip(co[0], -elon_vector[0]/2, elon_vector[0]/2)
            qo1 = co[1] - np.clip(co[1], -elon_vector[1] / 2, elon_vector[1] / 2)
            qo2 = co[2] - np.clip(co[2], -elon_vector[2] / 2, elon_vector[2] / 2)
            qo = np.asarray([qo0, qo1, qo2])
            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def rounding(self, rounding_radius: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rounds off the geometry - effectively thickening it by the rounding radius.
        rounding_radius: Radius by which the edges are rounded and the object thickened.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("rounding")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return geo_object(co, *params) - rounding_radius

        self.geo_object = new_geo_object
        return new_geo_object

    def rounding_cs(self,
                    rounding_radius: float | int,
                    bb_size: int | float) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rounds off the geometry, but the geometry will be contained in its bounding box.
        Bounding box size must be specified.

        Args:
            rounding_radius: Radius by which the edges are rounded.
            bb_size: Size of the bounding box. Typically, the largest dimension of the object.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("rounding_cs")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            scale = (1 - rounding_radius/bb_size)
            return scale*geo_object(co/scale, *params) - rounding_radius

        self.geo_object = new_geo_object
        return new_geo_object

    def boundary(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Get the boundary of a shape.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("boundary")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.abs(geo_object(co, *params))

        self.geo_object = new_geo_object
        return new_geo_object

    def signed_old(self, co_resolution: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Transform an Unsigned Distance Function into a Signed Distance Function.

        Args:
            co_resolution: Resolution of the coordinate system.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("signed")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            sp = geo_object(co, *params)
            if np.amin(sp) < 0:
                return sp

            s = smarter_reshape(sp, co_resolution)
            c = vector_smarter_reshape(co, co_resolution)

            seps = tuple(np.abs(c[i, 1*(i==0), 1*(i==1), 1*(i==2)] - c[i, 0, 0, 0]) for i in range(c.shape[0]))
            boundary = s < np.min(seps)

            mark = np.zeros(s.shape)
            fmark = np.zeros(s.shape)
            for i in range(1, s.shape[0]):
                chu = boundary[i, :, :] * ~boundary[i - 1, :, :]
                j = s.shape[0] - 1 - i
                chuu = boundary[j, :, :] * ~boundary[j + 1, :, :]
                mark[i, :, :] = chu + mark[i-1, :, :]
                fmark[i, :, :] = chuu + fmark[i - 1, :, :]
            fmark = np.flip(fmark, axis=0)
            interior = np.clip((mark % 2 + fmark % 2), 0, 1)

            mark = np.zeros(s.shape)
            fmark = np.zeros(s.shape)
            for i in range(1, s.shape[1]):
                chu = boundary[:, i, :] * ~boundary[:, i - 1, :]
                j = s.shape[1]-1-i
                chuu = boundary[:, j, :] * ~boundary[:, j + 1, :]
                mark[:, i, :] = chu + mark[:, i - 1, :]
                fmark[:, i, :] = chuu + fmark[:, i - 1, :]
            fmark = np.flip(fmark, axis=1)
            interior *= np.clip((mark % 2 + fmark % 2), 0, 1)

            interior = conv_averaging(interior, (2, 2, 1), 1)
            sign = 1 - 2*(interior > 0.5)
            sign = sign.flatten()

            return sp*sign

        self.geo_object = new_geo_object
        return new_geo_object

    def signed(self, co_resolution: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Transform an Unsigned Distance Function into a Signed Distance Function.

        Args:
            co_resolution: Resolution of the coordinate system.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("signed")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            sp = geo_object(co, *params)
            if np.amin(sp) < 0:
                return sp

            s = smarter_reshape(sp, co_resolution)
            c = vector_smarter_reshape(co, co_resolution)

            seps = tuple(np.abs(c[i, 1*(i==0), 1*(i==1), 1*(i==2)] - c[i, 0, 0, 0]) for i in range(c.shape[0]))
            boundary = s < np.min(seps)

            chu = np.zeros(s.shape)
            chuu = np.zeros(s.shape)
            chu[1:] = boundary[1:, :, :] * ~boundary[:-1, :, :]
            chuu[:-1] = boundary[:-1, :, :] * ~boundary[1:, :, :]
            mark = np.cumsum(chu, axis=0)
            fmark = np.cumsum(chuu, axis=0)
            fmark = np.flip(fmark, axis=0)
            interior = np.clip((mark % 2 + fmark % 2), 0, 1)

            chu = np.zeros(s.shape)
            chuu = np.zeros(s.shape)
            chu[:, 1:, :] = boundary[:, 1:, :] * ~boundary[:, :-1, :]
            chuu[:, :-1, :] = boundary[:, :-1, :] * ~boundary[:, 1:, :]
            mark = np.cumsum(chu, axis=1)
            fmark = np.cumsum(chuu, axis=1)
            fmark = np.flip(fmark, axis=1)
            interior *= np.clip((mark % 2 + fmark % 2), 0, 1)

            interior = conv_averaging(interior, (2, 2, 1), 1)

            interior = interior[1:-1, 1:-1, 1:-1]
            interior = np.pad(interior, pad_width=1, mode="edge")

            sign = 1 - 2*(interior > 0.5)
            sign = sign.flatten()

            return sp*sign

        self.geo_object = new_geo_object
        return new_geo_object

    def invert(self, direct: bool = False) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Invert the sign of the SDF.

        Args:
            direct: If True a function which returns inverts the sign of the SDF is returned without modifying the SDF.
                If False the same happens as described above but the SDF is also modified.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("invert")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return -geo_object(co, *params)

        if not direct:
            self.geo_object = new_geo_object
        return new_geo_object

    def sign(self, direct: bool = False) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Get the sign of the SDF.

        Args:
            direct: If True a function which returns the sign of the SDF is returned without modifying the SDF.
                If False the same happens as described above but the SDF is also modified.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("sign")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.sign(geo_object(co, *params))

        if not direct:
            self.geo_object = new_geo_object

        return new_geo_object

    def recover_volume(self,
                      interior: Callable[[np.ndarray, tuple], np.ndarray]) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Recovers the interior of the SDF if a function which outputs the correct value (-1 or 1) is provided.
        This function should take the same parameters as the SDF.
        Typically, this function is the output of self.sign().

        Args:
            interior: Function defining the interior (negative values) of an SDF.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("recover_volume")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return geo_object(co, *params)*interior(co, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def define_volume(self,
                      interior: Callable[[np.ndarray, tuple], np.ndarray],
                      interior_parameters: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Defines the interior of the SDF if with a function.

        Args:
            interior: Function defining the interior (-1) and exterior (1) of an SDF.
            interior_parameters: Parameters of the function defining the interior of the SDF.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("define_volume")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return geo_object(co, *params)*interior(co, *interior_parameters)

        self.geo_object = new_geo_object
        return new_geo_object

    def onion(self, thickness: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Transforms the geometry into a surface with some thickness.
        Args:
            thickness: Thickness of the resulting shape.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("onion")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.abs(geo_object(co, *params)) - thickness

        self.geo_object = new_geo_object
        return new_geo_object

    def concentric(self, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Transforms an isosurface into two concentric isosurfaces which are apart by the value of width.
        Transforms a volume into an isosurface and rounds it by width/2.

        Args:
            width: Thickness of the resulting shape.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("concentric")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return np.abs(geo_object(co, *params) - width/2)

        self.geo_object = new_geo_object
        return new_geo_object

    def revolution(self, radius: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Revolves a 2D shape around the y-axis to generate a 3D shape.
        First the 2D shape is translated along the x-axis by the radius of revolution,
        then it is revolved around the y-axis.

        Args:
            radius: Radius of revolution.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("revolution")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            mag_xz = np.linalg.norm([co[0,:], co[2,:]], axis=0)
            qo = np.zeros(co.shape)
            qo[0] = mag_xz - radius
            qo[1] = co[1]

            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def axis_revolution(self, radius: float | int, angle: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Revolves a 2D shape around an axis to generate a 3D shape.
        First the 2D shape is translated along the x-axis by the radius of revolution,
        then it is revolved around the axis of revolution.
        The axis of revolution is angled by the specified angle with respect to the y-axis.

        Args:
            radius: Radius of revolution.
            angle: Angle between the axis of revolution and the y-axis.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("revolution")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            rot = np.asarray([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
            co[:2, :] = rot.dot(co[:2])

            mag_xz = np.linalg.norm([co[0, :], co[2, :]], axis=0)

            qo = np.zeros(co.shape)
            qo[0] = mag_xz
            qo[1] = co[1]
            qo[:2] = rot.T.dot(qo[:2])
            qo[0] -= radius

            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def extrusion(self, distance: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Extrudes a 2D shape along the z-axis so that the height of the object is equal to the specified distance.

        Args:
            distance: Final height of the object - distance of extrusion.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("extrusion")

        geo_object = self.geo_object

        def new_geo_object(co, *params):
            d = geo_object(co[:2], *params)
            w = np.asarray((d, np.abs(co[2]) - distance/2))

            first_term = np.minimum(np.maximum(w[0], w[1]), 0)
            second_term = np.linalg.norm(np.maximum(w, 0), axis=0)

            return first_term + second_term

        self.geo_object = new_geo_object
        return new_geo_object

    def twist(self, pitch: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Twists the geometry around the z-axis based on the position along the z-axis and the pitch.

        Args:
            pitch: rad/unit length.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("twist")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = np.cos(pitch * co[2])
            s = np.sin(pitch * co[2])
            rot = np.asarray([[c,s],[-s,c]])
            qo = co.copy()
            qo[:2,:] = rot[0, :, :] * co[0, :] + rot[1, :, :] * co[1, :]

            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def bend2(self, radius, angle):
        # NOT WORKING!!!!!!
        self._mod.append("bend")

        geo_object = self.geo_object

        def new_geo_object(co, *params):


            f = np.clip(co[0]/radius, -angle/2, angle/2)
            c = np.cos(f)
            s = np.sin(f)

            qo = co.copy()
            qo[0] = s*radius - co[0] -s*co[1]
            qo[1] = radius*(1-c) + co[1]*(c-1)
            qo[2] = 0

            bo = co.copy()
            absco = np.abs(co[0])
            sco = np.sign(co[0])
            bo[2] = 0
            bo[0] = sco*(absco - radius*angle/2)*np.cos(angle/2)
            bo[1] = sco*bo[0] * np.tan(angle / 2)

            wo = qo + co + bo*(absco>radius*angle/2)

            # qo[:2, :] = rot[0, :, :] * co[0, :] + rot[1, :, :] * co[1, :]

            return geo_object(co, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def bend(self, radius: float | int, angle: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Bends the geometry around the z-axis, based on the specified bending radius and angle.
        The length of the bent section is radius*angle.

        Args:
            radius: Bending radius.
            angle: Bending angle.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("bend")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = np.cos(angle/2)
            s = np.sin(angle/2)
            rot = np.asarray([[c,s],[-s,c]])

            qo = co.copy()
            qo[1] -= radius
            phi = np.arctan2(qo[0], -qo[1])
            qo[1] = -radius + np.linalg.norm(qo[:2], axis=0)
            qo[0] = radius * phi

            mask1 = radius * angle / 2 <= np.abs(qo[0])
            mask2 = co[0, mask1] >= 0

            wo = co[:2].copy()
            wo = wo[:, mask1]

            sign = np.sign(co[0, mask1])

            wo[0] -= radius*s*sign
            wo[1] -= radius*(1-c)

            wo[:2, mask2] = rot.dot(wo[:2, mask2])
            wo[:2, ~mask2] = rot.T.dot(wo[:2, ~mask2])

            wo[0] += radius * (angle/2) * sign

            qo[:2, mask1] = wo
            return geo_object(qo, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def displacement(self,
                     displacement_function: Callable[[np.ndarray, tuple], np.ndarray],
                     displacement_function_parameters: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Displaces the surface of an object based on the values of the displacement function. Should be applied last.

        Args:
            displacement_function:
                Function which takes an array of coordinates with shape (D, N) and a tuple of parameters.
                D - number of dimensions (2 or 3);
                N - number of points in the point cloud.
            displacement_function_parameters: Parameters of the displacement function.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("displacement")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            return geo_object(co, *params) + displacement_function(co, *displacement_function_parameters)

        self.geo_object = new_geo_object
        return new_geo_object

    def infinite_repetition(self, distances: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Infinitely repeats geometry in space on a cubic lattice.
        
        Args:
            distances: 3vector determining the distances between instances along each axis.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("infinite_repetition")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            redistances = np.asarray(distances)/2
            qo = np.mod(np.add(co.T, redistances), distances) - redistances

            return geo_object(qo.T, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def finite_repetition(self,
                          size: tuple | list | np.ndarray,
                          repetitions: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Repeats the geometry finite amount of times along each axis within a defined bounding box.
        
        Args:
            size: Size of the bounding box along each axis in which the geometry is repeated.
            repetitions: Number of repetitions along each axis inside the bounding box.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("finite_repetition")

        size = np.asarray(size)
        rep = np.asarray(repetitions)

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = size*(1 - 1/rep)/2
            d = size*(1/2 - 1/rep)
            s = size/rep
            mx = (co[0, :] >= -d[0])*(co[0, :] <= d[0])
            my = (co[1, :] >= -d[1])*(co[1, :] <= d[1])
            mz = (co[2, :] >= -d[2])*(co[2, :] <= d[2])

            # outer
            v = np.abs(co)
            v = np.subtract(v.T, c)
            v[:, 0] -= 2 * v[:, 0] * (co[0, :] < 0)
            v[:, 1] -= 2 * v[:, 1] * (co[1, :] < 0)
            v[:, 2] -= 2 * v[:, 2] * (co[2, :] < 0)

            # inner
            u = np.subtract(co.T, d)
            u[:] = np.mod(u[:], s) - s/2
            v[mx, 0] = u[mx,0]
            v[my, 1] = u[my, 1]
            v[mz, 2] = u[mz, 2]

            return geo_object(v.T, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def finite_repetition_rescaled(self,
                                   size: tuple | list | np.ndarray,
                                   repetitions: tuple | list | np.ndarray,
                                   instance_size: tuple | list | np.ndarray,
                                   padding: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Repeats the geometry finite amount of times along each axis within a defined bounding box.
        The geometry is rescaled based on the provided geometry bounding box size and padding along each axis.
        Args:
            size: Size of the bounding box along each axis in which the geometry is repeated.
            repetitions: Number of repetitions along each axis inside the bounding box.
            instance_size: Size of the bounding box around one instance of geometry along each axis.
            padding: Padding around each instance along each axis.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("finite_repetition_rescaled")

        size = np.asarray(size)
        padding = np.asarray(padding)
        rep = np.asarray(repetitions)
        f = np.asarray(instance_size)

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            c = size*(1 - 1/rep)/2
            d = size*(1/2 - 1/rep)
            s = size/rep
            mx = (co[0, :] >= -d[0])*(co[0, :] <= d[0])
            my = (co[1, :] >= -d[1])*(co[1, :] <= d[1])
            mz = (co[2, :] >= -d[2])*(co[2, :] <= d[2])

            # outer
            v = np.abs(co)
            v = np.subtract(v.T, c)
            v[:, 0] -= 2 * v[:, 0] * (co[0, :] < 0)
            v[:, 1] -= 2 * v[:, 1] * (co[1, :] < 0)
            v[:, 2] -= 2 * v[:, 2] * (co[2, :] < 0)

            # inner
            u = np.subtract(co.T, d)
            u[:] = np.mod(u[:], s) - s/2
            v[mx, 0] = u[mx,0]
            v[my, 1] = u[my, 1]
            v[mz, 2] = u[mz, 2]

            sss = np.min(s/(f + padding))
            w = v/sss

            return sss*geo_object(w.T, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def symmetry(self, axis: int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies symmetry along an axis of the object.

        Args:
            axis: Index of the axis along which the symmetry is applied.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("symmetry")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            if axis > co.shape[0]:
                return geo_object(co, *params)

            co[axis, :] = np.abs(co[axis, :])
            return geo_object(co, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def mirror(self,
               a: tuple | list | np.ndarray,
               b: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies mirroring along an axis connecting the two specified points.

        Args:
            a: Position of the object's mirror image.
            b: Position of the object.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("mirror")

        a = np.asarray(a)
        b = np.asarray(b)
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            w = b - a
            c = (b + a) / 2
            l = np.linalg.norm(w)

            x = w / l
            y = np.asarray([-x[1], x[0], 0])
            y = y / np.linalg.norm(y)
            z = np.cross(x, y)
            rot = np.asarray([x, y, z])
            co = np.subtract(co.T, c).T
            co = rot.dot(co)

            # outer
            v = np.zeros(co.shape)
            v[1:, :] = co[1:]
            v[0, :] = np.abs(co[0, :]) - l / 2

            return geo_object(v, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def rotational_symmetry(self,
                            n: int,
                            radius: float | int,
                            phase: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies n-fold rotational symmetry to the object.

        Args:
            n: Order of the symmetry.
            radius: By how much the geometry is moved from the centre.
            phase: Angle by which the circular pattern is rotated by.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("rotational_symmetry")

        angle = 2*np.pi/n
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            rot = np.asarray([[np.cos(angle/2 - phase), np.sin(angle/2 - phase)],
                              [-np.sin(angle/2 - phase), np.cos(angle/2 - phase)]])
            co[:2] = rot.dot(co[:2])
            phi = np.arctan2(co[1], co[0])
            phi[phi < 0] = 2 * np.pi + phi[phi < 0]
            phi = np.mod(phi, angle) - angle/2
            radii = np.linalg.norm(co[:2,:], axis=0)
            co[0] = radii*np.cos(phi) - radius
            co[1] = radii*np.sin(phi)

            return geo_object(co, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def linear_instancing(self,
                          n: int,
                          a: tuple | list | np.ndarray,
                          b: tuple | list | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a segment defined by two points.

        Args:
            n: Number of instances along the segment.
            a: Starting point of the segment.
            b: Ending point of the segment.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("linear_instancing")

        a = np.asarray(a)
        b = np.asarray(b)
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            w = b-a
            c = (b + a) / 2
            l = np.linalg.norm(w)

            x = w/l
            y = np.asarray([-x[1], x[0], 0])
            y = y/np.linalg.norm(y)
            z = np.cross(x,y)
            rot = np.asarray([x,y,z])
            co = np.subtract(co.T, c).T
            co = rot.dot(co)

            s = l/(n-1)
            d = s/2
            mx = (co[0, :] >= -l/2 + d) * (co[0, :] <= l/2 - d)

            # outer
            v = np.zeros(co.shape)
            v[1:, :] = co[1:]
            v[0, :] = np.abs(co[0,:]) - l/2
            v[0, :] -= 2*v[0, :]*(co[0, :] < 0)

            # inner
            if n > 2:
                u = np.subtract(co.T, [l/2 - d, 0, 0]).T
                u[:] = np.mod(u[:], s) - d
                v[0, mx] = u[0, mx]

            return geo_object(v, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def curve_instancing(self,
                         f: Callable[[np.ndarray, tuple], np.ndarray],
                         f_parameters: tuple,
                         t_range: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a parametric curve defined by a function.

        Args:
            f: Function defining the parametric curve.
            f_parameters: Parameters of the function defining the parametric curve.
            t_range: (t_start, t_end, n_instances):
                t_start - initial value of the parameter t, which is feed into the parametric curve.
                t_end - final value of the parameter t.
                n_instances -  number of instances of the geometry along the parametric curve.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("parametric_curve_instancing")

        geo_object = self.geo_object

        def new_geo_object(co, *params):

            ts = np.linspace(*t_range)
            fval = f(ts, *f_parameters)
            va = np.zeros((3, t_range[-1]))
            va[:fval.shape[0]] = fval
            ix = np.arange(0, t_range[-1])

            interp = NearestNDInterpolator(va.T, ix)
            values = interp(co[0], co[1], co[2])

            w = np.zeros(co.shape)
            for i in range(t_range[-1]):
                mask = values == i
                v = np.subtract(co[:, mask].T, va[:, i]).T
                w[:, mask] = v[:, :]

            return geo_object(w, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def aligned_curve_instancing(self,
                         f: Callable[[np.ndarray, tuple], np.ndarray],
                         f_parameters: tuple,
                         t_range: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a parametric curve defined by a function.
        The x-axis of the geometry is aligned with the tangent vector of the parametric curve.

        Args:
            f: Function defining the parametric curve.
            f_parameters: Parameters of the function defining the parametric curve.
            t_range: (t_start, t_end, n_instances):
                t_start - initial value of the parameter t, which is feed into the parametric curve.
                t_end - final value of the parameter t.
                n_instances -  number of instances of the geometry along the parametric curve.
        
        Returns: 
            Modified SDF.
        """
        # approximation with the binormal (dy) and normal (dz) vector!!!
        self._mod.append("aligned_parametric_curve_instancing")

        geo_object = self.geo_object
        tol = 0.001

        def new_geo_object(co, *params):

            ts = np.linspace(*t_range)
            ts_min = ts-tol
            ts_max = ts+tol

            fval = f(ts, *f_parameters)
            va = np.zeros((3, t_range[-1]))
            va[:fval.shape[0]] = fval
            ix = np.arange(0, t_range[-1])

            fv_min = f(ts_min, *f_parameters)
            fv_max = f(ts_max, *f_parameters)
            der = (fv_max-fv_min)/(2*tol)
            der = der/np.linalg.norm(der, axis=0)

            dx = np.zeros((3, t_range[-1]))
            dx[:fval.shape[0]] = der
            dy = np.zeros((3, t_range[-1]))
            dy[0, :] = -dx[1]
            dy[1, :] = dx[0]
            dz = np.cross(dx.T, dy.T).T
            trot = np.asarray([dx, dy, dz])

            interp = NearestNDInterpolator(va.T, ix)
            values = interp(co[0], co[1], co[2])

            w = np.zeros(co.shape)
            for i in range(t_range[-1]):
                mask = values == i
                v = np.subtract(co[:, mask].T, va[:, i]).T
                v = trot[:, :, i].dot(v)
                w[:, mask] = v[:, :]

            return geo_object(w, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def fully_aligned_curve_instancing(self,
                         f: Callable[[np.ndarray, tuple], np.ndarray],
                         f_parameters: tuple,
                         t_range: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Instances the geometry n-times along the length of a parametric curve defined by a function.
        The x-axis of the geometry is aligned with the tangent vector of the parametric curve.
        The y-axis of the geometry is aligned with the normal vector of the parametric curve.
        The z-axis of the geometry is aligned with the binormal vector of the parametric curve.

        Args:
            f: Function defining the parametric curve.
            f_parameters: Parameters of the function defining the parametric curve.
            t_range: (t_start, t_end, n_instances):
                t_start - initial value of the parameter t, which is feed into the parametric curve.
                t_end - final value of the parameter t.
                n_instances -  number of instances of the geometry along the parametric curve.
        
        Returns: 
            Modified SDF.
        """
        self._mod.append("fully_aligned_parametric_curve_instancing")

        geo_object = self.geo_object
        tol = 0.001

        def new_geo_object(co, *params):

            ts = np.linspace(*t_range)
            ts_min = ts-tol
            ts_max = ts+tol

            fval = f(ts, *f_parameters)
            va = np.zeros((3, t_range[-1]))
            va[:fval.shape[0]] = fval
            ix = np.arange(0, t_range[-1])

            fv_min = f(ts_min, *f_parameters)
            fv_max = f(ts_max, *f_parameters)
            der = (fv_max-fv_min)/(2*tol)
            dermag = np.linalg.norm(der, axis=0)
            der = der/dermag

            der2 = (fv_max - 2*fval + fv_min)/(tol**2)
            der2 = der2/dermag
            der2mag = np.linalg.norm(der2, axis=0)
            der2 = der2/der2mag

            dx = np.zeros((3, t_range[-1]))
            dx[:fval.shape[0]] = der
            dy = np.zeros((3, t_range[-1]))
            dy[:fval.shape[0]] = der2
            dz = np.cross(dx.T, dy.T).T
            trot = np.asarray([dx, dy, dz])

            interp = NearestNDInterpolator(va.T, ix)
            values = interp(co[0], co[1], co[2])

            w = np.zeros(co.shape)
            for i in range(t_range[-1]):
                mask = values == i
                v = np.subtract(co[:, mask].T, va[:, i]).T
                v = trot[:, :, i].dot(v)
                w[:, mask] = v[:, :]

            return geo_object(w, *params)

        self.geo_object = new_geo_object
        return new_geo_object

    def custom_modification(self,
                            modification: Callable[[Callable[[np.ndarray, tuple], np.ndarray],
                                                    np.ndarray,
                                                    tuple,
                                                    tuple],
                            np.ndarray],
                            modification_parameters: tuple,
                            modification_name: str = "custom") -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies a custom user-specified modification to a scalar (Signed Distance Function) field.

        Args:
            modification: A custom modification which takes the SDF as a first argument,
                point cloud of coordinates at which the SDF is evaluated as the second argument,
                parameters of the SDF as the third argument,
                and the custom modification parameters as the fourth argument.
            modification_parameters: Parameters of the custom modification.
            modification_name: Name of the custom modification.

        Returns:
            Modified Scalar Field.
        """
        self._mod.append(modification_name)
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            return modification(geo_object, co, params, modification_parameters)

        self.geo_object = new_geo_object
        return new_geo_object

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
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return sigmoid_falloff(u, amplitude, width)

        self.geo_object = new_geo_object
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
        self._mod.append("positive_sigmoid_falloff")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return positive_sigmoid_falloff(u, amplitude, width)

        self.geo_object = new_geo_object
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
        self._mod.append("capped_exponential")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return capped_exponential(u, amplitude, width)

        self.geo_object = new_geo_object
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
        self._mod.append("hard_binarization")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return hard_binarization(u, threshold)

        self.geo_object = new_geo_object
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
        self._mod.append("linear_falloff")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return linear_falloff(u, amplitude, width)

        self.geo_object = new_geo_object
        return new_geo_object

    def relu(self, width: float | int) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Applies the ReLU function to the scalar (Signed Distance Function) field.

        Args:
            width: Range at which the value of the transformed field reaches one.

        Returns:
            Modified Scalar Field.
        """
        self._mod.append("relu")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return relu(u, width)

        self.geo_object = new_geo_object
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
        self._mod.append("smooth_relu")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return smooth_relu(u, smooth_width, width, threshold)

        self.geo_object = new_geo_object
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

        self._mod.append("slowstart")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return slowstart(u, smooth_width, width, threshold, ground=ground)

        self.geo_object = new_geo_object
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
        self._mod.append("gaussian_boundary")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return gaussian_boundary(u, amplitude, width)

        self.geo_object = new_geo_object
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
        self._mod.append("gaussian_falloff")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return gaussian_falloff(u, amplitude, width)

        self.geo_object = new_geo_object
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
        self._mod.append("conv_averaging")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            u = smarter_reshape(u, co_resolution)
            out = conv_averaging(u, kernel_size, iterations)
            return out.flatten()

        self.geo_object = new_geo_object
        return new_geo_object

    def conv_edge_detection(self,
                            co_resolution: tuple) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Edge detection based ona 3x3 convolutional kernel.

        Args:
            co_resolution: Resolution of the coordinate system.

        Returns:
            Modified Scalar Field.
        """
        self._mod.append("conv_edge_detection")
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            u = smarter_reshape(u, co_resolution)
            return conv_edge_detection(u)

        self.geo_object = new_geo_object
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
        self._mod.append(post_process_name)
        geo_object = self.geo_object

        def new_geo_object(co, *params):
            u = geo_object(co, *params)
            return function(u, *parameters)

        self.geo_object = new_geo_object
        return new_geo_object


class ModifyVectorObject:
    """Class containing all the possible modifications which can be applied to a vector field.

        Attributes:
            original_vf: Original vector field.
            vf: Modified vector field.

        Args:
            vf: Vector field.
        """

    def __init__(self, vf: Callable[[np.ndarray, tuple], np.ndarray]):
        self._mod: list = []
        self.original_vf: Callable[[np.ndarray, tuple], np.ndarray] = vf
        self.vf: Callable[[np.ndarray, tuple], np.ndarray] = vf

    @property
    def modifications(self) -> list:
        """
        All the modifications which were applied to the vector field in chronological order.

        Returns:
            List of modifications.
        """
        return self._mod

    @property
    def modified_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Modified vector field.

        Returns:
            Modified vector field.
        """
        return self.vf

    @property
    def original_object(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Unmodified vector field.

        Returns:
            Unmodified vector field.
        """
        return self.original_vf

    def add(self, second_field: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Adds a number, vector, or a vector field of the same shape to the vector field.

        Args:
            second_field: a number, vector, or a vector field to be added.

        Returns:
            Modified vector field.
        """
        self._mod.append("add")

        vf = self.vf

        def new_vf(p, *params):
            return add_vectors(vf(p, *params), second_field)

        self.vf = new_vf
        return new_vf

    def subtract(self, second_field: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Subtracts a number, vector, or a vector field of the same shape to the vector field.

        Args:
            second_field: a number, vector, or a vector field to be subtracted.

        Returns:
            Modified vector field.
        """
        self._mod.append("subtract")

        vf = self.vf

        def new_vf(p, *params):
            return subtract_vectors(vf(p, *params), second_field)

        self.vf = new_vf
        return new_vf

    def rescale(self, second_field: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rescales the lengths of vectors in the vector field by a number, vector, or a vector field of the same shape.

        Args:
            second_field: a number, vector, or a vector field used for rescaling.

        Returns:
            Modified vector field.
        """
        self._mod.append("rescale")

        vf = self.vf

        def new_vf(p, *params):
            return rescale_vectors(vf(p, *params), second_field)

        self.vf = new_vf
        return new_vf

    def rotate_phi(self, phi: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rotates vectors in a 3D vector field by some azimuthal angle.

        Args:
            phi: Spatially dependent or independent angle by which to rotate the vectors in the xy-plane.
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("rotate_phi")

        vf = self.vf

        def new_vf(p, *params):
            return rotate_vectors_phi(vf(p, *params), phi)

        self.vf = new_vf
        return new_vf

    def rotate_theta(self, theta: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rotates vectors in a 3D vector field by some polar angle.

        Args:
            theta: Spatially dependent or independent angle by which to rotate the vectors in the polar direction.
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("rotate_theta")

        vf = self.vf

        def new_vf(p, *params):
            return rotate_vectors_theta(vf(p, *params), theta)

        self.vf = new_vf
        return new_vf

    def rotate_x(self, alpha: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rotates vectors in a 3D vector field by some angle.

        Args:
            alpha: Spatially dependent or independent angle by which to rotate the vectors around the x axis.
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("rotate_x")

        vf = self.vf

        def new_vf(p, *params):
            return rotate_vectors_x_axis(vf(p, *params), alpha)

        self.vf = new_vf
        return new_vf

    def rotate_y(self, alpha: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rotates vectors in a 3D vector field by some angle.

        Args:
            alpha: Spatially dependent or independent angle by which to rotate the vectors around the y axis.
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("rotate_y")

        vf = self.vf

        def new_vf(p, *params):
            return rotate_vectors_y_axis(vf(p, *params), alpha)

        self.vf = new_vf
        return new_vf

    def rotate_z(self, alpha: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rotates vectors in a 3D vector field by some angle.

        Args:
            alpha: Spatially dependent or independent angle by which to rotate the vectors around the z axis.
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("rotate_z")

        vf = self.vf

        def new_vf(p, *params):
            return rotate_vectors_z_axis(vf(p, *params), alpha)

        self.vf = new_vf
        return new_vf

    def rotate_axis(self,
                    axis: tuple | list | np.ndarray,
                    alpha: int | float | np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Rotates vectors in a 3D vector field by some angle around some axis.

        Args:
            axis: A single or an array exes around which the vectors in the vector field will be rotated.
            alpha: Spatially dependent or independent angle by which to rotate the vectors around the specified axis.
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("rotate_axis")

        vf = self.vf

        def new_vf(p, *params):
            return rotate_vectors_axis(vf(p, *params), axis, alpha)

        self.vf = new_vf
        return new_vf

    def revolution_x(self, co: np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Revolves a 2D vector field around the x-axis to generate a 3D vector field.

        Args:
            co: Coordinate system with shape (3, N).
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("revolution_x")

        vf = self.vf

        def new_vf(p, *params):

            return revolve_field_x(co, vf(p, *params))

        self.vf = new_vf
        return new_vf

    def revolution_y(self, co: np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Revolves a 2D vector field around the y-axis to generate a 3D vector field.

        Args:
            co: Coordinate system with shape (3, N).
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("revolution_y")

        vf = self.vf

        def new_vf(p, *params):
            return revolve_field_y(co, vf(p, *params))

        self.vf = new_vf
        return new_vf

    def revolution_z(self, co: np.ndarray) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Revolves a 2D vector field around the z-axis to generate a 3D vector field.

        Args:
            co: Coordinate system with shape (3, N).
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("revolution_z")

        vf = self.vf

        def new_vf(p, *params):
            return revolve_field_z(co, vf(p, *params))

        self.vf = new_vf
        return new_vf

    def normalize(self) -> Callable[[np.ndarray, tuple], np.ndarray]:
        """
        Normalizes the vector field.
                                  
        Returns:
            Modified vector field.
        """
        self._mod.append("normalize")

        vf = self.vf

        def new_vf(p, *params):
            return batch_normalize(vf(p, *params))

        self.vf = new_vf
        return new_vf
