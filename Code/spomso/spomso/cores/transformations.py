# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.spatial.transform import Rotation as srot
from typing import Callable


class EuclideanTransform:
    """
    Class containing the Euclidian transforms which can be applied to a geometry.
    """

    def __init__(self):
        self._et: list = []
        self._center: np.ndarray | tuple | list = np.asarray((0.0,0.0,0.0))
        self._scale: float | int = 1.0
        self._rot_matrix: np.ndarray = np.eye(3)
        self._angle: float | int = 0.0
        self._axis: np.ndarray | tuple | list = np.asarray((0.0, 0.0, 1.0))

    @property
    def transformations(self) -> list:
        """All the Euclidian transformations which were applied to the geometry in chronological order.

        Returns:
            List of Euclidian transformations.
        """
        return self._et

    @property
    def center(self) -> np.ndarray | tuple | list:
        """Center of mass of the geometry.

        Returns:
            Position vector of the center of mass of the geometry.
        """
        return self._center

    @property
    def scale(self) -> float | list:
        """Scale factor of the geometry.

        Returns:
            Scale factor.
        """
        return self._scale

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix applied to the geometry.

        Returns:
            Rotation matrix (3, 3).
        """
        return self._rot_matrix

    @property
    def rotation_axis(self) -> np.ndarray | tuple | list:
        """Vector representing the axis of rotation.

        Returns:
            Axis of rotation.
        """
        return self._axis

    @property
    def rotation_angle(self) -> float | int:
        """Angle by which the geometry is rotated around the axis of rotation.

        Returns:
           Angle of rotation.
        """
        return self._angle

    def set_location(self, center: np.ndarray | tuple | list):
        """Set the position of the geometry in 3D space.
        
        Args:
            center: 3vector which defines the new position of the geometry.
        """
        self._et.append("set_location")

        center = np.asarray(center)
        if center.size <= 3:
            self._center[:center.size] = center
        else:
            raise SyntaxError(f"Array {center} is of incorrect size!")

    def move(self, move_vector: np.ndarray | tuple | list):
        """
        Move the geometry by a vector.
        
        Args:
            move_vector: 3vector by which the geometry is moved.
        """
        self._et.append("move")

        vector = np.asarray(move_vector)
        if vector.size <= 3:
            self._center += vector
        else:
            raise SyntaxError(f"Array {vector} is of incorrect size!")

    def set_scale(self, scale: float | int):
        """
        Sets the scaling of the geometry.
        
        Args:
            scale: Scaling factor.
        """
        self._et.append("set_scale")

        if any(isinstance(scale, type_) for type_ in [float, int]):
            self._scale = scale
        else:
            raise TypeError("Scale must be a float or an int")

    def rescale(self, scale: float | int):
        """
        Multiplies the existing scaling of the geometry by the scale factor.
        
        Args:
            scale: Scale factor.
        """
        self._et.append("rescale")

        if any(isinstance(scale, type_) for type_ in [float, int]):
            self._scale *= scale
        else:
            raise TypeError("Scale must be a float or an int")

    @staticmethod
    def get_rotation_matrix(angle: float | int, axis: np.ndarray | tuple | list) -> tuple:
        """
        Get the rotation matrix from the rotation angle and axis of rotation.
        
        Args:
            angle: Angle of rotation.
            axis: Axis of rotation.
        
        Returns:
            Rotation matrix (3, 3).
        """
        axis = np.asarray(axis)
        if axis.size <= 3:
            axis_ = np.zeros(3)
            axis_[:axis.size] = axis
        else:
            raise SyntaxError(f"Array {axis} is of incorrect size!")

        if any(isinstance(angle, type_) for type_ in [float, int]):
            angle_ = angle
        else:
            raise TypeError("Rotation angle must be a float or an int")

        r = srot.from_rotvec(angle_ * axis_)
        return r.as_matrix(), angle_, axis_

    def set_rotation(self, angle: float | int, axis: np.ndarray | tuple | list):
        """
        Sets the rotation matrix from the rotation angle and axis.
        
        Args:
            angle: Angle of rotation.
            axis: Axis of rotation.
        """
        self._et.append("set_rotation")

        self._rot_matrix, self._angle, self._axis = self.get_rotation_matrix(angle, axis)

    def rotate_rotvec(self, angle: float | int, axis: np.ndarray | tuple | list):
        """
        Multiplies the previous existing rotation matrix by a rotation matrix calculated
        from the angle and axis of rotation.
        
        Args:
            angle: Angle of rotation.
            axis: Axis of rotation.
        """
        de = np.array_equal(axis, np.zeros(3)[:axis.size])
        if de:
            raise ValueError("Axis cannot be zero!")

        axis = axis/np.linalg.norm(axis)
        rot_matrix, angle_, axis_ = self.get_rotation_matrix(angle, axis)

        self.rotate_matrix(rot_matrix)

    def rotate_matrix(self, rotation_matrix: np.ndarray | tuple | list):
        """
        Multiplies the previous existing rotation matrix by a given rotation matrix.
        
        Args:
            rotation_matrix: Rotation matrix (3, 3).
        """
        self._rot_matrix = np.matmul(rotation_matrix, self._rot_matrix)

        r = srot.from_matrix(self._rot_matrix)
        rot_vec_ = r.as_rotvec()
        self._angle = np.linalg.norm(rot_vec_)
        if not self._angle == 0:
            self._axis = rot_vec_ / self._angle
        else:
            self._axis = np.asarray((0.0, 0.0, 1.0))

    def rotate(self, *inputs: np.ndarray | tuple):
        """
        Rotates the geometry by a rotation matrix or a rotation vector.
        
        Args:
            inputs: Rotation metrix or angle and axis vector.
        """
        self._et.append("rotate")

        n_inputs = len(inputs)

        if n_inputs==1:
            rot_matrix = np.asarray(inputs)
            self.rotate_matrix(rot_matrix)

        if n_inputs==2:
            angle, axis = inputs
            self.rotate_rotvec(angle, np.asarray(axis))

        if n_inputs>2:
            raise SyntaxError("Wrong number of inputs!")

    @staticmethod
    def apply_ec_transforms(function_: Callable[[np.ndarray, tuple], np.ndarray],
                            co_: np.ndarray,
                            params_: tuple,
                            rm: np.ndarray, tm: np.ndarray, sm: float | int) -> np.ndarray:
        rm = rm.T
        co = rm.dot(co_)
        co = co/sm
        co = np.subtract(co.T, rm.dot(tm)).T

        return sm*function_(co, *params_)

    def apply(self,
              function_: Callable[[np.ndarray, tuple], np.ndarray],
              co_: np.ndarray,
              params_: tuple) -> np.ndarray:
        """
        Apply the transformations to the geometry (SDF).
        
        Args:
            function_: Original SDF
            co_:  Point cloud of coordinates with shape (D, N);
                D - number of dimensions (2 or 3);
                N - number of points in the point cloud.
            params_: Parameters of the SDF.
        
        Returns:
            Signed Distance field of shape (N,).
        """
        return self.apply_ec_transforms(function_, co_, params_,
                                        self.rotation_matrix,
                                        self.center,
                                        self.scale)


class EuclideanTransformPoints:
    """
    Class containing the Euclidian transforms which can be applied to a point cloud.
    """

    def __init__(self):
        self._et: list = []
        self._center: np.ndarray | tuple | list = np.asarray((0.0, 0.0, 0.0))
        self._scale: np.ndarray | tuple | list | float | int = 1.0
        self._rot_matrix: np.ndarray = np.eye(3)
        self._angle: float | int = 0.0
        self._axis: np.ndarray | tuple | list = np.asarray((0.0, 0.0, 1.0))

    @property
    def transformations(self) -> list:
        """All the Euclidian transformations which were applied to the point cloud in chronological order.

        Returns:
            List of Euclidian transformations.
        """
        return self._et

    @property
    def center(self) -> np.ndarray | tuple | list:
        """Center of mass of the point cloud.

        Returns:
            Position vector of the center of mass of the point cloud.
        """
        return self._center

    @property
    def scale(self) -> np.ndarray | tuple | list | float | int:
        """Scale factors of the point cloud.

        Returns:
            Scale factor.
        """
        return self._scale

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix applied to the point cloud.

        Returns:
            Rotation matrix (3, 3).
        """
        return self._rot_matrix

    @property
    def rotation_axis(self) -> np.ndarray | tuple | list:
        """Vector representing the axis of rotation.

        Returns:
            Axis of rotation.
        """
        return self._axis

    @property
    def rotation_angle(self) -> float | int:
        """Angle by which the point cloud is rotated around the axis of rotation.

        Returns:
           Angle of rotation.
        """
        return self._angle

    def set_location(self, center: np.ndarray | tuple | list):
        """Set the position of the point cloud in 3D space.
        
        Args:
            center: 3vector which defines the new position of the point cloud.
        """
        self._et.append("set_location")

        center = np.asarray(center)
        if center.size <= 3:
            self._center[:center.size] = center
        else:
            raise SyntaxError(f"Array {center} is of incorrect size!")

    def move(self, move_vector: np.ndarray | tuple | list):
        """
        Move the point cloud by a vector.
        
        Args:
            move_vector: 3vector by which the point cloud is moved.
        """
        self._et.append("move")

        vector = np.asarray(move_vector)
        if vector.size <= 3:
            self._center += vector
        else:
            raise SyntaxError(f"Array {vector} is of incorrect size!")

    def set_scale(self, scale: np.ndarray | tuple | list | float | int):
        """
        Sets the scaling of the point cloud.
        
        Args:
            scale: Scaling factors.
        """
        self._et.append("set_scale")

        if any(isinstance(scale, type_) for type_ in [float, int]):
            self._scale = scale*np.ones(3)
        elif any(isinstance(scale, type_) for type_ in [np.ndarray, tuple, list]):
            scale_ = np.ones(3)
            scale = np.asarray(scale)
            scale_[:scale.size] = scale
            self._scale = scale_
        else:
            raise TypeError("Wrong data type, try a 3vector (np.ndarray, tuple, list) or a scalar.")

    def rescale(self, scale: np.ndarray | tuple | list | float | int):
        """
        Multiplies the existing scaling of the point cloud by the scale factors.
        
        Args:
            scale: Scale factors.
        """
        self._et.append("rescale")

        if any(isinstance(scale, type_) for type_ in [float, int]):
            self._scale *= scale*np.ones(3)
        elif any(isinstance(scale, type_) for type_ in [np.ndarray, tuple, list]):
            self._scale = np.multiply(self._scale, scale)
        else:
            raise TypeError("Wrong data type, try a 3vector (np.ndarray, tuple, list) or a scalar.")

    @staticmethod
    def get_rotation_matrix(angle: float | int, axis: np.ndarray | tuple | list) -> tuple:
        """
        Get the rotation matrix from the rotation angle and axis of rotation.
        
        Args:
            angle: Angle of rotation.
            axis: Axis of rotation.
        
        Returns:
            Rotation matrix (3, 3).
        """
        axis = np.asarray(axis)
        if axis.size <= 3:
            axis_ = np.zeros(3)
            axis_[:axis.size] = axis
        else:
            raise SyntaxError(f"Array {axis} is of incorrect size!")

        if any(isinstance(angle, type_) for type_ in [float, int]):
            angle_ = angle
        else:
            raise TypeError("Rotation angle must be a float or an int.")

        r = srot.from_rotvec(angle_ * axis_)
        return r.as_matrix(), angle_, axis_

    def set_rotation(self, angle: float | int, axis: np.ndarray | tuple | list):
        """
        Sets the rotation matrix from the rotation angle and axis.
        
        Args:
            angle: Angle of rotation.
            axis: Axis of rotation.
        """
        self._et.append("set_rotation")

        self._rot_matrix, self._angle, self._axis = self.get_rotation_matrix(angle, axis)

    def rotate_rotvec(self, angle: float | int, axis: np.ndarray | tuple | list):
        """
        Multiplies the previous existing rotation matrix by a rotation matrix calculated
        from the rotation angle and axis of rotation.
        
        Args:
            angle: Angle of rotation.
            axis: Axis of rotation.
        """
        de = np.array_equal(axis, np.zeros(3)[:axis.size])
        if de:
            raise ValueError("Axis cannot be zero!")

        axis = axis/np.linalg.norm(axis)
        rot_matrix, angle_, axis_ = self.get_rotation_matrix(angle, axis)

        self.rotate_matrix(rot_matrix)

    def rotate_matrix(self, rotation_matrix: np.ndarray | tuple | list):
        """
        Multiplies the previous existing rotation matrix by a given rotation matrix.
        
        Args:
            rotation_matrix: Rotation matrix (3, 3).
        """
        self._rot_matrix = np.matmul(rotation_matrix, self._rot_matrix)

        r = srot.from_matrix(self._rot_matrix)
        rot_vec_ = r.as_rotvec()
        self._angle = np.linalg.norm(rot_vec_)
        if not self._angle == 0:
            self._axis = rot_vec_ / self._angle
        else:
            self._axis = np.asarray((0.0, 0.0, 1.0))

    def rotate(self, *inputs: np.ndarray | tuple):
        """
        Rotates by a rotation matrix or a rotation vector.
        
        Args:
            inputs: Rotation metrix or angle and axis vector.
        """
        self._et.append("rotate")

        n_inputs = len(inputs)

        if n_inputs==1:
            rot_matrix = np.asarray(inputs)
            self.rotate_matrix(rot_matrix)

        if n_inputs==2:
            angle, axis = inputs
            self.rotate_rotvec(angle, np.asarray(axis))

        if n_inputs>2:
            raise SyntaxError("Wrong number of inputs!")

    @staticmethod
    def apply_ec_transforms(points_: np.ndarray,
                            rm: np.ndarray, tm: np.ndarray, sm: np.ndarray) -> np.ndarray:

        co = np.multiply(points_.T, sm).T
        rm = rm.T
        co = rm.dot(co)
        co = np.subtract(co.T, rm.dot(tm)).T

        return co

    def apply(self, points_: np.ndarray) -> np.ndarray:
        """
        Apply the transformations to the point cloud.
        
        Args:
            points_: Point cloud of shape (D, N)
        
        Returns:
            Transformed point cloud of shape (D, N).
        """
        return self.apply_ec_transforms(points_,
                                        self.rotation_matrix,
                                        self.center,
                                        self.scale)