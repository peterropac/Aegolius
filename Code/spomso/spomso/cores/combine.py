# Copyright (C) 2023 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from spomso.cores.geom import GenericGeometry
from typing import Callable

def smoothmin_poly2(x, y, a):
    # https://iquilezles.org/articles/smin/
    h = np.maximum(a - np.abs(x - y), 0.0)/a
    return np.minimum(x, y) - h * h * a/4.0


def smoothmin_poly3(x, y, a):
    # https://iquilezles.org/articles/smin/
    if not a == 0.0:
        h = np.maximum(a - np.abs(x - y), 0.0)/a
        return np.minimum(x, y) - h * h * h * a / 6.0
    else:
        return np.minimum(y, y)


def smoothmax_boltzman(x, y, width):
    # https: // en.wikipedia.org / wiki / Smooth_maximum
    exp1 = np.exp(x/width)
    exp2 = np.exp(y/width)

    return (x*exp1 + y*exp2)/(exp1 + exp2)


class CombineGeometry:

    def __init__(self, operation_type: str):
        """
        Constructs all the necessary attributes for the CombineGeometry object.
        :param operation_type: Type of operation with which two or more geometric objects are combined.
        """
        self.operation_type = operation_type
        self._combined_geometry = None
        self.operations = {"UNION2": lambda obj1, obj2: np.minimum(obj1, obj2),
                           "UNION": lambda *objs: np.amin(objs, axis=0),
                           "SUBTRACT2": lambda obj1, obj2: np.maximum(obj1, -obj2),
                           "INTERSECT2": lambda obj1, obj2: np.maximum(obj1, obj2),
                           "INTERSECT": lambda *objs: np.amax(objs, axis=0),
                           }

        self.parametric_operations = {"SMOOTH_UNION2_2": lambda obj1, obj2, width: smoothmin_poly2(obj1,
                                                                                                   obj2,
                                                                                                   width),
                                      "SMOOTH_UNION2": lambda obj1, obj2, width: smoothmin_poly3(obj1,
                                                                                                 obj2,
                                                                                                 width),
                                      "SMOOTH_INTERSECT2": lambda obj1, obj2, width: -smoothmin_poly3(-obj1,
                                                                                                      -obj2,
                                                                                                      width),
                                      "SMOOTH_INTERSECT2_BOLTZMAN": lambda obj1, obj2, width: smoothmax_boltzman(obj1,
                                                                                                                 obj2,
                                                                                                                 width),
                                      "SMOOTH_SUBTRACT2": lambda obj1, obj2, width: -smoothmin_poly3(-obj1,
                                                                                                     obj2,
                                                                                                     width),
                                      "SMOOTH_SUBTRACT2_BOLTZMAN": lambda obj1, obj2, width: smoothmax_boltzman(obj1,
                                                                                                                -obj2,
                                                                                                                width)
                                      }

    @property
    def available_operations(self):
        """
        Available types of operations.
        :return: List of available operations.
        """
        operations_list = list(self.operations.keys())
        print(f"Available non-parametric operations are: {operations_list}")
        return operations_list

    @property
    def available_parametric_operations(self):
        """
        Available types of operations which require a parameter.
        :return: List of available parametric operations.
        """
        operations_list = list(self.parametric_operations.keys())
        print(f"Available parametric operations are: {operations_list}")
        return operations_list

    @property
    def combined_geometry(self) -> Callable[[np.ndarray, tuple], tuple]:
        """
        Returns the SDF of the combined geometries,
        which can be used to create a new geometry using GenericGeometry class.
        :return: SDF of the combined geometries
        """
        return self._combined_geometry

    def combine(self, *combined_objects: tuple) -> GenericGeometry:
        """
        Combines 2 or more geometric objects together.
        :param combined_objects: Tuple containing geometric objects.
        :return: New geometric object.
        """
        if self.operation_type not in self.operations.keys():
            raise SyntaxError(f"{self.operation_type} is not an implemented non-parametric operation.",
                              f"Possible operations are {self.operations.keys}")

        def new_geo_object(co, *params):
            sdfs = []
            for cobject in combined_objects:
                sdf_ = cobject.propagate(co, ())
                sdfs.append(sdf_)

            return self.operations[self.operation_type](*sdfs)

        self._combined_geometry = new_geo_object
        return GenericGeometry(new_geo_object, ())

    def combine_parametric(self, *combined_objects: object, parameters: tuple | float | int) -> GenericGeometry:
        """
        Combines 2 or more geometric objects together, based on the parameters of the operations.
        :param combined_objects: Tuple containing geometric objects.
        :param parameters: Parameters of the operation.
        :return: New geometric object.
        """
        if self.operation_type not in self.parametric_operations.keys():
            raise SyntaxError(f"{self.operation_type} is not an implemented parametric operation.",
                              f"Possible parametric operations are {self.parametric_operations.keys}")

        def new_geo_object(co, *params):
            sdfs = []
            for cobject in combined_objects:
                sdf_ = cobject.propagate(co, ())
                sdfs.append(sdf_)

            return self.parametric_operations[self.operation_type](*sdfs, parameters)

        self._combined_geometry = new_geo_object
        return GenericGeometry(new_geo_object, ())
