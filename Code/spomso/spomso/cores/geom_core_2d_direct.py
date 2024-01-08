# Copyright (C) 2024 Peter Ropač
# This file is part of SPOMSO.
# SPOMSO is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# SPOMSO is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with SPOMSO. If not, see <https://www.gnu.org/licenses/>.

from transformations import EuclideanTransform
from spomso.cores.modifications import ModifyObject
from sdf_2D import sdf_circle


class CircleDirect(EuclideanTransform, ModifyObject):

    def __init__(self, radius):
        EuclideanTransform.__init__(self)
        ModifyObject.__init__(self, sdf_circle)
        self._radius = radius
        self._sdf = sdf_circle

    def radius(self):
        return self._radius

    def create(self, co):
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, (self.radius, ))

    def propagate(self, co, *parameters_):
        self._sdf = self.modified_object
        return self.apply(self._sdf, co, (self.radius,))