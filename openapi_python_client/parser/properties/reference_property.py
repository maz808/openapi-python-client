import inspect
from typing import NewType, Optional, Any, Union

import attr

from .property import Property
from .enum_property import EnumProperty
from .meta import get_enum_default
from ..errors import PropertyError
from ... import schema as oai

ReferencePath = NewType("ReferencePath", str)


@attr.s(auto_attribs=True, frozen=True)
class ReferenceProperty(Property):
    """A property which refers to another schema"""

    ref_path: ReferencePath
    parent: Optional[oai.Schema]
    ref_resolution: Optional[Property] = None

    @property
    def nullable(self) -> Union[bool, None]:
        if self.parent:
            return self.parent.nullable
        return None

    # `__getattribute__` and `__getattr__` are used to expose attributes from the `ref_resolution`
    # Property object as the attributes of the ReferenceProperty instance. Attributes defined in
    # the ReferenceProperty class are directly retrieved from the ReferenceProperty instance.
    def __getattribute__(self, __name: str) -> Any:
        if not hasattr(super(), __name) or hasattr(object, __name):
            return object.__getattribute__(self, __name)
        return getattr(object.__getattribute__(self, "ref_resolution"), __name)

    def __getattr__(self, __name: str) -> Any:
        return getattr(object.__getattribute__(self, "ref_resolution"), __name)
    
    # @property
    # def ref_resolution(self) -> Union[None, Property]:
    #     return self._ref_resolution

    # @ref_resolution.setter
    # def ref_resolution(self, value: Property) -> Union[None, Property]:
    #     """Single use setter for property. A second assignment will raise an exception"""
    #     if self._ref_resolution:
    #         self._ref_resolution = "THIS SHOULD RAISE AN attr.exceptions.FrozenInstanceError"
    #     object.__setattr__(self, "_ref_resolution", value)

    def resolve(self, property: Property) -> Union[None, PropertyError]:
        """Sets the value of this ReferenceProperty's `ref_resolution` attribute

        Will raise an error if used after `ref_resolution` has been set

        Args:
            property: The `Property` instance which the reference path maps to
        
        Returns:
            PropertyError if it occured when calling `get_enum_default`, otherwise None

        Raises:
            attr.exceptions.FrozenInstanceError: Raised when this ReferenceProperty has already been resolved
        """
        if self.ref_resolution:
            self.ref_resolution = "THIS SHOULD RAISE AN attr.exceptions.FrozenInstanceError"

        prop_copy = attr.evolve(
            property,
            required=object.__getattribute__(self, "required"),
            name=object.__getattribute__(self, "name"),
            python_name=object.__getattribute__(self, "python_name"),
            nullable=object.__getattribute__(self, "nullable") or property.nullable
        )

        if self.parent:
            prop_copy = attr.evolve(prop_copy, nullable=self.parent.nullable)
            if isinstance(prop_copy, EnumProperty):
                default = get_enum_default(prop_copy, self.parent)
                if isinstance(default, PropertyError):
                    return default
                prop_copy = attr.evolve(prop_copy, default=default)

        object.__setattr__(self, "ref_resolution", prop_copy)
