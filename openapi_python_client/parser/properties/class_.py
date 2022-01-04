"""Module created to solve circular import errors"""
import attr

from ... import Config
from ...utils import PythonIdentifier, ClassName


@attr.s(auto_attribs=True, frozen=True)
class Class:
    """Represents Python class which will be generated from an OpenAPI schema"""

    name: ClassName
    module_name: PythonIdentifier

    @staticmethod
    def from_string(*, string: str, config: Config) -> "Class":
        """Get a Class from an arbitrary string"""
        class_name = string.split("/")[-1]  # Get rid of ref path stuff
        class_name = ClassName(class_name, config.field_prefix)
        override = config.class_overrides.get(class_name)

        if override is not None and override.class_name is not None:
            class_name = ClassName(override.class_name, config.field_prefix)

        if override is not None and override.module_name is not None:
            module_name = override.module_name
        else:
            module_name = class_name
        module_name = PythonIdentifier(module_name, config.field_prefix)

        return Class(name=class_name, module_name=module_name)
