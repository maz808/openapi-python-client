"""Module created to solve circular import errors"""
from typing import Union, Optional

from ..errors import PropertyError
from .enum_property import EnumProperty
from ... import schema as oai


def get_enum_default(prop: EnumProperty, data: oai.Schema) -> Union[Optional[str], PropertyError]:
    """
    Run through the available values in an EnumProperty and return the string representing the default value
    in `data`.

    Args:
        prop: The EnumProperty to search for the default value.
        data: The schema containing the default value for this enum.

    Returns:
        If `default` is `None`, then `None`.
            If `default` is a valid value in `prop`, then the string representing that variant (e.g. MyEnum.MY_VARIANT)
            If `default` is a value that doesn't match a variant of the enum, then a `PropertyError`.
    """
    default = data.default
    if default is None:
        return None

    inverse_values = {v: k for k, v in prop.values.items()}
    try:
        return f"{prop.class_info.name}.{inverse_values[default]}"
    except KeyError:
        return PropertyError(detail=f"{default} is an invalid default for enum {prop.class_info.name}", data=data)

