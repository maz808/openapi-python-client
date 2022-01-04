__all__ = ["Class", "Schemas", "parse_reference_path", "update_schemas_with_data"]

from typing import TYPE_CHECKING, Dict, List, Union, cast
from urllib.parse import urlparse

import attr

from ... import Config
from ... import schema as oai
from ...utils import ClassName, PythonIdentifier
from ..errors import ParseError, PropertyError
from .reference_property import ReferenceProperty, ReferencePath

if TYPE_CHECKING:  # pragma: no cover
    from .property import Property
else:
    Property = "Property"  # pylint: disable=invalid-name


def parse_reference_path(ref_path_raw: str) -> Union[ReferencePath, ParseError]:
    """
    Takes a raw string provided in a `$ref` and turns it into a validated `_ReferencePath` or a `ParseError` if
    validation fails.

    See Also:
        - https://swagger.io/docs/specification/using-ref/
    """
    parsed = urlparse(ref_path_raw)
    if parsed.scheme or parsed.path:
        return ParseError(detail=f"Remote references such as {ref_path_raw} are not supported yet.")
    return cast(ReferencePath, parsed.fragment)


@attr.s(auto_attribs=True, frozen=True)
class Schemas:
    """Structure for containing all defined, shareable, and reusable schemas (attr classes and Enums)"""

    classes_by_reference: Dict[ReferencePath, Property] = attr.ib(factory=dict)
    classes_by_name: Dict[ClassName, Property] = attr.ib(factory=dict)
    data_by_class_name: Dict[ClassName, oai.Schema] = attr.ib(factory=dict)
    unresolved_references: List[ReferenceProperty] = attr.ib(factory=list)
    errors: List[ParseError] = attr.ib(factory=list)

    def resolve_references(self):
        """Resolve unresolved ReferenceProperty instances"""
        for ref_prop in self.unresolved_references:
            if ref_prop.ref_resolution: continue
            ref_prop.resolve(self.classes_by_reference.get(ref_prop.ref_path))


def update_schemas_with_data(
    *, ref_path: ReferencePath, data: oai.Schema, schemas: Schemas, config: Config
) -> Union[Schemas, PropertyError]:
    """
    Update a `Schemas` using some new reference.

    Args:
        ref_path: The output of `parse_reference_path` (validated $ref).
        data: The schema of the thing to add to Schemas.
        schemas: `Schemas` up until now.
        config: User-provided config for overriding default behavior.

    Returns:
        Either the updated `schemas` input or a `PropertyError` if something went wrong.

    See Also:
        - https://swagger.io/docs/specification/using-ref/
    """
    from . import property_from_data

    prop: Union[PropertyError, Property]
    prop, schemas = property_from_data(
        data=data, name=ref_path, schemas=schemas, required=True, parent_name="", config=config
    )

    if isinstance(prop, PropertyError):
        prop.detail = f"{prop.header}: {prop.detail}"
        prop.header = f"Unable to parse schema {ref_path}"
        return prop

    schemas = attr.evolve(schemas, classes_by_reference={ref_path: prop, **schemas.classes_by_reference})
    return schemas
