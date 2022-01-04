from itertools import chain
from typing import ClassVar, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import attr

from ... import Config
from ... import schema as oai
from ... import utils
from ..errors import ParseError, PropertyError
from .enum_property import EnumProperty
from .property import Property
from .reference_property import ReferenceProperty
from .class_ import Class
from .schemas import Schemas, parse_reference_path


@attr.s(auto_attribs=True, frozen=True)
class ModelProperty(Property):
    """A property which refers to another Schema"""

    class_info: Class
    required_properties: List[Property]
    optional_properties: List[Property]
    description: str
    relative_imports: Set[str]
    additional_properties: Union[bool, Property]
    _json_type_string: ClassVar[str] = "Dict[str, Any]"

    template: ClassVar[str] = "model_property.py.jinja"
    json_is_dict: ClassVar[bool] = True
    is_multipart_body: bool = False

    def get_base_type_string(self) -> str:
        return self.class_info.name

    def get_imports(self, *, prefix: str) -> Set[str]:
        """
        Get a set of import strings that should be included when this property is used somewhere

        Args:
            prefix: A prefix to put before any relative (local) module names. This should be the number of . to get
            back to the root of the generated client.
        """
        imports = super().get_imports(prefix=prefix)
        imports.update(
            {
                f"from {prefix}models.{self.class_info.module_name} import {self.class_info.name}",
                "from typing import Dict",
                "from typing import cast",
            }
        )
        return imports


def _get_additional_properties(
    *,
    schema_additional: Union[None, bool, oai.Reference, oai.Schema],
    schemas: Schemas,
    class_name: str,
    config: Config,
) -> Tuple[Union[bool, Property, PropertyError], Schemas]:
    from . import property_from_data

    if schema_additional is None:
        return True, schemas

    if isinstance(schema_additional, bool):
        return schema_additional, schemas

    if isinstance(schema_additional, oai.Schema) and not any(schema_additional.dict().values()):
        # An empty schema
        return True, schemas

    additional_properties, schemas = property_from_data(
        name="AdditionalProperty",
        required=True,  # in the sense that if present in the dict will not be None
        data=schema_additional,
        schemas=schemas,
        parent_name=class_name,
        config=config,
    )
    return additional_properties, schemas


def build_model_property(
    *, data: oai.Schema, name: str, schemas: Schemas, required: bool, parent_name: Optional[str], config: Config
) -> Tuple[Union[ModelProperty, PropertyError], Schemas]:
    """
    A single ModelProperty from its OAI data

    Args:
        data: Data of a single Schema
        name: Name by which the schema is referenced, such as a model name.
            Used to infer the type name if a `title` property is not available.
        schemas: Existing Schemas which have already been processed (to check name conflicts)
        required: Whether or not this property is required by the parent (affects typing)
        parent_name: The name of the property that this property is inside of (affects class naming)
        config: Config data for this run of the generator, used to modifying names
    """
    class_string = data.title or name
    if parent_name:
        class_string = f"{utils.pascal_case(parent_name)}{utils.pascal_case(class_string)}"
    class_info = Class.from_string(string=class_string, config=config)

    additional_properties, schemas = _get_additional_properties(
        schema_additional=data.additionalProperties, schemas=schemas, class_name=class_info.name, config=config
    )

    if isinstance(additional_properties, PropertyError):
        return additional_properties, schemas

    prop = ModelProperty(
        class_info=class_info,
        # required_properties=property_data.required_props,
        required_properties=[],
        # optional_properties=property_data.optional_props,
        optional_properties=[],
        relative_imports=set(),
        description=data.description or "",
        default=None,
        nullable=data.nullable,
        required=required,
        name=name,
        additional_properties=additional_properties,
        python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
        example=data.example,
    )
    if class_info.name in schemas.classes_by_name:
        error = PropertyError(data=data, detail=f'Attempted to generate duplicate models with name "{class_info.name}"')
        return error, schemas

    schemas = attr.evolve(
        schemas,
        classes_by_name={**schemas.classes_by_name, class_info.name: prop},
        data_by_class_name={**schemas.data_by_class_name, class_info.name: data}
    )
    return prop, schemas
