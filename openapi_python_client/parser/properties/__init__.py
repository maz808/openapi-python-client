__all__ = [
    "AnyProperty",
    "Class",
    "EnumProperty",
    "ModelProperty",
    "Property",
    "Schemas",
    "build_schemas",
    "property_from_data",
]

from itertools import chain
from typing import Any, ClassVar, Dict, Generic, Iterable, List, NamedTuple, Optional, Set, Tuple, TypeVar, Union

import attr

from ... import Config
from ... import schema as oai
from ... import utils
from ..errors import ParseError, PropertyError, ValidationError
from .converter import convert, convert_chain
from .enum_property import EnumProperty
from .model_property import ModelProperty, build_model_property
from .property import Property
from .reference_property import ReferenceProperty
from .class_ import Class
from .meta import get_enum_default
from .schemas import Schemas, parse_reference_path, update_schemas_with_data


@attr.s(auto_attribs=True, frozen=True)
class AnyProperty(Property):
    """A property that can be any type (used for empty schemas)"""

    _type_string: ClassVar[str] = "Any"
    _json_type_string: ClassVar[str] = "Any"


@attr.s(auto_attribs=True, frozen=True)
class NoneProperty(Property):
    """A property that can only be None"""

    _type_string: ClassVar[str] = "None"
    _json_type_string: ClassVar[str] = "None"


@attr.s(auto_attribs=True, frozen=True)
class StringProperty(Property):
    """A property of type str"""

    max_length: Optional[int] = None
    pattern: Optional[str] = None
    _type_string: ClassVar[str] = "str"
    _json_type_string: ClassVar[str] = "str"
    _allowed_locations: ClassVar[Set[oai.ParameterLocation]] = {
        oai.ParameterLocation.QUERY,
        oai.ParameterLocation.PATH,
        oai.ParameterLocation.COOKIE,
        oai.ParameterLocation.HEADER,
    }


@attr.s(auto_attribs=True, frozen=True)
class DateTimeProperty(Property):
    """
    A property of type datetime.datetime
    """

    _type_string: ClassVar[str] = "datetime.datetime"
    _json_type_string: ClassVar[str] = "str"
    template: ClassVar[str] = "datetime_property.py.jinja"

    def get_imports(self, *, prefix: str) -> Set[str]:
        """
        Get a set of import strings that should be included when this property is used somewhere

        Args:
            prefix: A prefix to put before any relative (local) module names. This should be the number of . to get
            back to the root of the generated client.
        """
        imports = super().get_imports(prefix=prefix)
        imports.update({"import datetime", "from typing import cast", "from dateutil.parser import isoparse"})
        return imports


@attr.s(auto_attribs=True, frozen=True)
class DateProperty(Property):
    """A property of type datetime.date"""

    _type_string: ClassVar[str] = "datetime.date"
    _json_type_string: ClassVar[str] = "str"
    template: ClassVar[str] = "date_property.py.jinja"

    def get_imports(self, *, prefix: str) -> Set[str]:
        """
        Get a set of import strings that should be included when this property is used somewhere

        Args:
            prefix: A prefix to put before any relative (local) module names. This should be the number of . to get
            back to the root of the generated client.
        """
        imports = super().get_imports(prefix=prefix)
        imports.update({"import datetime", "from typing import cast", "from dateutil.parser import isoparse"})
        return imports


@attr.s(auto_attribs=True, frozen=True)
class FileProperty(Property):
    """A property used for uploading files"""

    _type_string: ClassVar[str] = "File"
    # Return type of File.to_tuple()
    _json_type_string: ClassVar[str] = "FileJsonType"
    template: ClassVar[str] = "file_property.py.jinja"

    def get_imports(self, *, prefix: str) -> Set[str]:
        """
        Get a set of import strings that should be included when this property is used somewhere

        Args:
            prefix: A prefix to put before any relative (local) module names. This should be the number of . to get
            back to the root of the generated client.
        """
        imports = super().get_imports(prefix=prefix)
        imports.update({f"from {prefix}types import File, FileJsonType", "from io import BytesIO"})
        return imports


@attr.s(auto_attribs=True, frozen=True)
class FloatProperty(Property):
    """A property of type float"""

    _type_string: ClassVar[str] = "float"
    _json_type_string: ClassVar[str] = "float"
    _allowed_locations: ClassVar[Set[oai.ParameterLocation]] = {
        oai.ParameterLocation.QUERY,
        oai.ParameterLocation.PATH,
        oai.ParameterLocation.COOKIE,
        oai.ParameterLocation.HEADER,
    }
    template: ClassVar[str] = "float_property.py.jinja"


@attr.s(auto_attribs=True, frozen=True)
class IntProperty(Property):
    """A property of type int"""

    _type_string: ClassVar[str] = "int"
    _json_type_string: ClassVar[str] = "int"
    _allowed_locations: ClassVar[Set[oai.ParameterLocation]] = {
        oai.ParameterLocation.QUERY,
        oai.ParameterLocation.PATH,
        oai.ParameterLocation.COOKIE,
        oai.ParameterLocation.HEADER,
    }
    template: ClassVar[str] = "int_property.py.jinja"


@attr.s(auto_attribs=True, frozen=True)
class BooleanProperty(Property):
    """Property for bool"""

    _type_string: ClassVar[str] = "bool"
    _json_type_string: ClassVar[str] = "bool"
    _allowed_locations: ClassVar[Set[oai.ParameterLocation]] = {
        oai.ParameterLocation.QUERY,
        oai.ParameterLocation.PATH,
        oai.ParameterLocation.COOKIE,
        oai.ParameterLocation.HEADER,
    }
    template: ClassVar[str] = "boolean_property.py.jinja"


InnerProp = TypeVar("InnerProp", bound=Property)


@attr.s(auto_attribs=True, frozen=True)
class ListProperty(Property, Generic[InnerProp]):
    """A property representing a list (array) of other properties"""

    inner_property: InnerProp
    template: ClassVar[str] = "list_property.py.jinja"

    def get_base_type_string(self) -> str:
        return f"List[{self.inner_property.get_type_string()}]"

    def get_base_json_type_string(self) -> str:
        return f"List[{self.inner_property.get_type_string(json=True)}]"

    def get_instance_type_string(self) -> str:
        """Get a string representation of runtime type that should be used for `isinstance` checks"""
        return "list"

    def get_imports(self, *, prefix: str) -> Set[str]:
        """
        Get a set of import strings that should be included when this property is used somewhere

        Args:
            prefix: A prefix to put before any relative (local) module names. This should be the number of . to get
            back to the root of the generated client.
        """
        imports = super().get_imports(prefix=prefix)
        imports.update(self.inner_property.get_imports(prefix=prefix))
        imports.add("from typing import cast, List")
        return imports


@attr.s(auto_attribs=True, frozen=True)
class UnionProperty(Property):
    """A property representing a Union (anyOf) of other properties"""

    inner_properties: List[Property]
    template: ClassVar[str] = "union_property.py.jinja"

    def _get_inner_type_strings(self, json: bool = False) -> Set[str]:
        return {p.get_type_string(no_optional=True, json=json) for p in self.inner_properties}

    @staticmethod
    def _get_type_string_from_inner_type_strings(inner_types: Set[str]) -> str:
        if len(inner_types) == 1:
            return inner_types.pop()
        return f"Union[{', '.join(sorted(inner_types))}]"

    def get_base_type_string(self) -> str:
        return self._get_type_string_from_inner_type_strings(self._get_inner_type_strings(json=False))

    def get_base_json_type_string(self) -> str:
        return self._get_type_string_from_inner_type_strings(self._get_inner_type_strings(json=True))

    def get_type_strings_in_union(self, no_optional: bool = False, json: bool = False) -> Set[str]:
        """
        Get the set of all the types that should appear within the `Union` representing this property.

        This function is called from the union property macros, thus the public visibility.

        Args:
            no_optional: Do not include `None` or `Unset` in this set.
            json: If True, this returns the JSON types, not the Python types, of this property.

        Returns:
            A set of strings containing the types that should appear within `Union`.
        """
        type_strings = self._get_inner_type_strings(json=json)
        if no_optional:
            return type_strings
        if self.nullable:
            type_strings.add("None")
        if not self.required:
            type_strings.add("Unset")
        return type_strings

    def get_type_string(self, no_optional: bool = False, json: bool = False) -> str:
        """
        Get a string representation of type that should be used when declaring this property.
        This implementation differs slightly from `Property.get_type_string` in order to collapse
        nested union types.
        """
        type_strings_in_union = self.get_type_strings_in_union(no_optional=no_optional, json=json)
        return self._get_type_string_from_inner_type_strings(type_strings_in_union)

    def get_imports(self, *, prefix: str) -> Set[str]:
        """
        Get a set of import strings that should be included when this property is used somewhere

        Args:
            prefix: A prefix to put before any relative (local) module names. This should be the number of . to get
            back to the root of the generated client.
        """
        imports = super().get_imports(prefix=prefix)
        for inner_prop in self.inner_properties:
            imports.update(inner_prop.get_imports(prefix=prefix))
        imports.add("from typing import cast, Union")
        return imports


def _string_based_property(
    name: str, required: bool, data: oai.Schema, config: Config
) -> Union[StringProperty, DateProperty, DateTimeProperty, FileProperty]:
    """Construct a Property from the type "string" """
    string_format = data.schema_format
    python_name = utils.PythonIdentifier(value=name, prefix=config.field_prefix)
    if string_format == "date-time":
        return DateTimeProperty(
            name=name,
            required=required,
            default=convert("datetime.datetime", data.default),
            nullable=data.nullable,
            python_name=python_name,
            description=data.description,
            example=data.example,
        )
    if string_format == "date":
        return DateProperty(
            name=name,
            required=required,
            default=convert("datetime.date", data.default),
            nullable=data.nullable,
            python_name=python_name,
            description=data.description,
            example=data.example,
        )
    if string_format == "binary":
        return FileProperty(
            name=name,
            required=required,
            default=None,
            nullable=data.nullable,
            python_name=python_name,
            description=data.description,
            example=data.example,
        )
    return StringProperty(
        name=name,
        default=convert("str", data.default),
        required=required,
        pattern=data.pattern,
        nullable=data.nullable,
        python_name=python_name,
        description=data.description,
        example=data.example,
    )


def build_enum_property(
    *,
    data: oai.Schema,
    name: str,
    required: bool,
    schemas: Schemas,
    enum: Union[List[Optional[str]], List[Optional[int]]],
    parent_name: Optional[str],
    config: Config,
) -> Tuple[Union[EnumProperty, NoneProperty, PropertyError], Schemas]:
    """
    Create an EnumProperty from schema data.

    Args:
        data: The OpenAPI Schema which defines this enum.
        name: The name to use for variables which receive this Enum's value (e.g. model property name)
        required: Whether or not this Property is required in the calling context
        schemas: The Schemas which have been defined so far (used to prevent naming collisions)
        enum: The enum from the provided data. Required separately here to prevent extra type checking.
        parent_name: The context in which this EnumProperty is defined, used to create more specific class names.
        config: The global config for this run of the generator

    Returns:
        A tuple containing either the created property or a PropertyError describing what went wrong AND update schemas.
    """

    if len(enum) == 0:
        return PropertyError(detail="No values provided for Enum", data=data), schemas

    class_name = data.title or name
    if parent_name:
        class_name = f"{utils.pascal_case(parent_name)}{utils.pascal_case(class_name)}"
    class_info = Class.from_string(string=class_name, config=config)

    # OpenAPI allows for null as an enum value, but it doesn't make sense with how enums are constructed in Python.
    # So instead, if null is a possible value, make the property nullable.
    # Mypy is not smart enough to know that the type is right though
    value_list: Union[List[str], List[int]] = [value for value in enum if value is not None]  # type: ignore
    if len(value_list) < len(enum):
        data.nullable = True

    # It's legal to have an enum that only contains null as a value, we don't bother constructing an enum in that case
    if len(value_list) == 0:
        return (
            NoneProperty(
                name=name,
                required=required,
                nullable=False,
                default="None",
                python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
                description=None,
                example=None,
            ),
            schemas,
        )
    values = EnumProperty.values_from_list(value_list)

    if class_info.name in schemas.classes_by_name:
        existing = schemas.classes_by_name[class_info.name]
        if not isinstance(existing, EnumProperty) or values != existing.values:
            return (
                PropertyError(
                    detail=f"Found conflicting enums named {class_info.name} with incompatible values.", data=data
                ),
                schemas,
            )

    value_type = type(next(iter(values.values())))

    prop = EnumProperty(
        name=name,
        required=required,
        nullable=data.nullable,
        class_info=class_info,
        values=values,
        value_type=value_type,
        default=None,
        python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
        description=data.description,
        example=data.example,
    )

    default = get_enum_default(prop, data)
    if isinstance(default, PropertyError):
        return default, schemas
    prop = attr.evolve(prop, default=default)

    schemas = attr.evolve(schemas, classes_by_name={**schemas.classes_by_name, class_info.name: prop})
    return prop, schemas


def build_union_property(
    *, data: oai.Schema, name: str, required: bool, schemas: Schemas, parent_name: str, config: Config
) -> Tuple[Union[UnionProperty, PropertyError], Schemas]:
    """
    Create a `UnionProperty` the right way.

    Args:
        data: The `Schema` describing the `UnionProperty`.
        name: The name of the property where it appears in the OpenAPI document.
        required: Whether or not this property is required where it's being used.
        schemas: The `Schemas` so far describing existing classes / references.
        parent_name: The name of the thing which holds this property (used for renaming inner classes).
        config: User-defined config values for modifying inner properties.

    Returns:
        `(result, schemas)` where `schemas` is the updated version of the input `schemas` and `result` is the
            constructed `UnionProperty` or a `PropertyError` describing what went wrong.
    """
    sub_properties: List[Property] = []

    for i, sub_prop_data in enumerate(chain(data.anyOf, data.oneOf)):
        sub_prop, schemas = property_from_data(
            name=f"{name}_type_{i}",
            required=required,
            data=sub_prop_data,
            schemas=schemas,
            parent_name=parent_name,
            config=config,
        )
        if isinstance(sub_prop, PropertyError):
            return PropertyError(detail=f"Invalid property in union {name}", data=sub_prop_data), schemas
        sub_properties.append(sub_prop)

    default = convert_chain((prop.get_base_type_string() for prop in sub_properties), data.default)
    return (
        UnionProperty(
            name=name,
            required=required,
            default=default,
            inner_properties=sub_properties,
            nullable=data.nullable,
            python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
            description=data.description,
            example=data.example,
        ),
        schemas,
    )


def build_list_property(
    *, data: oai.Schema, name: str, required: bool, schemas: Schemas, parent_name: str, config: Config
) -> Tuple[Union[ListProperty[Any], PropertyError], Schemas]:
    """
    Build a ListProperty the right way, use this instead of the normal constructor.

    Args:
        data: `oai.Schema` representing this `ListProperty`.
        name: The name of this property where it's used.
        required: Whether or not this `ListProperty` can be `Unset` where it's used.
        schemas: Collected `Schemas` so far containing any classes or references.
        parent_name: The name of the thing containing this property (used for naming inner classes).
        config: User-provided config for overriding default behaviors.

    Returns:
        `(result, schemas)` where `schemas` is an updated version of the input named the same including any inner
        classes that were defined and `result` is either the `ListProperty` or a `PropertyError`.
    """
    if data.items is None:
        return PropertyError(data=data, detail="type array must have items defined"), schemas
    inner_prop, schemas = property_from_data(
        name=f"{name}_item", required=True, data=data.items, schemas=schemas, parent_name=parent_name, config=config
    )
    if isinstance(inner_prop, PropertyError):
        inner_prop.header = f'invalid data in items of array named "{name}"'
        return inner_prop, schemas
    return (
        ListProperty(
            name=name,
            required=required,
            default=None,
            inner_property=inner_prop,
            nullable=data.nullable,
            python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
            description=data.description,
            example=data.example,
        ),
        schemas,
    )


# pylint: disable=too-many-arguments
def _property_from_ref(
    name: str,
    required: bool,
    parent: Union[oai.Schema, None],
    data: oai.Reference,
    schemas: Schemas,
    config: Config,
) -> Tuple[Union[Property, PropertyError], Schemas]:
    ref_path = parse_reference_path(data.ref)
    if isinstance(ref_path, ParseError):
        return PropertyError(data=data, detail=ref_path.detail), schemas

    prop = ReferenceProperty(
        required=required,
        name=name,
        python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
        nullable=False,
        description=None,
        example=None,
        default=None,
        ref_path=ref_path,
        parent=parent,
    )

    existing = schemas.classes_by_reference.get(prop.ref_path)
    if existing:
        prop.resolve(existing)
    else:
        schemas.unresolved_references.append(prop)

    return prop, schemas


# pylint: disable=too-many-arguments,too-many-return-statements
def _property_from_data(
    name: str,
    required: bool,
    data: Union[oai.Reference, oai.Schema],
    schemas: Schemas,
    parent_name: str,
    config: Config,
) -> Tuple[Union[Property, PropertyError], Schemas]:
    """Generate a Property from the OpenAPI dictionary representation of it"""
    name = utils.remove_string_escapes(name)
    if isinstance(data, oai.Reference):
        return _property_from_ref(
            name=name,
            required=required,
            parent=None,
            data=data,
            schemas=schemas,
            config=config,
        )

    sub_data: List[Union[oai.Schema, oai.Reference]] = data.allOf + data.anyOf + data.oneOf
    # A union of a single reference should just be passed through to that reference (don't create copy class)
    if len(sub_data) == 1 and isinstance(sub_data[0], oai.Reference):
        return _property_from_ref(
            name=name, required=required, parent=data, data=sub_data[0], schemas=schemas, config=config
        )

    if data.enum:
        return build_enum_property(
            data=data,
            name=name,
            required=required,
            schemas=schemas,
            enum=data.enum,
            parent_name=parent_name,
            config=config,
        )
    if data.anyOf or data.oneOf:
        return build_union_property(
            data=data, name=name, required=required, schemas=schemas, parent_name=parent_name, config=config
        )
    if data.type == oai.DataType.STRING:
        return _string_based_property(name=name, required=required, data=data, config=config), schemas
    if data.type == oai.DataType.NUMBER:
        return (
            FloatProperty(
                name=name,
                default=convert("float", data.default),
                required=required,
                nullable=data.nullable,
                python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
                description=data.description,
                example=data.example,
            ),
            schemas,
        )
    if data.type == oai.DataType.INTEGER:
        return (
            IntProperty(
                name=name,
                default=convert("int", data.default),
                required=required,
                nullable=data.nullable,
                python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
                description=data.description,
                example=data.example,
            ),
            schemas,
        )
    if data.type == oai.DataType.BOOLEAN:
        return (
            BooleanProperty(
                name=name,
                required=required,
                default=convert("bool", data.default),
                nullable=data.nullable,
                python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
                description=data.description,
                example=data.example,
            ),
            schemas,
        )
    if data.type == oai.DataType.ARRAY:
        return build_list_property(
            data=data, name=name, required=required, schemas=schemas, parent_name=parent_name, config=config
        )
    if data.type == oai.DataType.OBJECT or data.allOf:
        return build_model_property(
            data=data, name=name, schemas=schemas, required=required, parent_name=parent_name, config=config
        )
    return (
        AnyProperty(
            name=name,
            required=required,
            nullable=False,
            default=None,
            python_name=utils.PythonIdentifier(value=name, prefix=config.field_prefix),
            description=data.description,
            example=data.example,
        ),
        schemas,
    )


def property_from_data(
    *,
    name: str,
    required: bool,
    data: Union[oai.Reference, oai.Schema],
    schemas: Schemas,
    parent_name: str,
    config: Config,
) -> Tuple[Union[Property, PropertyError], Schemas]:
    """
    Build a Property from an OpenAPI schema or reference. This Property represents a single input or output for a
    generated API operation.

    Args:
        name: The name of the property, defined in OpenAPI as the key pointing at the schema. This is the parameter used
            to send this data to an API or that the API will respond with. This will be used to generate a `python_name`
            which is the name of the variable/attribute in generated Python.
        required: Whether or not this property is required in whatever source is creating it. OpenAPI defines this by
            including the property's name in the `required` list. If the property is required, `Unset` will not be
            included in the generated code's available types.
        data: The OpenAPI schema or reference that defines the details of this property (e.g. type, sub-properties).
        schemas: A structure containing all of the parsed schemas so far that will become generated classes. This is
            used to resolve references and to ensure that conflicting class names are not generated.
        parent_name: The name of the thing above this property, prepended to generated class names to reduce the chance
            of duplication.
        config: Contains the parsed config that the user provided to tweak generation settings. Needed to apply class
            name overrides for generated classes.

    Returns:
        A tuple containing either the parsed Property or a PropertyError (if something went wrong) and the updated
        Schemas (including any new classes that should be generated).
    """
    try:
        return _property_from_data(
            name=name,
            required=required,
            data=data,
            schemas=schemas,
            parent_name=parent_name,
            config=config,
        )
    except ValidationError:
        return PropertyError(detail="Failed to validate default value", data=data), schemas


def _values_are_subset(first: EnumProperty, second: EnumProperty) -> bool:
    return set(first.values.items()) <= set(second.values.items())


def _types_are_subset(first: EnumProperty, second: Property) -> bool:
    from . import IntProperty, StringProperty

    if first.value_type == int and isinstance(second, IntProperty):
        return True
    if first.value_type == str and isinstance(second, StringProperty):
        return True
    return False


def _enum_subset(first: Property, second: Property) -> Optional[EnumProperty]:
    """Return the EnumProperty that is the subset of the other, if possible."""

    if isinstance(first, EnumProperty):
        if isinstance(second, EnumProperty):
            if _values_are_subset(first, second):
                return first
            if _values_are_subset(second, first):  # pylint: disable=arguments-out-of-order
                return second
            return None
        return first if _types_are_subset(first, second) else None
    # pylint: disable=arguments-out-of-order
    if isinstance(second, EnumProperty) and _types_are_subset(second, first):
        return second
    return None


def _merge_properties(first: Property, second: Property) -> Union[Property, PropertyError]:
    if isinstance(first, ReferenceProperty):
        first = first.ref_resolution
    if isinstance(second, ReferenceProperty):
        second = second.ref_resolution
    nullable = first.nullable and second.nullable
    required = first.required or second.required

    err = None

    if first.__class__ == second.__class__:
        first = attr.evolve(first, nullable=nullable, required=required)
        second = attr.evolve(second, nullable=nullable, required=required)
        if first == second:
            return first
        err = PropertyError(header="Cannot merge properties", detail="Properties has conflicting values")

    enum_subset = _enum_subset(first, second)
    if enum_subset is not None:
        return attr.evolve(enum_subset, nullable=nullable, required=required)

    return err or PropertyError(
        header="Cannot merge properties",
        detail=f"{first.__class__}, {second.__class__}Properties have incompatible types",
    )


class _PropertyData(NamedTuple):
    optional_props: List[Property]
    required_props: List[Property]
    schemas: Schemas


# pylint: disable=too-many-locals,too-many-branches
def _process_properties(
    *, data: oai.Schema, schemas: Schemas, class_name: str, config: Config
) -> Union[_PropertyData, PropertyError]:
    from . import property_from_data

    properties: Dict[str, Property] = {}
    required_set = set(data.required or [])

    def _add_if_no_conflict(new_prop: Property) -> Optional[PropertyError]:
        nonlocal properties

        existing = properties.get(new_prop.name)
        merged_prop_or_error = _merge_properties(existing, new_prop) if existing else new_prop
        if isinstance(merged_prop_or_error, PropertyError):
            merged_prop_or_error.header = (
                f"Found conflicting properties named {new_prop.name} when creating {class_name}"
            )
            return merged_prop_or_error
        properties[merged_prop_or_error.name] = merged_prop_or_error
        return None

    unprocessed_props = data.properties or {}
    for sub_prop in data.allOf:
        if isinstance(sub_prop, oai.Reference):
            ref_path = parse_reference_path(sub_prop.ref)
            if isinstance(ref_path, ParseError):
                return PropertyError(detail=ref_path.detail, data=sub_prop)
            sub_model = schemas.classes_by_reference.get(ref_path)
            if sub_model is None:
                return PropertyError(f"Reference {sub_prop.ref} not found")
            if not isinstance(sub_model, ModelProperty):
                return PropertyError("Cannot take allOf a non-object")
            if len(sub_model.required_properties) + len(sub_model.optional_properties) == 0:
                return PropertyError("Recursive and circular references in allOf are not supported", data=sub_prop)
            for prop in chain(sub_model.required_properties, sub_model.optional_properties):
                err = _add_if_no_conflict(prop)
                if err is not None:
                    return err
        else:
            unprocessed_props.update(sub_prop.properties or {})
            required_set.update(sub_prop.required or [])

    for key, value in unprocessed_props.items():
        prop_required = key in required_set
        prop_or_error: Union[Property, PropertyError, None]
        prop_or_error, schemas = property_from_data(
            name=key, required=prop_required, data=value, schemas=schemas, parent_name=class_name, config=config
        )
        if isinstance(prop_or_error, Property):
            prop_or_error = _add_if_no_conflict(prop_or_error)
        if isinstance(prop_or_error, PropertyError):
            return prop_or_error

    required_properties = []
    optional_properties = []
    for prop in properties.values():
        if prop.required and not prop.nullable:
            required_properties.append(prop)
        else:
            optional_properties.append(prop)

    return _PropertyData(
        optional_props=optional_properties,
        required_props=required_properties,
        schemas=schemas,
    )


def _build_properties(model: ModelProperty, data: oai.Schema, schemas: Schemas, config: Config) -> Tuple[Optional[PropertyError], Schemas]:
    property_data = _process_properties(data=data, schemas=schemas, class_name=model.class_info.name, config=config)
    if isinstance(property_data, PropertyError):
        return property_data
    model.required_properties.extend(property_data.required_props)
    model.optional_properties.extend(property_data.optional_props)
    schemas = property_data.schemas
    return schemas


def _resolve_imports(component: ModelProperty):
    for prop in chain(component.required_properties, component.optional_properties):
        component.relative_imports.update(prop.get_imports(prefix=".."))
        if isinstance(component.additional_properties, Property):
            component.relative_imports.update(component.additional_properties.get_imports(prefix=".."))


def build_schemas(
    *, components: Dict[str, Union[oai.Reference, oai.Schema]], schemas: Schemas, config: Config
) -> Schemas:
    """Get a list of Schemas from an OpenAPI dict"""
    to_process: Iterable[Tuple[str, Union[oai.Reference, oai.Schema]]] = components.items()
    still_making_progress = True
    errors: List[PropertyError] = []

    # References could have forward References so keep going as long as we are making progress
    while still_making_progress:
        still_making_progress = False
        errors = []
        next_round = []
        # Only accumulate errors from the last round, since we might fix some along the way
        for name, data in to_process:
            if isinstance(data, oai.Reference):
                schemas.errors.append(PropertyError(data=data, detail="Reference schemas are not supported."))
                continue
            ref_path = parse_reference_path(f"#/components/schemas/{name}")
            if isinstance(ref_path, ParseError):
                schemas.errors.append(PropertyError(detail=ref_path.detail, data=data))
                continue
            schemas_or_err = update_schemas_with_data(ref_path=ref_path, data=data, schemas=schemas, config=config)
            if isinstance(schemas_or_err, PropertyError):
                next_round.append((name, data))
                errors.append(schemas_or_err)
                continue
            schemas = schemas_or_err
            still_making_progress = True
        to_process = next_round

    schemas.errors.extend(errors)
    schemas.resolve_references()

    still_making_progress = True
    # Models which refer to other models in their allOf must be processed after their referenced models
    while still_making_progress:
        still_making_progress = False
        # Only accumulate errors from the last round, since we might fix some along the way
        errors = []
        next_round = {}
        while schemas.data_by_class_name:
            class_name, data = schemas.data_by_class_name.popitem()
            schemas_or_err = _build_properties(schemas.classes_by_name[class_name], data, schemas, config=config)
            if isinstance(schemas_or_err, PropertyError):
                if isinstance(schemas_or_err.data, oai.Reference) and schemas_or_err.data.ref.endswith(class_name):  # pragma: nocover
                    schemas_or_err.detail += (f"\n\nRecursive allOf reference found in '{schemas_or_err.data.ref}'")
                    del schemas.classes_by_name[class_name]
                    schemas.errors.append(schemas_or_err)
                    continue
                next_round[class_name] = data
                errors.append(schemas_or_err)
                continue
            schemas = schemas_or_err
            still_making_progress = True
        schemas.data_by_class_name.update(next_round)

    schemas.errors.extend(errors)

    for component in schemas.classes_by_name.values():
        if not isinstance(component, ModelProperty):
            continue
        _resolve_imports(component)

    return schemas
