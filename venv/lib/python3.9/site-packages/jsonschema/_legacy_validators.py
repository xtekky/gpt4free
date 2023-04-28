from jsonschema import _utils
from jsonschema.exceptions import ValidationError


def id_of_ignore_ref(property="$id"):
    def id_of(schema):
        """
        Ignore an ``$id`` sibling of ``$ref`` if it is present.

        Otherwise, return the ID of the given schema.
        """
        if schema is True or schema is False or "$ref" in schema:
            return ""
        return schema.get(property, "")
    return id_of


def ignore_ref_siblings(schema):
    """
    Ignore siblings of ``$ref`` if it is present.

    Otherwise, return all keywords.

    Suitable for use with `create`'s ``applicable_validators`` argument.
    """
    ref = schema.get("$ref")
    if ref is not None:
        return [("$ref", ref)]
    else:
        return schema.items()


def dependencies_draft3(validator, dependencies, instance, schema):
    if not validator.is_type(instance, "object"):
        return

    for property, dependency in dependencies.items():
        if property not in instance:
            continue

        if validator.is_type(dependency, "object"):
            yield from validator.descend(
                instance, dependency, schema_path=property,
            )
        elif validator.is_type(dependency, "string"):
            if dependency not in instance:
                message = f"{dependency!r} is a dependency of {property!r}"
                yield ValidationError(message)
        else:
            for each in dependency:
                if each not in instance:
                    message = f"{each!r} is a dependency of {property!r}"
                    yield ValidationError(message)


def dependencies_draft4_draft6_draft7(
    validator,
    dependencies,
    instance,
    schema,
):
    """
    Support for the ``dependencies`` keyword from pre-draft 2019-09.

    In later drafts, the keyword was split into separate
    ``dependentRequired`` and ``dependentSchemas`` validators.
    """
    if not validator.is_type(instance, "object"):
        return

    for property, dependency in dependencies.items():
        if property not in instance:
            continue

        if validator.is_type(dependency, "array"):
            for each in dependency:
                if each not in instance:
                    message = f"{each!r} is a dependency of {property!r}"
                    yield ValidationError(message)
        else:
            yield from validator.descend(
                instance, dependency, schema_path=property,
            )


def disallow_draft3(validator, disallow, instance, schema):
    for disallowed in _utils.ensure_list(disallow):
        if validator.evolve(schema={"type": [disallowed]}).is_valid(instance):
            message = f"{disallowed!r} is disallowed for {instance!r}"
            yield ValidationError(message)


def extends_draft3(validator, extends, instance, schema):
    if validator.is_type(extends, "object"):
        yield from validator.descend(instance, extends)
        return
    for index, subschema in enumerate(extends):
        yield from validator.descend(instance, subschema, schema_path=index)


def items_draft3_draft4(validator, items, instance, schema):
    if not validator.is_type(instance, "array"):
        return

    if validator.is_type(items, "object"):
        for index, item in enumerate(instance):
            yield from validator.descend(item, items, path=index)
    else:
        for (index, item), subschema in zip(enumerate(instance), items):
            yield from validator.descend(
                item, subschema, path=index, schema_path=index,
            )


def items_draft6_draft7_draft201909(validator, items, instance, schema):
    if not validator.is_type(instance, "array"):
        return

    if validator.is_type(items, "array"):
        for (index, item), subschema in zip(enumerate(instance), items):
            yield from validator.descend(
                item, subschema, path=index, schema_path=index,
            )
    else:
        for index, item in enumerate(instance):
            yield from validator.descend(item, items, path=index)


def minimum_draft3_draft4(validator, minimum, instance, schema):
    if not validator.is_type(instance, "number"):
        return

    if schema.get("exclusiveMinimum", False):
        failed = instance <= minimum
        cmp = "less than or equal to"
    else:
        failed = instance < minimum
        cmp = "less than"

    if failed:
        message = f"{instance!r} is {cmp} the minimum of {minimum!r}"
        yield ValidationError(message)


def maximum_draft3_draft4(validator, maximum, instance, schema):
    if not validator.is_type(instance, "number"):
        return

    if schema.get("exclusiveMaximum", False):
        failed = instance >= maximum
        cmp = "greater than or equal to"
    else:
        failed = instance > maximum
        cmp = "greater than"

    if failed:
        message = f"{instance!r} is {cmp} the maximum of {maximum!r}"
        yield ValidationError(message)


def properties_draft3(validator, properties, instance, schema):
    if not validator.is_type(instance, "object"):
        return

    for property, subschema in properties.items():
        if property in instance:
            yield from validator.descend(
                instance[property],
                subschema,
                path=property,
                schema_path=property,
            )
        elif subschema.get("required", False):
            error = ValidationError(f"{property!r} is a required property")
            error._set(
                validator="required",
                validator_value=subschema["required"],
                instance=instance,
                schema=schema,
            )
            error.path.appendleft(property)
            error.schema_path.extend([property, "required"])
            yield error


def type_draft3(validator, types, instance, schema):
    types = _utils.ensure_list(types)

    all_errors = []
    for index, type in enumerate(types):
        if validator.is_type(type, "object"):
            errors = list(validator.descend(instance, type, schema_path=index))
            if not errors:
                return
            all_errors.extend(errors)
        else:
            if validator.is_type(instance, type):
                return
    else:
        reprs = []
        for type in types:
            try:
                reprs.append(repr(type["name"]))
            except Exception:
                reprs.append(repr(type))
        yield ValidationError(
            f"{instance!r} is not of type {', '.join(reprs)}",
            context=all_errors,
        )


def contains_draft6_draft7(validator, contains, instance, schema):
    if not validator.is_type(instance, "array"):
        return

    if not any(
        validator.evolve(schema=contains).is_valid(element)
        for element in instance
    ):
        yield ValidationError(
            f"None of {instance!r} are valid under the given schema",
        )


def recursiveRef(validator, recursiveRef, instance, schema):
    lookup_url, target = validator.resolver.resolution_scope, validator.schema

    for each in reversed(validator.resolver._scopes_stack[1:]):
        lookup_url, next_target = validator.resolver.resolve(each)
        if next_target.get("$recursiveAnchor"):
            target = next_target
        else:
            break

    fragment = recursiveRef.lstrip("#")
    subschema = validator.resolver.resolve_fragment(target, fragment)
    # FIXME: This is gutted (and not calling .descend) because it can trigger
    #        recursion errors, so there's a bug here. Re-enable the tests to
    #        see it.
    subschema
    return []


def find_evaluated_item_indexes_by_schema(validator, instance, schema):
    """
    Get all indexes of items that get evaluated under the current schema

    Covers all keywords related to unevaluatedItems: items, prefixItems, if,
    then, else, contains, unevaluatedItems, allOf, oneOf, anyOf
    """
    if validator.is_type(schema, "boolean"):
        return []
    evaluated_indexes = []

    if "additionalItems" in schema:
        return list(range(0, len(instance)))

    if "$ref" in schema:
        scope, resolved = validator.resolver.resolve(schema["$ref"])
        validator.resolver.push_scope(scope)

        try:
            evaluated_indexes += find_evaluated_item_indexes_by_schema(
                validator, instance, resolved,
            )
        finally:
            validator.resolver.pop_scope()

    if "items" in schema:
        if validator.is_type(schema["items"], "object"):
            return list(range(0, len(instance)))
        evaluated_indexes += list(range(0, len(schema["items"])))

    if "if" in schema:
        if validator.evolve(schema=schema["if"]).is_valid(instance):
            evaluated_indexes += find_evaluated_item_indexes_by_schema(
                validator, instance, schema["if"],
            )
            if "then" in schema:
                evaluated_indexes += find_evaluated_item_indexes_by_schema(
                    validator, instance, schema["then"],
                )
        else:
            if "else" in schema:
                evaluated_indexes += find_evaluated_item_indexes_by_schema(
                    validator, instance, schema["else"],
                )

    for keyword in ["contains", "unevaluatedItems"]:
        if keyword in schema:
            for k, v in enumerate(instance):
                if validator.evolve(schema=schema[keyword]).is_valid(v):
                    evaluated_indexes.append(k)

    for keyword in ["allOf", "oneOf", "anyOf"]:
        if keyword in schema:
            for subschema in schema[keyword]:
                errs = list(validator.descend(instance, subschema))
                if not errs:
                    evaluated_indexes += find_evaluated_item_indexes_by_schema(
                        validator, instance, subschema,
                    )

    return evaluated_indexes


def unevaluatedItems_draft2019(validator, unevaluatedItems, instance, schema):
    if not validator.is_type(instance, "array"):
        return
    evaluated_item_indexes = find_evaluated_item_indexes_by_schema(
        validator, instance, schema,
    )
    unevaluated_items = [
        item for index, item in enumerate(instance)
        if index not in evaluated_item_indexes
    ]
    if unevaluated_items:
        error = "Unevaluated items are not allowed (%s %s unexpected)"
        yield ValidationError(error % _utils.extras_msg(unevaluated_items))
