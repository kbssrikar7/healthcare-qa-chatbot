import os

fields_path = "venv/lib/python3.14/site-packages/pydantic/v1/fields.py"
with open(fields_path, "r") as f:
    fields_code = f.read()
if "raise errors_.ConfigError" in fields_code:
    fields_code = fields_code.replace(
        "raise errors_.ConfigError(f'unable to infer type for attribute \"{self.name}\"')",
        "self.type_ = __import__('typing').Any"
    )
    with open(fields_path, "w") as f:
        f.write(fields_code)
    print("Patched fields.py")

schema_path = "venv/lib/python3.14/site-packages/pydantic/v1/schema.py"
with open(schema_path, "r") as f:
    schema_code = f.read()

orig_block = """    if used_constraints and set(constraints.keys()) ^ used_constraints:
        unenforced_constraints = {
            k: v for k, v in constraints.items() if k not in used_constraints and v is not None
        }
        if unenforced_constraints:
            raise ValueError(
                f'On field "{field_name}" the following field constraints are set but not enforced: '
                f'{", ".join(unenforced_constraints.keys())}. \\n'
                f'For more details see https://docs.pydantic.dev/usage/schema/#unenforced-field-constraints'
            )

    return annotation"""

new_block = """    return annotation"""

if orig_block in schema_code:
    schema_code = schema_code.replace(orig_block, new_block)
    with open(schema_path, "w") as f:
        f.write(schema_code)
    print("Patched schema.py")
