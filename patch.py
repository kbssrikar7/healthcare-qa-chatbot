import re

with open("venv/lib/python3.14/site-packages/pydantic/v1/schema.py", "r") as f:
    code = f.read()

# Replace the block that raises ValueError with a simple return annotation
orig = """    if used_constraints and set(constraints.keys()) ^ used_constraints:
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

new = """    return annotation"""

if orig in code:
    code = code.replace(orig, new)
    with open("venv/lib/python3.14/site-packages/pydantic/v1/schema.py", "w") as f:
        f.write(code)
    print("Patched successfully")
else:
    print("Could not find block")
