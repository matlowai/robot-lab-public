"""JSON Schemas for every record that crosses a module boundary, plus a small validation helper."""

import json
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parent


@cache
def validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate(name: str, doc: dict) -> None:
    """Raise jsonschema.ValidationError if doc does not match schemas/<name>.schema.json."""
    validator(name).validate(doc)
