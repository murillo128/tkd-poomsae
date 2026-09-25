"""Regenerate the published JSON Schema from the Python contract types."""

import json
from pathlib import Path

from contracts.models import _adapter

SCHEMA_PATH = Path(__file__).with_name("artifact.schema.json")


def schema_text() -> str:
    schema = _adapter.json_schema(mode="validation")
    schema["$id"] = (
        "https://github.com/murillo128/tkd-poomsae/contracts/artifact.schema.json"
    )
    schema["title"] = "TKD Poomsae artifact 1.0.0"
    return json.dumps(schema, indent=2, sort_keys=True, allow_nan=False) + "\n"


if __name__ == "__main__":
    SCHEMA_PATH.write_text(schema_text())
