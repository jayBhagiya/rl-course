from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def emit_json(payload: dict[str, Any], output: str | Path | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

