"""
Public JSON Schema definitions for contradish's interchange formats.

Contradish's differentiation from a generic consistency-testing library is
that it's connected to a real theory (Compression-Aware Intelligence) of
*why* models destabilize under compression of unresolved contradiction, not
just *that* they did. The Constraint Support Graph -- the discovered,
causally-probed set of distinctions that support a constraint, and how their
hold rates move across model versions or prompt rewrites -- is the concrete
artifact that theory produces.

That artifact only becomes foundational if its format gets adopted outside
contradish itself. A technique can be absorbed by a well-resourced lab in
the time it takes to read a paper; a data format other tools already emit
or consume is much harder to quietly discard. So the schemas this module
loads are published as versioned, standalone JSON Schema documents under
contradish/schema/ -- readable and vendorable by any tool, in any language,
with no dependency on this package -- rather than treated as a private
internal detail of DistinctionLossMap.to_dict() and
diff_distinction_reports(). See contradish/schema/README.md for the full
rationale and the versioning policy.

Three schemas ship today:

    distinction_report   the shape of `DistinctionLossMap.to_dict()`, i.e.
                          what `contradish distinguish --json` writes: one
                          domain's distinctions, each with hold rates per
                          framing and overall.

    distinction_diff     the shape of `diff_distinction_reports()`, i.e. the
                          `distinction_diff` block `contradish compare`
                          prints: a baseline-vs-candidate diff of a
                          distinction_report, flagging newly_collapsed pairs.

    transition_contract  the shape of `TransitionContract.to_dict()`
                          (transition_derivation.py): one scenario pair's
                          automatically-derived warranted-transition
                          judgment(s) -- the ground-truth spec
                          decision_relevance.py and minimal_intervention_delta.py
                          otherwise require a human to hand-author, published
                          so an independently produced contract for the same
                          pair can be compared to contradish's own.

Usage:
    from contradish.schema import list_schemas, load_schema, validate_against_schema

    list_schemas()                                          # ["distinction_diff", "distinction_report", "transition_contract"]
    load_schema("distinction_report")                       # parsed JSON Schema document (stdlib only)
    validate_against_schema(payload, "distinction_report")   # [] if valid, else error strings
                                                              # (requires the optional jsonschema package)

CLI:
    contradish schema --list
    contradish schema --show distinction_report
    contradish schema --validate my_report.json --against distinction_report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).parent

_SCHEMA_FILES = {
    "distinction_report":   "distinction_report.schema.json",
    "distinction_diff":     "distinction_diff.schema.json",
    "transition_contract":  "transition_contract.schema.json",
}


def list_schemas() -> list[str]:
    """Names of the published schemas, e.g. for `contradish schema --list`."""
    return sorted(_SCHEMA_FILES)


def load_schema(name: str) -> dict[str, Any]:
    """
    Load a published JSON Schema document by name.

    Args:
        name: one of list_schemas(), e.g. "distinction_report".

    Returns:
        The parsed JSON Schema document (a plain dict). Reads the .json
        file fresh every call -- stdlib only, no caching, so a schema file
        edited on disk is picked up immediately.

    Raises:
        KeyError: if `name` isn't a published schema.
    """
    if name not in _SCHEMA_FILES:
        raise KeyError(
            f"no published schema named {name!r}. Available: {', '.join(list_schemas())}"
        )
    path = SCHEMA_DIR / _SCHEMA_FILES[name]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_against_schema(data: dict, name: str) -> list[str]:
    """
    Validate `data` against the published schema `name`.

    Requires the optional `jsonschema` package (pip install jsonschema, or
    `pip install "contradish[schema]"`). This is intentionally not a hard
    dependency of contradish itself -- see pyproject.toml's "no hard
    dependencies" policy -- since most callers only need to read or write
    the format, not validate a payload against it, and load_schema() alone
    (stdlib only) covers that.

    Args:
        data: the JSON-shaped dict to validate, e.g. a loaded
              `contradish distinguish --json` report.
        name: one of list_schemas(), e.g. "distinction_report".

    Returns:
        A list of human-readable "path: message" validation error strings,
        one per schema violation, sorted by path. Empty list means `data`
        is valid against the schema.

    Raises:
        KeyError: if `name` isn't a published schema.
        ImportError: if the jsonschema package isn't installed.
    """
    try:
        import jsonschema
    except ImportError as e:
        raise ImportError(
            "jsonschema is required to validate against a contradish schema. "
            "Install with:\n    pip install \"contradish[schema]\""
        ) from e

    schema = load_schema(name)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.path)))
    return [
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
        for e in errors
    ]
