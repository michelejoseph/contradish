# Contradish Constraint Support Graph schemas

This directory publishes contradish's distinction-measurement formats as
standalone JSON Schema documents (draft 2020-12): plain files, no
dependency on the `contradish` package, readable and vendorable by any
tool in any language.

## Why this exists

Contradish's differentiation is that it's connected to a real theory
(Compression-Aware Intelligence) of *why* models destabilize under
compression of unresolved contradiction, not just *that* they did. The
Constraint Support Graph, the discovered set of distinctions that support
a constraint and how their hold rates move across model versions or
prompt rewrites, is the concrete artifact that theory produces.

A technique can be absorbed by a well-resourced lab in the time it takes
to read a paper. A data format other tools already emit or consume is
much harder to quietly discard. Publishing this format on its own, versioned
and separate from any particular implementation, is what gives contradish
a path to being adopted as shared infrastructure rather than absorbed as
a feature inside someone else's internal tool.

## What's published

| Schema | File | Produced by | What it describes |
|---|---|---|---|
| `distinction_report` | `distinction_report.schema.json` | `DistinctionLossMap.to_dict()`, `contradish distinguish --json` | One domain's Constraint Support Graph as measured against a model: which distinctions were probed and how well each survived pressure framing. |
| `distinction_diff` | `distinction_diff.schema.json` | `diff_distinction_reports()`, `contradish compare --distinctions` / `--baseline-distinctions`/`--candidate-distinctions` | A baseline-vs-candidate diff of two distinction reports, flagging distinctions that newly collapsed. |
| `transition_contract` | `transition_contract.schema.json` | `TransitionContract.to_dict()` (`transition_derivation.py`) | One scenario pair's automatically-derived warranted-transition judgment(s): which commitments are warranted to change, the correct new content, and confidence -- the spec `decision_relevance.py`/`minimal_intervention_delta.py` otherwise require a human to hand-author. |

Each document is self-contained (its own `$id`, `$defs`, and `additionalProperties: true`
for forward compatibility) and can be fetched, vendored, or referenced
independently of the others.

## Versioning

Every payload carries a `schema_version` field as `"<major>.<minor>"`.

- A **minor** version bump only ever adds a new optional field. Anything
  written against an earlier minor version of the same major version
  keeps parsing correctly.
- A **major** version bump is the only kind of change allowed to remove
  or repurpose an existing field.

All three schemas ship today at `1.0`.

## Using these schemas

From Python, `contradish.schema` loads and validates against these files
without you needing to read them off disk yourself:

```python
from contradish.schema import list_schemas, load_schema, validate_against_schema

list_schemas()                       # ["distinction_diff", "distinction_report", "transition_contract"]
load_schema("distinction_report")    # the parsed JSON Schema document
validate_against_schema(payload, "distinction_report")   # [] if valid, else error strings
```

`validate_against_schema` needs the optional `jsonschema` package
(`pip install "contradish[schema]"`); `load_schema` and `list_schemas` are
stdlib-only.

From the CLI:

```
contradish schema --list
contradish schema --show distinction_report
contradish schema --validate my_report.json --against distinction_report
```

From any other language or tool: read the `.schema.json` files directly.
They're standard JSON Schema draft 2020-12 with no contradish-specific
extensions, so any conformant validator works.
