"""
Integration exporters for contradish.

Push CAI results into Langfuse or Arize Phoenix. Feeds your stack, doesn't replace it.

Usage:
    from contradish.exporters import to_langfuse, to_phoenix

    # Langfuse
    from langfuse import Langfuse
    to_langfuse(report, Langfuse(), dataset_name="cai-ecommerce")

    # Phoenix
    to_phoenix(report, dataset_name="cai-ecommerce")

Each item carries the contradiction pair, CAI score, severity, unstable patterns,
and suggested fix. Passing results go too so you have a regression baseline.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import Report


def to_langfuse(
    report: "Report",
    client,
    dataset_name: str = "contradish-cai",
    *,
    include_passing: bool = True,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Push a contradish Report into a Langfuse dataset.

    Failures become dataset items. Passing results included by default
    for regression baseline.

    Args:
        report:          A contradish Report (from suite.run()).
        client:          An authenticated langfuse.Langfuse() instance.
        dataset_name:    Name of the Langfuse dataset to write to (created if missing).
        include_passing: Also export passing results (tagged passing=True).
        metadata:        Optional extra key/values merged into each item's metadata.

    Returns:
        Dict with keys: dataset_name, items_created, failures_exported, passing_exported.

    Raises:
        ImportError: If the langfuse package is not installed.

    Example:
        from langfuse import Langfuse
        from contradish.exporters import to_langfuse

        client = Langfuse()
        to_langfuse(report, client, dataset_name="cai-ecommerce-v2")
    """
    try:
        client.get_dataset  # lightweight attribute probe (works for v2 and v3)
    except AttributeError:
        raise TypeError(
            "client must be a langfuse.Langfuse instance. "
            "Install with: pip install langfuse"
        )

    # Create dataset if it doesn't exist
    try:
        client.get_dataset(dataset_name)
    except Exception:
        client.create_dataset(dataset_name)

    extra_meta = metadata or {}
    failures_exported = 0
    passing_exported  = 0

    # Export failures
    for result in report.failed:
        for pair in result.contradictions or []:
            item_meta = {
                "rule":             result.test_case.name,
                "cai_score":        result.cai_score,
                "severity":         pair.severity,
                "unstable_patterns": result.unstable_patterns,
                "suggested_fix":    result.suggestion,
                "passing":          False,
                **extra_meta,
            }
            client.create_dataset_item(
                dataset_name=dataset_name,
                input={
                    "input_a":   pair.input_a,
                    "input_b":   pair.input_b,
                },
                expected_output={
                    "should_be_consistent": True,
                },
                metadata=item_meta,
            )
            failures_exported += 1

        # If no contradiction pairs, still export the rule as a failing item
        if not result.contradictions:
            item_meta = {
                "rule":             result.test_case.name,
                "cai_score":        result.cai_score,
                "unstable_patterns": result.unstable_patterns,
                "suggested_fix":    result.suggestion,
                "passing":          False,
                **extra_meta,
            }
            client.create_dataset_item(
                dataset_name=dataset_name,
                input={"input": result.test_case.input},
                metadata=item_meta,
            )
            failures_exported += 1

    # Export passing results
    if include_passing:
        for result in report.passed:
            item_meta = {
                "rule":      result.test_case.name,
                "cai_score": result.cai_score,
                "passing":   True,
                **extra_meta,
            }
            client.create_dataset_item(
                dataset_name=dataset_name,
                input={"input": result.test_case.input},
                metadata=item_meta,
            )
            passing_exported += 1

    return {
        "dataset_name":     dataset_name,
        "items_created":    failures_exported + passing_exported,
        "failures_exported": failures_exported,
        "passing_exported":  passing_exported,
    }


def to_phoenix(
    report: "Report",
    dataset_name: str = "contradish-cai",
    *,
    client=None,
    include_passing: bool = True,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Push a contradish Report into an Arize Phoenix dataset.

    Requires arize-phoenix >= 4.0.

    Args:
        report:          A contradish Report (from suite.run()).
        dataset_name:    Name of the Phoenix dataset to write to.
        client:          Optional phoenix.Client() instance. If None, uses
                         the default px.Client() from environment.
        include_passing: Also export passing results.
        metadata:        Optional extra key/values merged into each item's metadata.

    Returns:
        Dict with keys: dataset_name, items_created, failures_exported, passing_exported.

    Raises:
        ImportError: If arize-phoenix is not installed.

    Example:
        import phoenix as px
        from contradish.exporters import to_phoenix

        to_phoenix(report, dataset_name="cai-ecommerce")
    """
    try:
        import phoenix as px  # type: ignore
    except ImportError:
        raise ImportError(
            "arize-phoenix is not installed. "
            "Install with: pip install arize-phoenix"
        )

    _client = client or px.Client()
    extra_meta = metadata or {}

    examples = []

    # Failures
    for result in report.failed:
        for pair in result.contradictions or []:
            examples.append({
                "input": {
                    "input_a": pair.input_a,
                    "input_b": pair.input_b,
                },
                "output": {
                    "output_a": pair.output_a,
                    "output_b": pair.output_b,
                },
                "metadata": {
                    "rule":             result.test_case.name,
                    "cai_score":        result.cai_score,
                    "severity":         pair.severity,
                    "explanation":      pair.explanation,
                    "unstable_patterns": result.unstable_patterns,
                    "suggested_fix":    result.suggestion,
                    "passing":          False,
                    **extra_meta,
                },
            })

        if not result.contradictions:
            examples.append({
                "input":    {"input": result.test_case.input},
                "output":   {},
                "metadata": {
                    "rule":             result.test_case.name,
                    "cai_score":        result.cai_score,
                    "unstable_patterns": result.unstable_patterns,
                    "suggested_fix":    result.suggestion,
                    "passing":          False,
                    **extra_meta,
                },
            })

    # Passing
    if include_passing:
        for result in report.passed:
            examples.append({
                "input":    {"input": result.test_case.input},
                "output":   {},
                "metadata": {
                    "rule":      result.test_case.name,
                    "cai_score": result.cai_score,
                    "passing":   True,
                    **extra_meta,
                },
            })

    dataset = _client.upload_dataset(
        dataset_name=dataset_name,
        inputs=[e["input"]    for e in examples],
        outputs=[e["output"]   for e in examples],
        metadata=[e["metadata"] for e in examples],
    )

    failures_exported = sum(1 for e in examples if not e["metadata"].get("passing"))
    passing_exported  = sum(1 for e in examples if     e["metadata"].get("passing"))

    return {
        "dataset_name":      dataset_name,
        "items_created":     len(examples),
        "failures_exported": failures_exported,
        "passing_exported":  passing_exported,
        "dataset":           dataset,
    }
