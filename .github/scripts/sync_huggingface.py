"""
Syncs the CAI-Bench dataset and results to HuggingFace.

Uploads:
  - contradish/benchmarks/v1/*.json   (frozen question set)
  - results/*.json                    (submitted model results)
  - BENCHMARK.md                      (methodology -- becomes the dataset card)
  - CITATION.bib

Run automatically by .github/workflows/sync-huggingface.yml on push to main.
Requires HF_TOKEN and HF_DATASET_REPO environment variables.
"""

import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, CommitOperationAdd
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = os.environ.get("HF_DATASET_REPO", "").strip()

    if not token:
        print("HF_TOKEN not set. Skipping HuggingFace sync.")
        sys.exit(1)
    if not repo_id:
        print("HF_DATASET_REPO not set. Skipping HuggingFace sync.")
        sys.exit(1)

    root = Path(__file__).parent.parent.parent
    api = HfApi(token=token)

    operations = []

    # Frozen benchmark question sets
    benchmark_dir = root / "contradish" / "benchmarks" / "v1"
    for json_file in sorted(benchmark_dir.glob("*.json")):
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"v1/{json_file.name}",
                path_or_fileobj=json_file,
            )
        )
        print(f"  queuing: v1/{json_file.name}")

    # Submitted model results
    results_dir = root / "results"
    for json_file in sorted(results_dir.glob("*.json")):
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"results/{json_file.name}",
                path_or_fileobj=json_file,
            )
        )
        print(f"  queuing: results/{json_file.name}")

    # Dataset card (BENCHMARK.md becomes README.md on HuggingFace)
    benchmark_md = root / "BENCHMARK.md"
    if benchmark_md.exists():
        # Prepend HuggingFace dataset card header
        content = benchmark_md.read_text()
        card = "---\ntask_categories:\n- text-generation\nlanguage:\n- en\ntags:\n- llm\n- evaluation\n- consistency\n- benchmark\nlicense: mit\n---\n\n" + content
        operations.append(
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=card.encode(),
            )
        )
        print("  queuing: README.md (from BENCHMARK.md)")

    # Citation file
    citation = root / "CITATION.bib"
    if citation.exists():
        operations.append(
            CommitOperationAdd(
                path_in_repo="CITATION.bib",
                path_or_fileobj=citation,
            )
        )
        print("  queuing: CITATION.bib")

    # Build a leaderboard summary JSON for easy consumption
    leaderboard = build_leaderboard_summary(results_dir)
    leaderboard_bytes = json.dumps(leaderboard, indent=2).encode()
    operations.append(
        CommitOperationAdd(
            path_in_repo="leaderboard.json",
            path_or_fileobj=leaderboard_bytes,
        )
    )
    print("  queuing: leaderboard.json")

    if not operations:
        print("Nothing to sync.")
        return

    print(f"\nPushing {len(operations)} file(s) to {repo_id} ...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="sync: update CAI-Bench dataset and results",
    )
    print(f"Done. https://huggingface.co/datasets/{repo_id}")


def build_leaderboard_summary(results_dir: Path) -> list:
    """Build a sorted leaderboard from all result JSON files."""
    entries = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if data.get("avg_cai_strain") is None:
                continue
            entries.append({
                "model":               data["model"],
                "provider":            data["provider"],
                "avg_cai_score":       data["avg_cai_score"],
                "avg_cai_strain":      data["avg_cai_strain"],
                "benchmark_version":   data.get("benchmark_version", "unknown"),
                "independent_judging": data.get("independent_judging", False),
                "judge_provider":      data.get("judge_provider"),
                "judge_model":         data.get("judge_model"),
                "date":                data["date"],
                "policies_tested":     data.get("policies_tested", []),
            })
        except Exception as e:
            print(f"  skipping {path.name}: {e}")

    entries.sort(key=lambda x: x["avg_cai_strain"])
    return entries


if __name__ == "__main__":
    main()
