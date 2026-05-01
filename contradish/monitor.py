"""
contradish monitor — production drift detection.

Where the benchmark runs synthetic adversarial cases, the monitor runs on
your actual production traffic and finds the consistency failures that are
already happening — across real users, asking the same thing in different ways.

A model that passes the benchmark can still drift in production. The monitor
finds it before your users notice it or your lawyers do.

Usage:
    from contradish.monitor import load_log, find_clusters, score_clusters, analyze_monitor

    conversations = load_log("production_logs.jsonl")
    clusters      = find_clusters(conversations, judge, min_cluster_size=3)
    scored        = score_clusters(clusters, judge)
    analysis      = analyze_monitor(scored, total_conversations=len(conversations))

Log format — JSONL, one conversation per line:
    {"input": "user message", "output": "model response"}

Optional fields per line:
    timestamp, session_id, model, domain (used for grouping in reports)
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional


# ─────────────────────────────────────────────────────────
# Log loading
# ─────────────────────────────────────────────────────────

def load_log(
    path: str,
    max_conversations: int = 200,
    format: str = "auto",
) -> list[dict]:
    """
    Load a production conversation log.

    Supported formats:
      jsonl — one {"input": ..., "output": ...} dict per line  (default)
      json  — JSON array of conversation dicts
      csv   — CSV with 'input' and 'output' columns

    Returns a list of dicts. Each dict has at minimum 'input' and 'output'.
    Any extra fields (timestamp, session_id, model, domain) are preserved.

    Raises ValueError on unrecognised format or missing required fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    # Auto-detect format from extension
    if format == "auto":
        ext = p.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            format = "jsonl"
        elif ext == ".json":
            format = "json"
        elif ext in (".csv", ".tsv"):
            format = "csv"
        else:
            # Try JSONL first
            format = "jsonl"

    conversations = []

    if format == "jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        conversations.append(obj)
                except json.JSONDecodeError:
                    continue

    elif format == "json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            conversations = [d for d in data if isinstance(d, dict)]
        elif isinstance(data, dict):
            # Maybe it's wrapped: {"conversations": [...]}
            for key in ("conversations", "data", "results", "logs"):
                if isinstance(data.get(key), list):
                    conversations = data[key]
                    break

    elif format == "csv":
        with open(p, encoding="utf-8", newline="") as f:
            dialect = "excel-tab" if p.suffix.lower() == ".tsv" else "excel"
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                conversations.append(dict(row))

    else:
        raise ValueError(f"Unsupported format: {format}. Use jsonl, json, or csv.")

    # Validate required fields
    valid = []
    for conv in conversations:
        inp = conv.get("input") or conv.get("question") or conv.get("prompt") or conv.get("user")
        out = conv.get("output") or conv.get("response") or conv.get("answer") or conv.get("assistant")
        if inp and out:
            conv["input"]  = str(inp).strip()
            conv["output"] = str(out).strip()
            valid.append(conv)

    if not valid:
        raise ValueError(
            f"No valid conversations found in {path}. "
            "Each entry must have 'input' and 'output' fields."
        )

    # Cap at max_conversations
    if len(valid) > max_conversations:
        valid = valid[:max_conversations]

    return valid


# ─────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────

def find_clusters(
    conversations: list[dict],
    judge,
    min_cluster_size: int = 3,
    batch_size: int = 25,
    quiet: bool = False,
) -> list[dict]:
    """
    Group semantically equivalent conversations into clusters using the LLM judge.

    Processes conversations in batches of batch_size. Within each batch, the judge
    identifies groups of inputs that are asking the same thing in different ways.

    Args:
        conversations:    loaded conversation dicts (each has 'input' and 'output')
        judge:            Judge instance (from contradish.judge)
        min_cluster_size: minimum number of conversations for a cluster to be scored
        batch_size:       inputs per clustering batch (keep ≤ 30 for reliable JSON)
        quiet:            suppress progress output

    Returns:
        list of cluster dicts:
          {
            "topic":         str,
            "conversations": list[dict],   # full conversation dicts
            "size":          int,
          }
        Only clusters with size >= min_cluster_size are returned.
    """
    if not conversations:
        return []

    all_clusters: list[dict] = []  # {topic, conversations}

    # Process in batches
    for batch_start in range(0, len(conversations), batch_size):
        batch = conversations[batch_start:batch_start + batch_size]
        inputs = [c["input"] for c in batch]

        if not quiet:
            end = min(batch_start + batch_size, len(conversations))
            print(f"  clustering [{batch_start+1}-{end}/{len(conversations)}]...")

        result = judge.cluster_inputs(inputs)

        for cluster_spec in result.get("clusters", []):
            indices = cluster_spec.get("input_indices", [])
            topic   = cluster_spec.get("topic", "unknown topic")
            convs   = [batch[i] for i in indices if 0 <= i < len(batch)]
            if len(convs) >= 2:
                # Try to merge with an existing cluster on the same topic
                merged = False
                for existing in all_clusters:
                    if _topics_match(existing["topic"], topic):
                        existing["conversations"].extend(convs)
                        merged = True
                        break
                if not merged:
                    all_clusters.append({"topic": topic, "conversations": convs})

    # Filter by min_cluster_size and add size
    result_clusters = []
    for c in all_clusters:
        c["size"] = len(c["conversations"])
        if c["size"] >= min_cluster_size:
            result_clusters.append(c)

    # Sort largest first
    result_clusters.sort(key=lambda c: -c["size"])
    return result_clusters


def _topics_match(a: str, b: str) -> bool:
    """
    Simple heuristic: two topic strings match if they share enough words.
    Used to merge clusters across batches.
    """
    def words(s):
        return set(s.lower().split())
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    overlap = wa & wb
    # Remove stopwords
    stopwords = {"the", "a", "an", "of", "for", "to", "in", "on", "about", "with", "how", "what", "when"}
    overlap -= stopwords
    if not overlap:
        return False
    # Match if overlap covers at least half of the shorter topic
    shorter = min(len(wa), len(wb))
    return len(overlap) / shorter >= 0.5


# ─────────────────────────────────────────────────────────
# Consistency scoring
# ─────────────────────────────────────────────────────────

def score_clusters(
    clusters: list[dict],
    judge,
    quiet: bool = False,
) -> list[dict]:
    """
    Score consistency within each cluster.

    For each cluster, treats the first conversation as the canonical reference
    and scores all others against it using the judge's evaluate_consistency method.

    Adds to each cluster dict:
      consistency_score       float  (1.0 = all consistent, 0.0 = fully inconsistent)
      cts                     float  (1 - consistency_score; matches CTS convention)
      drifted                 bool   (cts > 0.3)
      per_conversation_scores list[float]
      disagreements           list[str]
      summary                 str
      example_consistent      str | None   (a consistent output, for the report)
      example_drifted         str | None   (a drifted output, for the report)
    """
    scored = []

    for i, cluster in enumerate(clusters):
        convs = cluster["conversations"]
        topic = cluster["topic"]

        if not quiet:
            print(f"  [{i+1}/{len(clusters)}] scoring '{topic}'  (n={len(convs)})")

        inputs  = [c["input"]  for c in convs]
        outputs = [c["output"] for c in convs]

        # Use first input as the "canonical question" for the judge
        canonical_question = inputs[0]

        result = judge.evaluate_consistency(
            question=canonical_question,
            inputs=inputs,
            outputs=outputs,
        )

        consistency_score = result.get("consistency_score", 0.5)
        cts               = round(1.0 - consistency_score, 4)
        drifted           = cts > 0.3

        # Find an example of a consistent vs drifted output
        per_scores = result.get("per_variant_scores", [])
        example_consistent = None
        example_drifted    = None

        if per_scores:
            # per_variant_scores covers indices 1..n-1 (index 0 is canonical)
            best_idx  = max(range(len(per_scores)), key=lambda j: per_scores[j])
            worst_idx = min(range(len(per_scores)), key=lambda j: per_scores[j])
            example_consistent = outputs[best_idx + 1][:300] if per_scores[best_idx] >= 0.7 else None
            example_drifted    = outputs[worst_idx + 1][:300] if per_scores[worst_idx] < 0.5 else None

        scored.append({
            **cluster,
            "consistency_score":        consistency_score,
            "cts":                      cts,
            "drifted":                  drifted,
            "per_conversation_scores":  per_scores,
            "disagreements":            result.get("disagreements", []),
            "summary":                  result.get("summary", ""),
            "example_input_canonical":  inputs[0][:300],
            "example_output_canonical": outputs[0][:300],
            "example_consistent":       example_consistent,
            "example_drifted":          example_drifted,
        })

    return scored


# ─────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────

def analyze_monitor(
    scored_clusters: list[dict],
    total_conversations: int,
) -> dict:
    """
    Aggregate scored clusters into the full monitor report.

    Returns:
      total_conversations     int
      clusters_found          int
      clusters_scored         int
      drifted_clusters        int
      drift_rate              float   (drifted / scored)
      avg_cts                 float
      hotspots                list    (drifted clusters sorted by CTS descending)
      clean_clusters          list    (consistent clusters sorted by CTS ascending)
      domain_breakdown        dict    (domain → drift_count if domain field present)
    """
    if not scored_clusters:
        return {
            "total_conversations": total_conversations,
            "clusters_found":      0,
            "clusters_scored":     0,
            "drifted_clusters":    0,
            "drift_rate":          0.0,
            "avg_cts":             0.0,
            "hotspots":            [],
            "clean_clusters":      [],
            "domain_breakdown":    {},
        }

    scored  = scored_clusters
    n       = len(scored)
    drifted = [c for c in scored if c["drifted"]]
    clean   = [c for c in scored if not c["drifted"]]

    drift_rate = round(len(drifted) / n, 4) if n > 0 else 0.0
    avg_cts    = round(sum(c["cts"] for c in scored) / n, 4) if n > 0 else 0.0

    # Sort hotspots by CTS descending (worst first)
    hotspots = sorted(drifted, key=lambda c: -c["cts"])
    clean    = sorted(clean,   key=lambda c:  c["cts"])

    # Domain breakdown: if conversations have a 'domain' field
    domain_counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "drifted": 0})
    for cluster in scored:
        for conv in cluster.get("conversations", []):
            domain = conv.get("domain", "unknown")
            domain_counts[domain]["total"] += 1
            if cluster["drifted"]:
                domain_counts[domain]["drifted"] += 1

    domain_breakdown = {
        d: {
            "total":      v["total"],
            "drifted":    v["drifted"],
            "drift_rate": round(v["drifted"] / v["total"], 3) if v["total"] > 0 else 0.0,
        }
        for d, v in sorted(domain_counts.items(), key=lambda x: -x[1]["drifted"])
    }

    def _slim(c: dict) -> dict:
        """Return a cluster dict without full conversation lists (for export)."""
        return {
            "topic":                  c["topic"],
            "size":                   c["size"],
            "cts":                    c["cts"],
            "consistency_score":      c["consistency_score"],
            "drifted":                c["drifted"],
            "summary":                c["summary"],
            "disagreements":          c["disagreements"][:3],
            "example_input":          c.get("example_input_canonical", ""),
            "example_output":         c.get("example_output_canonical", ""),
            "example_drifted":        c.get("example_drifted"),
            "example_consistent":     c.get("example_consistent"),
        }

    return {
        "total_conversations": total_conversations,
        "clusters_found":      n,
        "clusters_scored":     n,
        "drifted_clusters":    len(drifted),
        "drift_rate":          drift_rate,
        "avg_cts":             avg_cts,
        "hotspots":            [_slim(c) for c in hotspots],
        "clean_clusters":      [_slim(c) for c in clean],
        "domain_breakdown":    domain_breakdown,
    }


# ─────────────────────────────────────────────────────────
# Terminal output
# ─────────────────────────────────────────────────────────

_TTY = __import__("sys").stdout.isatty()
def _c(code, t): return f"\033[{code}m{t}\033[0m" if _TTY else t
RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
GREEN  = lambda t: _c("32", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


def print_monitor_summary(analysis: dict, log_path: str) -> None:
    """Print the production drift detection report to stdout."""
    total   = analysis["total_conversations"]
    n       = analysis["clusters_scored"]
    drifted = analysis["drifted_clusters"]
    rate    = analysis["drift_rate"]
    avg_cts = analysis["avg_cts"]

    rate_color = GREEN if rate < 0.15 else (YELLOW if rate < 0.35 else RED)
    cts_color  = GREEN if avg_cts < 0.20 else (YELLOW if avg_cts < 0.40 else RED)

    print()
    print(f"  {BOLD('contradish monitor')}  {DIM('production drift detection')}")
    print(f"  source:    {log_path}")
    print(f"  total:     {total} conversations  →  {n} topic clusters")
    print()
    print(f"  drift rate:  {rate_color(f'{rate:.0%}')}  ({drifted}/{n} clusters drifting)")
    print(f"  avg CTS:     {cts_color(f'{avg_cts:.2f}')}")
    print()

    # Hotspots
    hotspots = analysis.get("hotspots", [])
    if hotspots:
        print(f"  {BOLD(RED('DRIFT HOTSPOTS'))}  {DIM('topics where your model is inconsistent across users')}")
        for h in hotspots[:5]:
            cts_val = h["cts"]
            c = RED if cts_val >= 0.5 else YELLOW
            size_str = f"n={h['size']}"
            print(f"  {c(f'CTS {cts_val:.2f}')}  {h['topic']}  {DIM(size_str)}")
            if h.get("summary"):
                print(f"         {DIM(h['summary'])}")
            if h.get("example_drifted"):
                drifted_ex = h["example_drifted"][:120].replace("\n", " ")
                print(f"         {DIM('drifted: ')} {drifted_ex}")
            if h.get("disagreements"):
                for d in h["disagreements"][:1]:
                    print(f"         {DIM('issue:   ')} {d[:120]}")
        print()

    # Clean clusters
    clean = analysis.get("clean_clusters", [])
    if clean:
        print(f"  {BOLD(GREEN('CONSISTENT'))}  {DIM('topics where your model is holding across users')}")
        for c in clean[:3]:
            cts_s = f"CTS {c['cts']:.2f}"
            n_s   = f"n={c['size']}"
            print(f"  {GREEN(cts_s)}  {c['topic']}  {DIM(n_s)}")
        print()

    # Domain breakdown
    domain = analysis.get("domain_breakdown", {})
    if domain and len(domain) > 1:
        print(f"  {BOLD('BY DOMAIN')}")
        for d, v in list(domain.items())[:5]:
            dr = v["drift_rate"]
            col = RED if dr >= 0.5 else (YELLOW if dr >= 0.2 else GREEN)
            conv_s = f"{v['drifted']}/{v['total']} conversations"
            print(f"  {col(f'{dr:.0%} drift')}  {d}  {DIM(conv_s)}")
        print()

    if not hotspots:
        print(f"  {GREEN('No significant drift detected.')}  "
              f"{DIM('Your model is consistent across the sampled conversations.')}")
        print()
