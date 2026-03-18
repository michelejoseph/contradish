"""
Terminal output — rich colors in TTY, plain text in CI.
Auto-detected. No dependencies required.
"""

import sys
import os


def _is_tty() -> bool:
    """True if we're writing to a real terminal (not piped / CI)."""
    ci_vars = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_URL", "BUILDKITE")
    if any(os.environ.get(v) for v in ci_vars):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _is_tty()

# ANSI codes
_RESET  = "\033[0m"  if USE_COLOR else ""
_BOLD   = "\033[1m"  if USE_COLOR else ""
_DIM    = "\033[2m"  if USE_COLOR else ""
_RED    = "\033[91m" if USE_COLOR else ""
_GREEN  = "\033[92m" if USE_COLOR else ""
_YELLOW = "\033[93m" if USE_COLOR else ""
_CYAN   = "\033[96m" if USE_COLOR else ""
_WHITE  = "\033[97m" if USE_COLOR else ""
_GRAY   = "\033[90m" if USE_COLOR else ""


def _risk_color(risk: str) -> str:
    if not USE_COLOR:
        return ""
    return {"low": _GREEN, "medium": _YELLOW, "high": _RED}.get(risk, "")


def _score_color(score: float, invert: bool = False) -> str:
    if not USE_COLOR:
        return ""
    good = score >= 0.80
    if invert:
        good = score <= 0.20
    return _GREEN if good else (_YELLOW if (score >= 0.60 if not invert else score <= 0.40) else _RED)


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar."""
    filled = round(score * width)
    empty  = width - filled
    if USE_COLOR:
        color = _score_color(score)
        return f"{color}{'█' * filled}{'░' * empty}{_RESET}"
    return f"{'#' * filled}{'.' * empty}"


def print_progress(msg: str):
    print(f"  {_GRAY}→{_RESET} {_DIM}{msg}{_RESET}")


def print_report(report) -> None:
    """Print a full Report to stdout."""
    from .models import RiskLevel

    width = 62

    # ── Header ────────────────────────────────────────────────────
    print()
    print(f"{_BOLD}{_CYAN}{'━' * width}{_RESET}")
    print(f"{_BOLD}{_CYAN}  contradish  ·  reasoning stability report{_RESET}")
    print(f"{_BOLD}{_CYAN}{'━' * width}{_RESET}")
    print()

    total  = len(report.results)
    passed = len(report.passed)
    failed = len(report.failed)

    pass_str = f"{_GREEN}{passed} passed{_RESET}" if USE_COLOR else f"{passed} passed"
    fail_str = f"{_RED}{failed} failed{_RESET}"   if (USE_COLOR and failed) else f"{failed} failed"

    print(f"  {_BOLD}Tests:{_RESET} {total}   {pass_str}   {fail_str}")
    print()

    # Aggregate scores
    if report.avg_consistency is not None or report.avg_contradiction is not None:
        print(f"  {_BOLD}Aggregate{_RESET}")
        if report.avg_consistency is not None:
            c = report.avg_consistency
            print(f"    consistency   {_bar(c)}  {_score_color(c)}{c:.2f}{_RESET}")
        if report.avg_contradiction is not None:
            c = report.avg_contradiction
            print(f"    contradiction {_bar(c, 20)[::-1] if not USE_COLOR else _bar(1 - c)}  {_score_color(c, invert=True)}{c:.2f}{_RESET}")
        print()

    # ── Per-test results ───────────────────────────────────────────
    for result in report.results:
        tc   = result.test_case
        risk = result.risk.value
        rc   = _risk_color(risk)
        ok   = result.passed(report.thresholds)
        icon = f"{_GREEN}✓{_RESET}" if (USE_COLOR and ok) else ("✓" if ok else (f"{_RED}✗{_RESET}" if USE_COLOR else "✗"))

        print(f"  {icon}  {_BOLD}{tc.name}{_RESET}  {_DIM}[risk: {rc}{risk}{_RESET}{_DIM}]{_RESET}")

        if result.consistency_score is not None:
            c = result.consistency_score
            print(f"       consistency   {_bar(c)}  {_score_color(c)}{c:.2f}{_RESET}")
        if result.contradiction_score is not None:
            c = result.contradiction_score
            print(f"       contradiction {_bar(1 - c)}  {_score_color(c, invert=True)}{c:.2f}{_RESET}")

        # Contradictions found
        if result.contradictions:
            print()
            sev_color = _RED if USE_COLOR else ""
            print(f"       {sev_color}{_BOLD}Contradictions detected ({len(result.contradictions)}){_RESET}")
            for pair in result.contradictions[:2]:
                print(f"       {_DIM}┌ [{pair.severity}]{_RESET} {pair.explanation}")
                print(f"       {_DIM}│ A:{_RESET} {pair.output_a[:90].strip()}")
                print(f"       {_DIM}│ B:{_RESET} {pair.output_b[:90].strip()}")
                print(f"       {_DIM}└{_RESET}")

        # Unstable patterns
        if result.unstable_patterns:
            print()
            for pat in result.unstable_patterns:
                print(f"       {_YELLOW}⚠{_RESET}  {pat}")

        # Suggestion
        if result.suggestion:
            print()
            print(f"       {_CYAN}→ Fix:{_RESET} {result.suggestion}")

        print()

    # ── Footer ────────────────────────────────────────────────────
    print(f"{_DIM}{'─' * width}{_RESET}")
    if failed:
        print(f"  {_RED}{_BOLD}{failed} test{'s' if failed > 1 else ''} failed.{_RESET}  Reasoning instability detected.")
    else:
        print(f"  {_GREEN}{_BOLD}All tests passed.{_RESET}  No instability detected.")
    print(f"{_DIM}{'─' * width}{_RESET}")
    print()


def print_step(step: str, test_name: str, current: int, total: int):
    """Print a single progress line."""
    prefix = f"[{current}/{total}]"
    print(f"  {_GRAY}{prefix}{_RESET} {_BOLD}{test_name}{_RESET}  {_DIM}{step}{_RESET}")
