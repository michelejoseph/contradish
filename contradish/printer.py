"""
Terminal output — clean, readable, bug-first.
Colors in TTY, plain text in CI. Auto-detected.
"""

import sys
import os
import textwrap


def _is_tty() -> bool:
    ci_vars = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_URL", "BUILDKITE")
    if any(os.environ.get(v) for v in ci_vars):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _is_tty()

_RESET  = "\033[0m"  if USE_COLOR else ""
_BOLD   = "\033[1m"  if USE_COLOR else ""
_DIM    = "\033[2m"  if USE_COLOR else ""
_RED    = "\033[91m" if USE_COLOR else ""
_GREEN  = "\033[92m" if USE_COLOR else ""
_YELLOW = "\033[93m" if USE_COLOR else ""
_CYAN   = "\033[96m" if USE_COLOR else ""
_GRAY   = "\033[90m" if USE_COLOR else ""


def _wrap(text: str, width: int = 68, indent: str = "             ") -> str:
    return textwrap.fill(
        text, width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    ).lstrip()


def print_progress(msg: str):
    print(f"  {_GRAY}{msg}{_RESET}")


def print_step(step: str, test_name: str, current: int, total: int):
    print(f"\n{_BOLD}[{current}/{total}] {test_name}{_RESET}  {_GRAY}{step}{_RESET}")


def print_report(report) -> None:
    """
    Output priority:
      1. What broke — the actual contradiction in plain English
      2. Why — which input pattern caused it
      3. Fix — what to change
      4. Scores — last, for reference only
    """
    total  = len(report.results)
    passed = len(report.passed)
    failed = len(report.failed)

    print()
    print(f"{_BOLD}contradish{_RESET}  {_GRAY}reasoning stability report{_RESET}")
    print(f"{_GRAY}{'─' * 60}{_RESET}")
    print()

    for result in report.results:
        tc = result.test_case
        ok = result.passed(report.thresholds)

        # ── Passing test: one clean line ───────────────────────────
        if ok:
            c = result.consistency_score
            sc = (_GREEN if c >= 0.80 else (_YELLOW if c >= 0.60 else _RED)) if c is not None else _GRAY
            score_str = f"{sc}{c:.0%}{_RESET}" if c is not None else ""
            print(f"  {_GREEN}PASS{_RESET}  {_BOLD}{tc.name}{_RESET}")
            print(f"        {_GRAY}consistent across all paraphrases  {score_str}{_RESET}")
            print()
            continue

        # ── Failing test: lead with the bug ────────────────────────
        risk = result.risk.value
        rc = _RED if risk == "high" else (_YELLOW if risk == "medium" else _GRAY)
        print(f"  {_RED}FAIL{_RESET}  {_BOLD}{tc.name}{_RESET}  "
              f"{_GRAY}[{rc}{risk} risk{_RESET}{_GRAY}]{_RESET}")
        print()

        # 1. The contradiction — show it as a real example, not a score
        if result.contradictions:
            n = len(result.contradictions)
            noun = "contradiction" if n == 1 else "contradictions"
            print(f"  {_RED}Your app gave {n} contradictory {noun}:{_RESET}")
            print()

            for i, pair in enumerate(result.contradictions[:2], 1):
                q_a = pair.input_a.strip()[:80]
                q_b = pair.input_b.strip()[:80]
                a_a = pair.output_a.strip()[:120]
                a_b = pair.output_b.strip()[:120]

                print(f"  {_GRAY}Question:{_RESET}  {q_a}")
                print(f"  {_GRAY}Answer:  {_RESET}  {a_a}")
                print()
                print(f"  {_GRAY}Question:{_RESET}  {q_b}")
                print(f"  {_GRAY}Answer:  {_RESET}  {a_b}")
                print()

                if pair.explanation and pair.explanation.lower() not in ("none", ""):
                    expl = _wrap(pair.explanation, indent="             ")
                    print(f"  {_YELLOW}Conflict:{_RESET}  {expl}")
                    print()

                if i < min(n, 2):
                    print(f"  {_GRAY}{'- ' * 22}{_RESET}")
                    print()

        elif result.unstable_patterns or result.consistency_score is not None:
            # No hard contradictions but still failing — show inconsistency
            print(f"  {_YELLOW}Your app gave inconsistent answers across paraphrases.{_RESET}")
            print()

        # 2. Pattern — what input phrasing caused it
        if result.unstable_patterns:
            for pat in result.unstable_patterns[:1]:
                wrapped = _wrap(pat, indent="             ")
                print(f"  {_YELLOW}Pattern: {_RESET}  {wrapped}")
            print()

        # 3. Fix — one concrete action
        if result.suggestion:
            wrapped = _wrap(result.suggestion, indent="             ")
            print(f"  {_CYAN}Fix:     {_RESET}  {wrapped}")
            print()

        # 4. Scores — reference only, not the headline
        scores = []
        if result.consistency_score is not None:
            c = result.consistency_score
            sc = _GREEN if c >= 0.80 else (_YELLOW if c >= 0.60 else _RED)
            scores.append(f"consistency {sc}{c:.0%}{_RESET}")
        if result.contradiction_score is not None:
            c = result.contradiction_score
            sc = _GREEN if c <= 0.10 else (_YELLOW if c <= 0.30 else _RED)
            scores.append(f"contradiction rate {sc}{c:.0%}{_RESET}")
        if scores:
            print(f"  {_GRAY}Scores:  {'   '.join(scores)}{_RESET}")
            print()

        print(f"{_GRAY}{'─' * 60}{_RESET}")
        print()

    # ── Summary ────────────────────────────────────────────────────
    if failed == 0:
        print(f"  {_GREEN}{_BOLD}All {total} tests passed.{_RESET}  "
              f"{_GRAY}No contradictions detected.{_RESET}")
    else:
        print(f"  {_RED}{_BOLD}{failed} of {total} tests failed.{_RESET}  "
              f"{_GRAY}{passed} passed.{_RESET}")
    print()
