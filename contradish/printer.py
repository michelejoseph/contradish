"""
Terminal output — zero interpretation required.
Every line should be immediately understood by any developer.
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


def _wrap(text: str, width: int = 70, indent: str = "") -> str:
    return textwrap.fill(text.strip(), width=width,
                         initial_indent=indent,
                         subsequent_indent=indent)


def print_progress(msg: str):
    print(f"  {_GRAY}{msg}{_RESET}")


def print_step(step: str, test_name: str, current: int, total: int):
    print(f"\n{_BOLD}[{current}/{total}]{_RESET} {_GRAY}{test_name}{_RESET}")


def print_report(report) -> None:
    total  = len(report.results)
    passed = len(report.passed)
    failed = len(report.failed)

    print()

    # ── Opening line — immediately tells them if they have a problem ──
    if failed == 0:
        print(f"{_GREEN}{_BOLD}contradish found no bugs.{_RESET}  "
              f"{_GRAY}All {total} rules tested clean.{_RESET}")
        print()
        for result in report.results:
            print(f"  {_GREEN}✓{_RESET}  {result.test_case.name}  "
                  f"{_GRAY}consistent across all phrasings{_RESET}")
        print()
        return

    bug_word = "bug" if failed == 1 else "bugs"
    print(f"{_RED}{_BOLD}contradish found {failed} {bug_word} in your app.{_RESET}")
    print()

    # ── Each failing result ────────────────────────────────────────────
    for result in report.results:
        tc = result.test_case
        ok = result.passed(report.thresholds)

        if ok:
            print(f"  {_GREEN}✓{_RESET}  {_GRAY}{tc.name} — clean{_RESET}")
            print()
            continue

        # Header
        print(f"{_RED}{_BOLD}YOUR APP CONTRADICTS ITSELF ON:{_RESET}")
        print(f'{_BOLD}"{tc.name}"{_RESET}')
        print()

        # Show the contradiction as a user scenario
        if result.contradictions:
            pair = result.contradictions[0]

            q_a = pair.input_a.strip()
            a_a = pair.output_a.strip()
            q_b = pair.input_b.strip()
            a_b = pair.output_b.strip()

            # Wrap long outputs
            if len(a_a) > 100:
                a_a = a_a[:97] + "..."
            if len(a_b) > 100:
                a_b = a_b[:97] + "..."

            print(f"  {_GRAY}A user asked:{_RESET}   \"{q_a}\"")
            print(f"  {_GRAY}Your app said:{_RESET}  \"{a_a}\"")
            print()
            print(f"  {_GRAY}Same user, different wording:{_RESET}  \"{q_b}\"")
            print(f"  {_GRAY}Your app said:{_RESET}  \"{a_b}\"")
            print()

            print(f"{_YELLOW}{_BOLD}These answers cannot both be right."
                  f" One will reach a real user.{_RESET}")
            print()

            # More contradictions if present
            if len(result.contradictions) > 1:
                extra = len(result.contradictions) - 1
                print(f"  {_GRAY}+ {extra} more contradiction{'s' if extra > 1 else ''} "
                      f"detected on this rule.{_RESET}")
                print()

        elif result.unstable_patterns:
            print(f"{_YELLOW}{_BOLD}Your app gives inconsistent answers "
                  f"to the same question.{_RESET}")
            print()

        # Pattern
        if result.unstable_patterns:
            pat = result.unstable_patterns[0]
            print(f"  {_YELLOW}WHY:{_RESET}  {_wrap(pat, width=66, indent='        ').lstrip()}")
            print()

        # Fix — the most important part
        if result.suggestion:
            print(f"  {_CYAN}{_BOLD}THE FIX:{_RESET}")
            lines = _wrap(result.suggestion, width=66, indent="  ").split("\n")
            for line in lines:
                print(f"  {_CYAN}{line.strip()}{_RESET}")
            print()

        print(f"{_GRAY}{'─' * 60}{_RESET}")
        print()

    # ── Summary ───────────────────────────────────────────────────────
    clean_list = [r for r in report.results if r.passed(report.thresholds)]
    if clean_list:
        for r in clean_list:
            print(f"  {_GREEN}✓{_RESET}  {_GRAY}{r.test_case.name} — clean{_RESET}")
        print()

    print(f"{_RED}{_BOLD}{failed} bug{'s' if failed > 1 else ''} found.{_RESET}  "
          f"{_GRAY}{passed} rule{'s' if passed != 1 else ''} clean.{_RESET}")
    print()
