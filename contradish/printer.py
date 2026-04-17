"""
Terminal output. Zero interpretation required.
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


def _cai_label(score: float) -> str:
    if score >= 0.80:
        return "stable"
    if score >= 0.60:
        return "marginal"
    return "unstable"


def print_start(prompt_preview: str = "") -> None:
    """
    Print a brief orientation line before any work starts.
    Gives the developer something to look at while extraction runs.
    """
    if prompt_preview:
        preview = prompt_preview[:72].strip()
        if len(prompt_preview) > 72:
            preview += "..."
        print(f"\n{_GRAY}  scanning: \"{preview}\"{_RESET}\n")
    else:
        print(f"\n{_GRAY}  running CAI tests{_RESET}\n")


def print_progress(msg: str) -> None:
    print(f"  {_GRAY}{msg}{_RESET}")


def print_step(label: str, test_name: str, current: int, total: int) -> None:
    print(f"\n{_BOLD}[{current}/{total}]{_RESET}  testing \"{test_name}\"")


def print_report(report) -> None:
    total  = len(report.results)
    passed = len(report.passed)
    failed = len(report.failed)

    print()

    # ── All clean ─────────────────────────────────────────────────────
    if failed == 0:
        print(f"{_GREEN}{_BOLD}No CAI failures.{_RESET}  "
              f"{_GRAY}All {total} rule{'s' if total != 1 else ''} stable.{_RESET}")
        print()
        for result in report.results:
            score     = result.consistency_score
            score_str = f"{score:.2f}" if score is not None else "n/a"
            print(f"  {_GREEN}✓{_RESET}  {result.test_case.name}  "
                  f"{_GRAY}CAI score: {score_str}{_RESET}")
        print()
        return

    # ── Header ────────────────────────────────────────────────────────
    fail_word = "CAI failure" if failed == 1 else "CAI failures"
    print(f"{_RED}{_BOLD}contradish found {failed} {fail_word}.{_RESET}")
    print()

    # ── Each failing result ────────────────────────────────────────────
    for result in report.results:
        tc = result.test_case
        ok = result.passed(report.thresholds)

        score     = result.consistency_score
        score_str = f"{score:.2f}" if score is not None else "n/a"

        if ok:
            print(f"  {_GREEN}✓{_RESET}  {_GRAY}{tc.name}  "
                  f"CAI score: {score_str}  (stable){_RESET}")
            print()
            continue

        label = _cai_label(score) if score is not None else "unstable"

        # ── Failure header ─────────────────────────────────────────────
        sev_badge = ""
        if result.contradictions:
            sev = result.contradictions[0].severity
            if sev and sev != "unknown" and sev != "none":
                sev_badge = f"  {_GRAY}[{sev}]{_RESET}"

        print(f"{_RED}{_BOLD}CAI FAILURE{_RESET}  "
              f"\"{tc.name}\"  "
              f"{_GRAY}score {score_str}{_RESET}"
              f"{sev_badge}")
        print()

        # ── Show the exact contradiction ───────────────────────────────
        if result.contradictions:
            pair = result.contradictions[0]

            q_a = pair.input_a.strip()
            a_a = pair.output_a.strip()
            q_b = pair.input_b.strip()
            a_b = pair.output_b.strip()

            if len(a_a) > 110:
                a_a = a_a[:107] + "..."
            if len(a_b) > 110:
                a_b = a_b[:107] + "..."

            print(f"  {_GRAY}asked:{_RESET}  \"{q_a}\"")
            print(f"  {_GRAY}said: {_RESET}  \"{a_a}\"")
            print()
            print(f"  {_GRAY}asked:{_RESET}  \"{q_b}\"")
            print(f"  {_GRAY}said: {_RESET}  \"{a_b}\"")
            print()

            if pair.explanation and pair.explanation.lower() != "none":
                expl = pair.explanation.strip()
                wrapped_expl = _wrap(expl, width=70, indent="  ")
                print(f"{_YELLOW}{wrapped_expl}{_RESET}")
                print()

            if len(result.contradictions) > 1:
                extra = len(result.contradictions) - 1
                print(f"  {_GRAY}+{extra} more contradiction{'s' if extra > 1 else ''} on this rule.{_RESET}")
                print()

        elif result.unstable_patterns:
            print(f"{_YELLOW}Inconsistent answers to the same question.{_RESET}")
            print()

        # ── Pattern / Root cause / Fix ─────────────────────────────────
        patterns = result.unstable_patterns
        if len(patterns) >= 2:
            # patterns[0] = trigger pattern, patterns[1] = root cause
            print(f"  {_YELLOW}PATTERN{_RESET}  "
                  f"{_wrap(patterns[0], width=62, indent='           ').lstrip()}")
            print()
            print(f"  {_YELLOW}WHY    {_RESET}  "
                  f"{_wrap(patterns[1], width=62, indent='           ').lstrip()}")
            print()
        elif len(patterns) == 1:
            print(f"  {_YELLOW}WHY{_RESET}  "
                  f"{_wrap(patterns[0], width=64, indent='       ').lstrip()}")
            print()

        # ── Fix ────────────────────────────────────────────────────────
        if result.suggestion:
            print(f"  {_CYAN}{_BOLD}FIX{_RESET}  "
                  f"{_GRAY}add to your system prompt:{_RESET}")
            print()
            wrapped = _wrap(result.suggestion, width=64, indent="  ")
            for line in wrapped.split("\n"):
                print(f"  {_CYAN}\"{line.strip()}\"{_RESET}")
            print()

        print(f"{_GRAY}{'─' * 60}{_RESET}")
        print()

    # ── Passing rules at the bottom ────────────────────────────────────
    clean_list = [r for r in report.results if r.passed(report.thresholds)]
    if clean_list:
        for r in clean_list:
            score     = r.consistency_score
            score_str = f"{score:.2f}" if score is not None else "n/a"
            print(f"  {_GREEN}✓{_RESET}  {_GRAY}{r.test_case.name}  "
                  f"CAI score: {score_str}  (stable){_RESET}")
        print()

    # ── Summary line ───────────────────────────────────────────────────
    pass_word = "rule" if passed == 1 else "rules"
    print(f"{_RED}{_BOLD}{failed} {fail_word} found.{_RESET}  "
          f"{_GRAY}{passed} {pass_word} clean.{_RESET}")
    print()


def print_next_steps(report) -> None:
    """
    Print after print_report. The artifact a developer takes away:
    - what to do immediately
    - their evals.yaml (ready to save)
    - the CI command
    """
    failed  = len(report.failed)
    results = report.results

    print(f"{_GRAY}{'─' * 60}{_RESET}")
    print()

    if failed > 0:
        fail_word = "failure" if failed == 1 else "failures"
        print(f"  {_BOLD}Next:{_RESET}  apply the fix above, then re-run to see your new score.")
    else:
        print(f"  {_BOLD}Next:{_RESET}  add to CI to catch regressions.")

    print()

    # ── evals.yaml artifact ────────────────────────────────────────────
    print(f"  {_GRAY}Save your test cases as{_RESET} {_BOLD}evals.yaml{_RESET}{_GRAY}:{_RESET}")
    print()
    print(f"  {_CYAN}test_cases:{_RESET}")
    for r in results:
        safe_input = r.test_case.input.replace('"', "'")
        safe_name  = r.test_case.name.replace('"', "'")  if r.test_case.name else safe_input[:30]
        print(f"  {_CYAN}  - name:  \"{safe_name}\"{_RESET}")
        print(f"  {_CYAN}    input: \"{safe_input}\"{_RESET}")
    print()

    # ── CI commands ────────────────────────────────────────────────────
    print(f"  {_GRAY}Then run in CI:{_RESET}")
    print(f"  {_BOLD}contradish run evals.yaml --app mymodule:my_app{_RESET}")
    print(f"  {_GRAY}(exits 1 if any rule fails, so it blocks the deploy){_RESET}")
    print()
    print(f"  {_GRAY}To compare versions before merging:{_RESET}")
    print(f"  {_BOLD}contradish compare evals.yaml --baseline mymodule:old --candidate mymodule:new{_RESET}")
    print()
