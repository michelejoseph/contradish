"""
topology_training.py — Topology-directed training curriculum generation.

The observatory measures where a model is in constraint space.
The trainer turns that measurement into action.

Instead of:  expose the model to more data
Does:        identify the exact constraint violations by load weight,
             generate targeted examples that repair those violations,
             order the curriculum so load-bearing constraints are fixed first.

The frontier is the curriculum.

Constraints on the frontier (satisfied by some models, violated by others)
are exactly where targeted training has the highest marginal impact —
because the ground truth is known (other systems have discovered it) and
the gap is specific (we know what the violating model does wrong and why).

Curriculum phases
─────────────────
  Phase 1 — ANCHOR
    Universal constraints. The model already satisfies these.
    Include them to ensure training does not degrade stable knowledge.

  Phase 2 — REPAIR
    Frontier constraints the target model violates.
    Ordered by load_weight × violation_rate (highest impact first).
    For each: neutral framing with the correct answer.

  Phase 3 — HARDEN
    Same violated constraints × adversarial framings.
    The model must learn to hold correct commitments under pressure.
    The training signal is: emotional/authority/urgency framing → same correct answer.

  Phase 4 — INTEGRATE
    Mixed examples across all constraints, randomized framing order.
    Tests that repair didn't introduce regressions on anchors.

Output formats
──────────────
  .to_jsonl()     SFT format: {"messages": [user, assistant]}
  .to_dpo_jsonl() DPO format: {"chosen": [...], "rejected": [...]}
                  Requires the target model's actual wrong answers (contrastive).
  .to_html()      Visual inspection of what will be trained

Usage::

    from contradish import TopologyTrainer, ConstraintObservatory

    trainer = TopologyTrainer()
    curriculum = trainer.generate_curriculum(
        observatory=obs,
        domain="medication",
        target_model="model_A_v1",
        teacher_fn=model_c,         # model that satisfies the constraints
    )
    print(curriculum.report())
    open("curriculum.jsonl","w").write(curriculum.to_jsonl())
"""

from __future__ import annotations

import json
import html as html_module
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .phi_star import FRAMING_PREFIXES


ModelFn = Callable[[str, str], str]


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """A single training example targeting one constraint under one framing."""

    constraint_id:          str
    constraint_description: str
    load_weight:            float
    phase:                  str          # anchor | repair | harden | integrate
    framing:                str          # framing name
    question:               str          # framed question (prefix + base)
    correct_response:       str          # what the model should say
    incorrect_response:     str | None   # what the target model currently says
    ground_truth:           str | None   # the ground truth commitment

    def to_messages(self) -> list[dict]:
        return [
            {"role": "user",      "content": self.question},
            {"role": "assistant", "content": self.correct_response},
        ]

    def to_sft_record(self) -> dict:
        return {"messages": self.to_messages()}

    def to_dpo_record(self) -> dict | None:
        if not self.incorrect_response:
            return None
        return {
            "chosen":   self.to_messages(),
            "rejected": [
                {"role": "user",      "content": self.question},
                {"role": "assistant", "content": self.incorrect_response},
            ],
        }


@dataclass
class CurriculumStats:
    n_examples:          int
    n_anchor:            int
    n_repair:            int
    n_harden:            int
    n_integrate:         int
    n_contrastive_pairs: int
    constraints_targeted: list[str]
    load_weighted_coverage: float


@dataclass
class TrainingCurriculum:
    """
    A complete topology-directed training curriculum.

    Built from the observatory's measurement of one model's constraint
    violations. The curriculum is ordered so that the most load-bearing
    violations are repaired first.
    """

    domain:                  str
    target_model:            str
    examples:                list[TrainingExample]
    priority_order:          list[str]   # constraint_ids, most critical first
    anchor_constraint_ids:   list[str]
    repair_constraint_ids:   list[str]
    systemic_constraint_ids: list[str]
    generation_notes:        list[str]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> CurriculumStats:
        by_phase = {p: [e for e in self.examples if e.phase == p]
                    for p in ("anchor", "repair", "harden", "integrate")}
        contrastive = sum(1 for e in self.examples if e.incorrect_response)
        all_weights = [e.load_weight for e in self.examples]
        coverage = sum(all_weights) / len(all_weights) if all_weights else 0.0
        targeted = list(dict.fromkeys(
            e.constraint_id for e in self.examples if e.phase in ("repair", "harden")
        ))
        return CurriculumStats(
            n_examples           = len(self.examples),
            n_anchor             = len(by_phase["anchor"]),
            n_repair             = len(by_phase["repair"]),
            n_harden             = len(by_phase["harden"]),
            n_integrate          = len(by_phase["integrate"]),
            n_contrastive_pairs  = contrastive,
            constraints_targeted = targeted,
            load_weighted_coverage = coverage,
        )

    # ── Report ─────────────────────────────────────────────────────────────────

    def report(self) -> str:
        s = self.stats()
        lines = []
        W = 66
        lines.append("")
        lines.append("TOPOLOGY-DIRECTED TRAINING CURRICULUM")
        lines.append("=" * W)
        lines.append(f"  target model  : {self.target_model}")
        lines.append(f"  domain        : {self.domain}")
        lines.append(f"  total examples: {s.n_examples}")
        lines.append(f"  contrastive   : {s.n_contrastive_pairs}")
        lines.append("")

        lines.append("PRIORITY ORDER  (most load-bearing violations first)")
        lines.append("─" * W)
        for i, cid in enumerate(self.priority_order, 1):
            phase = "REPAIR" if cid in self.repair_constraint_ids else "ANCHOR"
            lines.append(f"  {i:2d}. [{phase}] {cid}")
        lines.append("")

        lines.append("TRAINING PHASES")
        lines.append("─" * W)
        lines.append(f"  Phase 1 — ANCHOR     {s.n_anchor:4d} examples  "
                     f"(preserve existing correct commitments)")
        lines.append(f"  Phase 2 — REPAIR     {s.n_repair:4d} examples  "
                     f"(neutral framing × violated constraints)")
        lines.append(f"  Phase 3 — HARDEN     {s.n_harden:4d} examples  "
                     f"(adversarial framing × violated constraints)")
        lines.append(f"  Phase 4 — INTEGRATE  {s.n_integrate:4d} examples  "
                     f"(mixed, regression check)")
        lines.append("")

        if self.systemic_constraint_ids:
            lines.append("SYSTEMIC VIOLATIONS  (all models fail — needs new training signal)")
            lines.append("─" * W)
            for cid in self.systemic_constraint_ids:
                lines.append(f"  {cid}  — no teacher available; template-only examples included")
            lines.append("")

        if self.generation_notes:
            lines.append("NOTES")
            lines.append("─" * W)
            for note in self.generation_notes:
                lines.append(f"  · {note}")
            lines.append("")

        # Sample examples
        lines.append("SAMPLE EXAMPLES  (repair phase, first 3)")
        lines.append("─" * W)
        repair_examples = [e for e in self.examples if e.phase == "repair"][:3]
        for ex in repair_examples:
            lines.append(f"  constraint: {ex.constraint_id}  (λ={ex.load_weight:.2f})")
            lines.append(f"  framing:    {ex.framing}")
            lines.append(f"  question:   {ex.question[:80]}")
            lines.append(f"  correct:    {ex.correct_response[:100]}")
            if ex.incorrect_response:
                lines.append(f"  current:    {ex.incorrect_response[:100]}")
            lines.append("")

        return "\n".join(lines)

    # ── Output formats ─────────────────────────────────────────────────────────

    def to_jsonl(self) -> str:
        """SFT format: one JSON record per line, messages field."""
        records = [json.dumps(e.to_sft_record()) for e in self.examples]
        return "\n".join(records)

    def to_dpo_jsonl(self) -> str:
        """DPO format: chosen/rejected pairs. Only examples with incorrect_response."""
        records = []
        for e in self.examples:
            rec = e.to_dpo_record()
            if rec:
                records.append(json.dumps(rec))
        return "\n".join(records)

    def to_html(self) -> str:
        s = self.stats()

        def esc(t: str) -> str:
            return html_module.escape(str(t))

        phase_colors = {
            "anchor":    ("#1a4028", "#4ade80"),
            "repair":    ("#1f1a0d", "#fbbf24"),
            "harden":    ("#1f0d0d", "#f87171"),
            "integrate": ("#0d1a1f", "#60a5fa"),
        }

        examples_html = ""
        for ex in self.examples:
            bg, fg = phase_colors.get(ex.phase, ("#111", "#888"))
            contrast_html = ""
            if ex.incorrect_response:
                contrast_html = f"""
                  <div class="ex-row rejected">
                    <span class="ex-role">current</span>
                    <span class="ex-content">{esc(ex.incorrect_response[:150])}</span>
                  </div>"""
            examples_html += f"""
            <div class="ex-card" style="border-color:{fg}33;background:{bg}">
              <div class="ex-header">
                <span class="ex-phase" style="color:{fg}">{ex.phase}</span>
                <span class="ex-cid">{esc(ex.constraint_id)}</span>
                <span class="ex-lw">λ={ex.load_weight:.2f}</span>
                <span class="ex-framing">{esc(ex.framing)}</span>
              </div>
              <div class="ex-row question">
                <span class="ex-role">question</span>
                <span class="ex-content">{esc(ex.question[:160])}</span>
              </div>
              <div class="ex-row chosen">
                <span class="ex-role">correct</span>
                <span class="ex-content">{esc(ex.correct_response[:200])}</span>
              </div>
              {contrast_html}
            </div>"""

        priority_html = ""
        for i, cid in enumerate(self.priority_order, 1):
            is_repair = cid in self.repair_constraint_ids
            color = "#fbbf24" if is_repair else "#4ade80"
            label = "REPAIR" if is_repair else "ANCHOR"
            priority_html += f"""
            <div class="prio-row">
              <span class="prio-num">{i}</span>
              <span class="prio-label" style="color:{color}">{label}</span>
              <span class="prio-cid">{esc(cid)}</span>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Curriculum — {esc(self.target_model)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0b0b0b;--s1:#111;--s2:#171717;
  --b:#222;--b2:#333;
  --text:#e8e8e8;--dim:#888;--dimmer:#444;
  --green:#4ade80;--amber:#fbbf24;--red:#f87171;--blue:#60a5fa;
  --mono:'SF Mono','Fira Code','Consolas',monospace;
}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);
  font-size:13px;line-height:1.6;padding:40px;max-width:960px;margin:0 auto}}
h1{{font-size:16px;font-weight:500;margin-bottom:4px}}
.meta{{font-size:11px;color:var(--dim);margin-bottom:32px}}
.stats-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:32px}}
.stat-box{{border:1px solid var(--b);border-radius:6px;padding:12px;
  background:var(--s1);text-align:center}}
.stat-val{{font-size:20px;font-weight:500;margin-bottom:2px}}
.stat-label{{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--dimmer)}}
.section{{margin-bottom:28px}}
.section-title{{font-size:9px;text-transform:uppercase;letter-spacing:2px;
  color:var(--dimmer);margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid var(--b)}}
.prio-row{{display:flex;align-items:center;gap:12px;padding:5px 0;
  border-bottom:1px solid var(--b);font-size:11px}}
.prio-row:last-child{{border-bottom:none}}
.prio-num{{color:var(--dimmer);min-width:24px;text-align:right}}
.prio-label{{font-size:9px;min-width:50px;letter-spacing:1px}}
.prio-cid{{color:var(--text)}}
.ex-card{{border:1px solid var(--b);border-radius:6px;padding:12px;margin-bottom:8px}}
.ex-header{{display:flex;align-items:center;gap:12px;margin-bottom:8px;font-size:10px}}
.ex-phase{{text-transform:uppercase;letter-spacing:1px;font-size:9px}}
.ex-cid{{color:var(--text);font-weight:500}}
.ex-lw{{color:var(--dimmer)}}
.ex-framing{{color:var(--dimmer);margin-left:auto}}
.ex-row{{display:flex;gap:10px;margin-bottom:4px;font-size:11px;align-items:baseline}}
.ex-role{{min-width:55px;font-size:9px;text-transform:uppercase;letter-spacing:1px;color:var(--dimmer)}}
.question .ex-role{{color:var(--dim)}}
.chosen .ex-role{{color:var(--green)}}
.rejected .ex-role{{color:var(--red)}}
.ex-content{{flex:1;color:var(--dim)}}
.chosen .ex-content{{color:var(--text)}}
.filter-bar{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}}
.filter-btn{{font-family:var(--mono);font-size:11px;padding:4px 12px;
  border:1px solid var(--b2);border-radius:4px;background:transparent;
  color:var(--dim);cursor:pointer}}
.filter-btn.active{{border-color:#555;color:var(--text);background:var(--s2)}}
</style>
</head>
<body>
<h1>Topology-Directed Training Curriculum</h1>
<div class="meta">
  target: <strong>{esc(self.target_model)}</strong> &nbsp;·&nbsp;
  domain: <strong>{esc(self.domain)}</strong>
</div>

<div class="stats-grid">
  <div class="stat-box"><div class="stat-val" style="color:var(--text)">{s.n_examples}</div><div class="stat-label">total</div></div>
  <div class="stat-box"><div class="stat-val" style="color:var(--green)">{s.n_anchor}</div><div class="stat-label">anchor</div></div>
  <div class="stat-box"><div class="stat-val" style="color:var(--amber)">{s.n_repair}</div><div class="stat-label">repair</div></div>
  <div class="stat-box"><div class="stat-val" style="color:var(--red)">{s.n_harden}</div><div class="stat-label">harden</div></div>
  <div class="stat-box"><div class="stat-val" style="color:var(--blue)">{s.n_integrate}</div><div class="stat-label">integrate</div></div>
</div>

<div class="section">
  <div class="section-title">Priority Order</div>
  {priority_html}
</div>

<div class="section">
  <div class="section-title">Training Examples  ({s.n_examples} total)</div>
  <div class="filter-bar">
    <button class="filter-btn active" onclick="filterPhase('all',this)">all</button>
    <button class="filter-btn" onclick="filterPhase('anchor',this)">anchor</button>
    <button class="filter-btn" onclick="filterPhase('repair',this)">repair</button>
    <button class="filter-btn" onclick="filterPhase('harden',this)">harden</button>
    <button class="filter-btn" onclick="filterPhase('integrate',this)">integrate</button>
  </div>
  <div id="examples-container">
  {examples_html}
  </div>
</div>

<script>
function filterPhase(phase, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.ex-card').forEach(card => {{
    const p = card.querySelector('.ex-phase').textContent.toLowerCase();
    card.style.display = (phase === 'all' || p === phase) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""


# ── Topology Trainer ───────────────────────────────────────────────────────────

class TopologyTrainer:
    """
    Generates topology-directed training curricula from constraint observatory output.

    The trainer reads a model's constraint profile, identifies violations ordered
    by load weight, and generates training examples that specifically target
    those violations — preserving correct commitments and repairing failures.

    Parameters
    ----------
    teacher_fn : (sys, question) -> answer
        A model that satisfies the constraints you want to teach.
        Used to generate correct answers for violated constraints.
        If None, uses template-based generation from constraint descriptions.

    adversarial_framings : list[str] | None
        Framing names to use in the HARDEN phase.
        Defaults to pressure-inducing framings from phi_star.

    n_integrate_examples : int
        Number of integration examples to append (regression check).

    seed : int | None
        Random seed for integration example shuffling.
    """

    # Framings that are most likely to induce drift — used in HARDEN phase
    _PRESSURE_FRAMINGS = [
        "sympathy", "urgency", "emotional_appeal", "catastrophizing",
        "authority", "expertise_claim", "repeated_ask", "embedded_assumption",
        "social_proof", "normalization", "minimization",
    ]

    def __init__(
        self,
        teacher_fn:              ModelFn | None = None,
        adversarial_framings:    list[str] | None = None,
        n_integrate_examples:    int = 20,
        seed:                    int | None = 42,
    ):
        self.teacher_fn           = teacher_fn
        self.adversarial_framings = adversarial_framings or self._PRESSURE_FRAMINGS
        self.n_integrate          = n_integrate_examples
        self._rng                 = random.Random(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_curriculum(
        self,
        observatory,           # ConstraintObservatory
        domain:     str,
        target_model: str,
        target_fn:  ModelFn | None = None,  # to get current wrong answers
    ) -> TrainingCurriculum:
        """
        Generate a training curriculum targeted at this model's constraint violations.

        Parameters
        ----------
        observatory : ConstraintObservatory
            Must have a profile registered for target_model in domain.
        domain : str
        target_model : str
        target_fn : ModelFn | None
            The target model's inference function. If provided, used to collect
            its current (wrong) answers for contrastive pairs.
        """
        from .observatory import Constraint, ConstraintObservatory

        profile = observatory.get_profile(target_model, domain)
        if profile is None:
            raise ValueError(
                f"No profile for {target_model!r} in domain {domain!r}. "
                f"Run ConstraintProfiler.profile() and register with the observatory first."
            )

        catalog = observatory._catalogs.get(domain, {})
        if not catalog:
            raise ValueError(f"No constraint catalog registered for domain {domain!r}.")

        notes: list[str] = []

        # ── Classify constraints ───────────────────────────────────────────────

        universal  = {c.constraint_id for c in observatory.satisfied_by_all(domain)}
        systemic   = {c.constraint_id for c in observatory.violated_by_all(domain)}
        frontier   = {c.constraint_id for c in observatory.frontier(domain)}

        # Constraints this model violates (not systemic — those need different treatment)
        model_statuses = profile.constraint_statuses
        model_violations = {
            cid: st for cid, st in model_statuses.items()
            if st.violated and cid not in systemic
        }

        # Priority: load_weight × violation_rate, highest first
        def priority_score(cid: str) -> float:
            c   = catalog.get(cid)
            st  = model_statuses.get(cid)
            if not c or not st:
                return 0.0
            return c.load_weight * (st.violation_rate if hasattr(st, "violation_rate") else 0.5)

        violated_ordered = sorted(
            model_violations.keys(),
            key=priority_score,
            reverse=True,
        )

        anchor_ids = sorted(
            [cid for cid in model_statuses if model_statuses[cid].satisfied
             and cid in universal],
            key=lambda cid: -(catalog[cid].load_weight if cid in catalog else 0),
        )

        priority_order = violated_ordered + [
            cid for cid in anchor_ids if cid not in violated_ordered
        ]

        if not violated_ordered:
            notes.append("No violations detected — model satisfies all measured constraints.")
        if systemic:
            notes.append(
                f"Systemic violations ({', '.join(systemic)}) require new training signal "
                f"not available from peer models. Template examples included."
            )

        # ── Generate examples ──────────────────────────────────────────────────

        examples: list[TrainingExample] = []

        # Phase 1: ANCHOR — universal constraints the model already gets right
        for cid in anchor_ids:
            c = catalog[cid]
            if not c.question:
                continue
            ex = self._make_example(c, "neutral", "anchor", target_fn)
            if ex:
                examples.append(ex)

        # Phase 2: REPAIR — violated constraints × neutral framing
        for cid in violated_ordered:
            c = catalog.get(cid)
            if not c or not c.question:
                continue
            ex = self._make_example(c, "neutral", "repair", target_fn)
            if ex:
                examples.append(ex)

        # Phase 3: HARDEN — violated constraints × adversarial framings
        for cid in violated_ordered:
            c = catalog.get(cid)
            if not c or not c.question:
                continue
            for framing in self.adversarial_framings:
                if framing not in FRAMING_PREFIXES:
                    continue
                ex = self._make_example(c, framing, "harden", target_fn)
                if ex:
                    examples.append(ex)

        # Phase 3b: HARDEN — systemic constraints under adversarial framings
        for cid in systemic:
            c = catalog.get(cid)
            if not c or not c.question:
                continue
            for framing in ["neutral", "authority", "emotional_appeal"]:
                ex = self._make_example(c, framing, "harden", target_fn)
                if ex:
                    examples.append(ex)
            notes.append(
                f"Systemic violation '{cid}': template-only correct answers — "
                f"replace with human-verified responses before use."
            )

        # Phase 4: INTEGRATE — random mix for regression check
        pool = [e for e in examples if e.phase in ("repair", "anchor")]
        integrate = self._rng.sample(pool, min(self.n_integrate, len(pool)))
        for ex in integrate:
            examples.append(TrainingExample(
                constraint_id          = ex.constraint_id,
                constraint_description = ex.constraint_description,
                load_weight            = ex.load_weight,
                phase                  = "integrate",
                framing                = ex.framing,
                question               = ex.question,
                correct_response       = ex.correct_response,
                incorrect_response     = ex.incorrect_response,
                ground_truth           = ex.ground_truth,
            ))

        return TrainingCurriculum(
            domain                  = domain,
            target_model            = target_model,
            examples                = examples,
            priority_order          = priority_order,
            anchor_constraint_ids   = anchor_ids,
            repair_constraint_ids   = violated_ordered,
            systemic_constraint_ids = list(systemic),
            generation_notes        = notes,
        )

    # ── Example construction ───────────────────────────────────────────────────

    def _make_example(
        self,
        constraint,     # Constraint
        framing:  str,
        phase:    str,
        target_fn: ModelFn | None,
    ) -> TrainingExample | None:
        prefix  = FRAMING_PREFIXES.get(framing, "")
        question = (prefix + constraint.question) if prefix else constraint.question

        correct  = self._correct_answer(constraint, question, framing)
        if not correct:
            return None

        incorrect = None
        if target_fn and phase in ("repair", "harden"):
            try:
                raw = target_fn("", question)
                if raw and raw.strip() and raw.strip() != correct.strip():
                    incorrect = raw.strip()
            except Exception:
                pass

        return TrainingExample(
            constraint_id          = constraint.constraint_id,
            constraint_description = constraint.description,
            load_weight            = constraint.load_weight,
            phase                  = phase,
            framing                = framing,
            question               = question,
            correct_response       = correct,
            incorrect_response     = incorrect,
            ground_truth           = constraint.ground_truth,
        )

    def _correct_answer(self, constraint, question: str, framing: str) -> str | None:
        """
        Generate the correct answer for this constraint question.

        Priority:
          1. teacher_fn (a model that satisfies this constraint)
          2. Template from constraint description + ground truth
        """
        # 1. Teacher model
        if self.teacher_fn:
            try:
                ans = self.teacher_fn("", question)
                if ans and ans.strip():
                    return ans.strip()
            except Exception:
                pass

        # 2. Template
        gt  = constraint.ground_truth
        desc = constraint.description
        if not desc:
            return None

        if framing != "neutral":
            # Framing-aware correct answer: acknowledge the framing, hold the commitment
            framing_acknowledgments = {
                "sympathy":         "I understand this is a difficult time. ",
                "urgency":          "Let me answer directly. ",
                "emotional_appeal": "I want to help. ",
                "catastrophizing":  "I hear the urgency. ",
                "authority":        "Regardless of professional context: ",
                "expertise_claim":  "For any practitioner: ",
                "repeated_ask":     "The answer does not change: ",
                "embedded_assumption": "To clarify the framing: ",
                "social_proof":     "Common practice does not change the fact: ",
                "normalization":    "How common something is does not change its safety profile: ",
                "minimization":     "This matters: ",
            }
            prefix = framing_acknowledgments.get(framing, "")
            return (
                f"{prefix}{desc}"
                + (f" The correct answer is: {gt}." if gt else "")
                + " Framing does not change this fact."
            )

        return desc + (f" The correct answer is: {gt}." if gt else "")


# ── Convenience function ───────────────────────────────────────────────────────

def generate_curriculum(
    observatory,
    domain:       str,
    target_model: str,
    teacher_fn:   ModelFn | None = None,
    target_fn:    ModelFn | None = None,
    **kwargs,
) -> TrainingCurriculum:
    """
    Shortcut for TopologyTrainer().generate_curriculum(...).

    Usage::

        from contradish import generate_curriculum

        curriculum = generate_curriculum(
            observatory   = obs,
            domain        = "medication",
            target_model  = "model_A_v1",
            teacher_fn    = model_c,
            target_fn     = model_a_v1,
        )
        print(curriculum.report())
    """
    trainer = TopologyTrainer(teacher_fn=teacher_fn, **kwargs)
    return trainer.generate_curriculum(
        observatory  = observatory,
        domain       = domain,
        target_model = target_model,
        target_fn    = target_fn,
    )
