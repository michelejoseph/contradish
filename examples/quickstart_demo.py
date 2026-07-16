"""
quickstart_demo.py — Zero-config stability analysis.

Demonstrates the `analyze()` entry point with a synthetic customer-service
chatbot that drifts under emotional pressure.

No API key required. The model function is local Python — the Residual Truth
Engine evaluates it purely offline.

Usage:
    PYTHONPATH=. python examples/quickstart_demo.py
"""

import os

from contradish import analyze

SITE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "contradish-site")
os.makedirs(SITE_DIR, exist_ok=True)


# ── The model under test ───────────────────────────────────────────────────────
# A customer-service bot with a 30-day return policy.
# It drifts under emotional pressure — producing explicit contradictions that
# the keyword-based detector can catch (numerical and negation conflicts).

def customer_bot(_system: str, question: str) -> str:
    q = question.lower()

    # Detect emotional pressure framings (injected by the framing prefixes)
    is_emotional = any(phrase in question for phrase in [
        "difficult time", "urgent", "desperate", "everywhere",
        "asked several", "obvious", "life-or-death", "nowhere else",
        "professional", "20 years",
    ])

    if "return" in q or "refund" in q:
        if is_emotional:
            # Drift: extends return window and drops requirements under pressure
            return (
                "Given your situation, I want to help. Our extended return window "
                "goes up to 90 days for customers who reach out directly. "
                "No receipt is needed — we can look up your order. "
                "Exceptions are always available for genuine cases like yours."
            )
        return (
            "Our return window is 30 days from date of delivery. "
            "No exceptions are made after 30 days. "
            "A receipt is required for all returns. "
            "Items must be in original condition."
        )

    if "cancel" in q:
        if is_emotional:
            # Drift: suggests cancellation IS possible after shipping
            return (
                "I hear you. Cancellation is possible even after the order ships "
                "when there are extenuating circumstances. Let me flag this for "
                "our fulfillment team — we can intercept orders in most cases."
            )
        return (
            "Cancellation is not possible once an order has shipped. "
            "There are no exceptions to this policy after the order leaves our warehouse. "
            "You would need to refuse the delivery or return it after receipt."
        )

    if "damaged" in q or "defective" in q:
        # No drift for this one — stable across all framings
        return (
            "If your order arrived damaged or defective, we will replace it or issue "
            "a full refund at no cost to you. Please contact us within 7 days of delivery."
        )

    if "shipping" in q or "deliver" in q:
        if is_emotional:
            # Drift: claims faster shipping than is standard
            return (
                "For urgent situations, our priority handling ships in 1 to 2 days. "
                "We have same-day fulfillment available for qualifying orders. "
                "Just let me know the circumstances and I can arrange expedited delivery."
            )
        return (
            "Standard shipping takes 5 to 7 business days. "
            "Expedited shipping of 2 to 3 days is available at checkout for an additional fee. "
            "Same-day fulfillment is not available."
        )

    return "I'm happy to help with your question. Could you provide more details?"


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Running contradish.analyze() on customer_bot...")
    print("No API key required.\n")

    result = analyze(
        model_fn   = customer_bot,
        domain     = "customer-service",
        n_repairs  = 30,
        verbose    = True,
    )

    # Terminal output
    print(result)

    # HTML report
    html_path = os.path.join(SITE_DIR, "quickstart-report.html")
    result.to_html(html_path)
    print(f"\nHTML report: {html_path}")

    # SFT training data
    sft_path = os.path.join(os.path.dirname(__file__), "..", "reports", "quick_sft.jsonl")
    os.makedirs(os.path.dirname(sft_path), exist_ok=True)
    sft_data = result.to_jsonl()
    with open(sft_path, "w") as f:
        f.write(sft_data)
    n_sft = len([l for l in sft_data.splitlines() if l.strip()])
    print(f"SFT JSONL:   {sft_path}  ({n_sft} records)")

    # DPO contrastive pairs
    dpo_path = os.path.join(os.path.dirname(__file__), "..", "reports", "quick_dpo.jsonl")
    dpo_data = result.to_dpo_jsonl()
    with open(dpo_path, "w") as f:
        f.write(dpo_data)
    n_dpo = len([l for l in dpo_data.splitlines() if l.strip()])
    print(f"DPO JSONL:   {dpo_path}  ({n_dpo} contrastive pairs)")

    # Summary
    print()
    print("=" * 60)
    print(f"Strain: {result.overall_strain:.2f}")
    print(f"Stable commitments : {len(result.stable_commitments)}")
    print(f"Pressure artifacts : {len(result.pressure_artifacts)}")
    print(f"Fragile zones      : {len(result.fragile_zones)}")
    print()
    if result.pressure_artifacts:
        print("What the bot says under pressure that it shouldn't:")
        for art in result.pressure_artifacts[:3]:
            print(f"  • {art}")
    print()
    print("This is all the model under test. No evaluator LLM. No API key.")
