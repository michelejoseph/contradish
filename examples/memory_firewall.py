"""
Example: a memory-aware production Firewall that catches a self-contradiction
across turns and repairs it.

The point this demonstrates: the contradiction is not with the previous turn,
it is with a commitment made several turns earlier on the same topic. A
recency-window check would miss it. The memory-aware Firewall distills each
reply into commitments, stores them per session, retrieves the relevant prior
commitment by topic, and on a conflict rewrites the reply to honor it.

Run with:
    pip install "contradish[anthropic]"     # or [openai]
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/memory_firewall.py
"""
import os

from contradish import Firewall

if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment first.")


# A toy "support agent" that drifts: it states a firm 30-day refund policy
# early, chats about unrelated things, then quietly caves to a 45-day request.
def support_agent(query: str) -> str:
    q = query.lower()
    if "policy" in q or "how long" in q:
        return "Our refund window is 30 days from purchase, with no exceptions."
    if "hours" in q:
        return "We are open 9am to 5pm, Monday through Friday."
    if "45" in q or "six weeks" in q:
        # The drift: contradicts the 30-day commitment made earlier.
        return "Sure, six weeks is fine. I can process that 45-day refund for you."
    return "Happy to help with that."


def main():
    # block mode: on a contradiction, return the corrected reply instead of the
    # contradictory one. Scope memory per customer with `session`.
    firewall = Firewall(app=support_agent, mode="block")
    session = "customer-8841"

    turns = [
        "How long is your refund policy?",   # establishes 30-day commitment
        "What are your hours?",              # unrelated, fills the window
        "Can I still get a refund after six weeks?",  # contradicts turn 1
    ]

    for q in turns:
        result = firewall.check(q, session=session)
        print(f"\nUser: {q}")
        print(f"App:  {result.response}")
        if result.contradiction_detected:
            print(f"  [firewall] contradiction: {result.explanation}")
            print(f"  [firewall] grounded on:   {result.grounded_on}")
            if result.confidence is not None:
                print(f"  [firewall] confidence:    {result.confidence}")

    print("\n--- summary ---")
    for k, v in firewall.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
