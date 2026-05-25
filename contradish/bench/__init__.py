"""contradish.bench: CAI-Bench runners.

Relocated from the repo root so they ship inside the wheel and resolve their
benchmark data through the installed contradish package. Without this, a fresh
`pip install contradish` could not run `contradish benchmark` (the modules were
not in the wheel).
"""
