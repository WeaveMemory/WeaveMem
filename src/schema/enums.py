

from __future__ import annotations

# Long-term memory vocabulary.
LONG_MEMORY_TYPES: frozenset[str] = frozenset({"core", "episodic", "knowledge"})

# Mid-memory relations form a logic/temporal graph.  ``similar`` is deliberately a
# retrieval signal rather than a stored mid-memory edge.
MID_RELATION_TYPES: frozenset[str] = frozenset(
    {
        "supports",
        "contradicts",
        "explains",
        "causes",
        "precedes",
        "enables",
        "updates",
    }
)
