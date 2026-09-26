"""Benchmark-specific response text used by lexical metrics."""

import re


_FINAL_ANSWER_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?FINAL[ \t]+ANSWER[ \t]*:[ \t]*"
)
_LONGMEMEVAL_FINAL_ANSWER_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?"
    r"FINAL[ \t]+ANSWER[ \t]*(?:\*\*|__)?[ \t]*:"
    r"[ \t]*(?:\*\*|__)?[ \t]*(?:\r?\n)?"
)


def extract_final_answer(response: str) -> str:
    """Require one line-anchored FINAL ANSWER marker and a non-empty suffix."""
    matches = list(_FINAL_ANSWER_RE.finditer(response))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one line-anchored FINAL ANSWER marker, found {len(matches)}"
        )
    answer = response[matches[0].end():].strip()
    if not answer:
        raise ValueError("empty content after FINAL ANSWER marker")
    return answer


def extract_longmemeval_final_answer(response: str) -> str:
    """Use the last line-level marker, including Markdown heading variants."""
    text = str(response or "")
    matches = list(_LONGMEMEVAL_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        raise ValueError("FINAL ANSWER marker is missing")
    answer = text[matches[-1].end():].strip()
    if not answer:
        raise ValueError("FINAL ANSWER section is empty")
    return answer


def project_lexical_answer(benchmark: str, response: str) -> str:
    if benchmark == "locomo":
        return extract_final_answer(response)
    if benchmark == "longmemeval":
        return extract_longmemeval_final_answer(response)
    if benchmark == "beam":
        return response
    raise ValueError(f"No lexical answer protocol for {benchmark!r}")
