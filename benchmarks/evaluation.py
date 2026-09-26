
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Callable

from benchmarks.answer_text import project_lexical_answer
from benchmarks.lexical_metrics import calculate_answer_metrics


_PROMPTS = Path(__file__).parent / "prompts"
_BENCHMARKS = {"locomo", "longmemeval", "personamem", "beam"}
Judge = Callable[[list[dict]], str]

SCORE_PROTOCOLS = {
    "locomo": "weavemem_semantic_judge",
    "longmemeval": "longmemeval_official_judge",
    "personamem": "personamem_official_option_set",
    "beam": "beam_nugget_v2",
}


def _benchmark(value: str) -> str:
    if value not in _BENCHMARKS:
        raise ValueError(f"Unknown benchmark {value!r}; choose {sorted(_BENCHMARKS)}")
    return value


def _template(filename: str) -> str:
    return (_PROMPTS / filename).read_text(encoding="utf-8")


def _render_literal(template: str, **values: object) -> str:
    # One pass: braces inside a retrieved memory remain literal model input.
    return re.sub(r"\{\{([A-Za-z_]+)\}\}", lambda m: str(values[m[1]]), template)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _memory_context(memories: list[dict]) -> str:
    """Render projected memory content with selected time and provenance fields.

    Explicit field selection keeps retrieval scores and other metadata out of
    the answer prompt. ``content`` already includes attached child facts.
    """
    lines = []
    for index, memory in enumerate(memories, 1):
        if not isinstance(memory, dict):
            raise ValueError("Each retrieved memory must be an object")
        text = _text(memory.get("content") or memory.get("text") or memory.get("summary"))
        provenance = []
        for key in (
            "id",
            "session_id",
            "session_date",
            "chat_ids",
            "user_id",
            "eventTime",
            "mentionTime",
            "originalTimeExpression",
            "timePrecision",
        ):
            value = memory.get(key)
            if value not in (None, "", [], {}):
                provenance.append(f"{key}: {_text(value)}")
        heading = f"Memory {index}" + (" | " + " | ".join(provenance) if provenance else "")
        lines.append(f"[{heading}]\n{text}")
    return "\n\n".join(lines) or "(No memories retrieved.)"


def _recent_context(context: object) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, list):
        lines = []
        for item in context:
            if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                raise ValueError("Recent context must contain role/content message objects")
            lines.append(f"{item.get('role', 'user')}: {item['content']}")
        return "\n".join(lines)
    raise ValueError("Recent context must be text or a list of messages")


_CHILD_MARKER = "\nRelevant child facts:\n"
_HIDDEN_FIELDS = (
    "intent_primary",
    "intent_secondary",
    "intent_description",
    "originalTimeExpression",
    "timePrecision",
    "eventTime",
    "mentionTime",
)
_HIDDEN_LINE = re.compile(r"^[ \t]*(?:" + "|".join(_HIDDEN_FIELDS) + r")[ \t]*:")


def _append_source_evidence(content: str, record: dict) -> str:
    """Frozen answer-time source quote rendering, independent of QA evidence."""
    raw = record.get("sourceEvidence") or record.get("source_evidence") or []
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    lines = []
    for item in items:
        if isinstance(item, dict):
            ident = " ".join(str(item.get("chatId") or item.get("chat_id") or "").split())
            quote = " ".join(
                str(item.get("quote") or item.get("verbatimSpan") or item.get("text") or "").split()
            )
        else:
            ident, quote = "", " ".join(str(item or "").split())
        parts = ([f"chatId: {ident}"] if ident else []) + ([f"quote: {quote}"] if quote else [])
        if parts:
            lines.append("- " + " | ".join(parts))
    evidence = "\n".join(["sourceEvidence:", *lines]) if lines else ""
    base = str(content or "").strip()
    return (
        "\n".join(part for part in (base, evidence) if part)
        if evidence and evidence not in base
        else base
    )


def _locomo_context(memories: list[dict]) -> str:
    # The frozen LoCoMo reference consumes the content packet already assembled
    # by frozen retrieval; do not add a second provenance header, graph tail, or
    # recent context.
    return (
        "\n".join(
            "- "
            + _append_source_evidence(
                str(m.get("content") or m.get("summary") or m.get("topic_subject") or ""),
                m,
            )
            for m in memories
        )
        or "(no memories retrieved)"
    )


def _filter_fields(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _HIDDEN_LINE.match(line))


def _children(memory: dict) -> list[dict]:
    children = memory.get("query_relevant_child_facts") or []
    if not isinstance(children, list) or any(not isinstance(c, dict) for c in children):
        raise ValueError("Selected child facts must be a list of objects")
    return children


def _synthetic_span(record: dict) -> tuple[str, str]:
    span = tuple(record.get("synthetic_source_mention_" + part) for part in ("start", "end"))
    if not all(isinstance(value, str) and value.strip() for value in span):
        raise ValueError(
            "PersonaMem requires validated synthetic_source_mention_start/end on every selected memory and child"
        )
    try:
        parsed = [datetime.fromisoformat(value) for value in span]
        if parsed[0] > parsed[1]:
            raise ValueError("reversed span")
    except (ValueError, TypeError) as exc:
        raise ValueError("PersonaMem synthetic source mention span is invalid") from exc
    return span


def _span_label(span: tuple[str, str]) -> str:
    return f"Synthetic source mention span: {span[0]} — {span[1]}"


def _persona_context(memories: list[dict]) -> str:
    prepared = []
    for rank, memory in enumerate(memories):
        span = _synthetic_span(memory)
        children = _children(memory)
        content = str(
            memory.get("content") or memory.get("summary") or memory.get("topic_subject") or ""
        )
        if children:
            base, marker, embedded = content.partition(_CHILD_MARKER)
            expected = "\n".join(
                f"{i}. {child.get('content') or ''}" for i, child in enumerate(children, 1)
            )
            if not marker or embedded.strip() != expected.strip():
                raise ValueError(
                    "PersonaMem embedded child-fact packet does not match selected child content"
                )
        else:
            base = content
            if _CHILD_MARKER in base:
                raise ValueError(
                    "PersonaMem has embedded child facts without their source span records"
                )
        child_rows = [
            (_synthetic_span(child), i, _filter_fields(str(child.get("content") or "")).strip())
            for i, child in enumerate(children)
        ]
        child_rows.sort(key=lambda row: (row[0][0], row[0][1], row[1]))
        lines = [_filter_fields(base).strip()]
        if child_rows:
            lines.append("Relevant child facts:")
            for i, (child_span, _, text) in enumerate(child_rows, 1):
                lines.extend((f"{i}. [{_span_label(child_span)}]", text))
        body = _append_source_evidence("\n".join(lines), memory)
        prepared.append((span, rank, f"- [{_span_label(span)}]\n{body}"))
    prepared.sort(key=lambda row: (row[0][0], row[0][1], row[1]))
    return "\n".join(row[2] for row in prepared) or "(no memories retrieved)"


def _source_record(record: dict) -> dict:
    source = record.get("source_record")
    if not isinstance(source, dict):
        raise ValueError("BEAM answer projection requires each selected memory's source_record")
    return source


def _field(row: dict, *names: str) -> str:
    return next(
        (str(row.get(name) or "").strip() for name in names if str(row.get(name) or "").strip()),
        "unknown",
    )


def _structured_field(row: dict, field: str) -> str:
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (list, dict)) and value:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    raise ValueError(f"BEAM mid_dialogue_state projection requires nonempty {field}")


def _beam_child(child: dict) -> str:
    row = _source_record(child)
    kind = row.get("type")
    content = str(row.get("content") or "").strip()
    if not content:
        raise ValueError("BEAM child source_record requires nonempty content")
    if kind == "core":
        entity = str(row.get("entity_name") or row.get("entityName") or "").strip()
        user = str(row.get("user_id") or "").strip()
        subject = user if entity.casefold() == "user" and user else entity
        lines = [
            f"user_id: {user or 'unknown'}",
            f"topic: {_field(row, 'topic')}",
            f"subtopic: {_field(row, 'subtopic', 'subTopic')}",
            f"content: {content}",
            f"entity_name: {subject or 'unknown'}",
        ]
    elif kind == "episodic":
        lines = [
            f"user_id: {_field(row, 'user_id')}",
            f"content: {content}",
            f"context: {_field(row, 'context')}",
        ]
    elif kind == "knowledge":
        lines = [f"name: {_field(row, 'name')}", f"content: {content}"]
    else:
        raise ValueError(f"BEAM formal projection does not support child type {kind!r}")
    return _filter_fields("\n".join(lines))


def _beam_context(memories: list[dict]) -> str:
    relations = memories[0].get("selected_graph_relations", []) if memories else []
    if not isinstance(relations, list):
        raise ValueError("BEAM selected_graph_relations must be a list")
    labels = {str(m.get("id") or ""): f"M{i}" for i, m in enumerate(memories, 1)}
    blocks = []
    for index, memory in enumerate(memories, 1):
        row = _source_record(memory)
        lines = []
        for field, aliases in (
            ("topic_subject", ("topic_subject", "topic")),
            ("summary", ("summary",)),
        ):
            value = next(
                (
                    str(row.get(key) or "").strip()
                    for key in aliases
                    if str(row.get(key) or "").strip()
                ),
                "",
            )
            if value:
                lines.append(f"{field}: {value}")
        base = _filter_fields("\n".join(lines))
        base += "".join(
            f"\n{field}: {_structured_field(row, field)}"
            for field in ("dialogue_phase", "user_attitude")
        )
        children = _children(memory)
        if children:
            base += _CHILD_MARKER + "\n".join(
                f"{i}. {_beam_child(child)}" for i, child in enumerate(children, 1)
            )
        blocks.append((f"- [M{index}]\n" if relations else "- ") + base)
    if relations:
        blocks.append("Selected logical relationships among retrieved memories:")
        for edge in relations:
            if not isinstance(edge, dict) or any(
                not isinstance(edge.get(k), str) or not edge[k].strip()
                for k in ("source_id", "target_id", "relation_type", "description")
            ):
                raise ValueError("BEAM selected relation has incomplete fields")
            if edge["source_id"] not in labels or edge["target_id"] not in labels:
                raise ValueError(
                    "BEAM selected relation references a memory outside the answer context"
                )
            blocks.append(
                f"- [{labels[edge['source_id']]}] --{edge['relation_type']}--> [{labels[edge['target_id']]}]\n"
                f"  description: {edge['description']}"
            )
    return "\n".join(blocks) or "(no memories retrieved)"


def _beam_timeline(memories: list[dict]) -> str:
    entries = []
    for index, memory in enumerate(memories):
        row = _source_record(memory)
        raw = memory.get("session_date") or row.get("session_date")
        text = str(raw or "").strip()
        parsed = None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            for fmt in ("%B-%d-%Y", "%b-%d-%Y"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    pass
        label = parsed.strftime("%Y-%m-%d") if parsed else str(raw or "unknown")
        session = str(memory.get("session_id") or row.get("session_id") or "")
        if session:
            label += f" (session {session})"
        topic = str(row.get("topic_subject") or "").strip()
        entries.append(
            (
                (parsed is None, parsed or datetime.max, index),
                f"- {label} | {topic[:120]}" + ("…" if len(topic) > 120 else ""),
            )
        )
    entries.sort(key=lambda entry: entry[0])
    return "\n".join(entry[1] for entry in entries)


def answer_messages(benchmark: str, question: dict, memories: list[dict]) -> list[dict]:
    """Build model input without reading answer, category, evidence or metadata.

    ``context`` is adapter-sanitized, model-visible recent chat. Gold references
    and grading rubrics are only consumed by ``score_prediction``.
    """
    benchmark = _benchmark(benchmark)
    query = str(question.get("question") or "").strip()
    if not query:
        raise ValueError("A non-empty question is required")
    if not isinstance(memories, list) or any(not isinstance(m, dict) for m in memories):
        raise ValueError("Retrieved memories must be a list of objects")
    if benchmark == "locomo":
        context = _locomo_context(memories)
        prompt = _render_literal(_template("locomo_answer.txt"), MEMORIES=context, QUESTION=query)
    elif benchmark == "longmemeval":
        context = _memory_context(memories)
        recent = _recent_context(question.get("context"))
        if recent:
            context += "\n\nRecent conversation context:\n" + recent
        date = question.get("question_date")
        if not date:
            raise ValueError("LongMemEval requires question_date; never use the wall-clock date")
        prompt = _render_literal(
            _template("longmemeval_answer.txt"),
            MEMORIES=context,
            QUESTION=query,
            QuestionDate=date,
        )
    elif benchmark == "personamem":
        context = _persona_context(memories)
        choices = question.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 4
            or not all(isinstance(c, str) for c in choices)
        ):
            raise ValueError("PersonaMem requires exactly four string choices")
        options = "\n".join(
            "({}) {}".format(chr(97 + i), re.sub(r"^\([a-dA-D]\)\s*", "", choice))
            for i, choice in enumerate(choices)
        )
        prompt = _template("personamem_answer.txt").format(
            context=context, question=query, options=options
        )
    else:
        prompt = _template("beam_answer.txt").format(
            context=_beam_context(memories),
            timeline=_beam_timeline(memories),
            question_date=question.get("question_date") or "unknown",
            question=query,
        )
    return [{"role": "user", "content": prompt}]


def _option_set(text: str) -> set[str]:
    lowered = text.lower()
    parenthesized = re.findall(r"\(([a-d])\)", lowered)
    return set(parenthesized or re.findall(r"\b([a-d])\b", lowered))


def _persona_parts(response: str) -> tuple[set[str], set[str]]:
    final = response.strip().split("<final_answer>")[-1].strip()
    if final.endswith("</final_answer>"):
        final = final[: -len("</final_answer>")].strip()
    return _option_set(final), _option_set(response)


def parse_answer(benchmark: str, response: str) -> str:
    """Extract the answer using the named protocol, independently of any gold.

    PersonaMem invalid/ambiguous responses return an empty label and count as
    wrong. LoCoMo, LongMemEval and BEAM retain the full response, including
    any reasoning sections, because their configured judges see that full text.
    """
    benchmark = _benchmark(benchmark)
    text = str(response or "").strip()
    if benchmark == "personamem":
        for options in _persona_parts(text):
            if len(options) == 1:
                return f"({next(iter(options))})"
        return ""
    if not text:
        raise ValueError(f"{benchmark} answer response is empty")
    return text


def _judge(judge: Judge | None, prompt: str, *, system: str | None = None) -> str:
    if judge is None:
        raise ValueError(
            "This scoring protocol requires judge(messages) -> str; configure a judge model or run answer-only"
        )
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    raw = judge(messages)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Judge returned no text; this question was not scored")
    return raw.strip()


def _json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Judge returned invalid JSON; this question was not scored")


def _longmemeval_judge(question: dict, prediction: str, judge: Judge | None) -> dict:
    abstention = str(question.get("id", "")).endswith("_abs")
    category = str(question.get("category") or "")
    names = {
        "single-session-user": "standard",
        "single-session-assistant": "standard",
        "multi-session": "standard",
        "temporal-reasoning": "temporal",
        "knowledge-update": "update",
        "single-session-preference": "preference",
    }
    if not abstention and category not in names:
        raise ValueError(f"Unsupported LongMemEval category: {category!r}")
    name = "abstention" if abstention else names[category]
    raw = _judge(
        judge,
        _render_literal(
            _template(f"longmemeval_judge_{name}.txt"),
            QUESTION=question["question"],
            GOLD_ANSWER=_text(question.get("answer")),
            GENERATED_ANSWER=prediction,
        ),
    )
    # The official released evaluator uses this substring rule, including for
    # explanatory responses. Keep it explicit instead of silently substituting EM.
    score = float("yes" in raw.lower())
    return {
        "score": score,
        "metric": "longmemeval_official_judge_accuracy",
        "details": {
            "protocol": SCORE_PROTOCOLS["longmemeval"],
            "judge_response": raw,
            "category": category,
            "abstention": abstention,
            "official_answer_pipeline_comparable": False,
            "label_policy": "official_yes_substring",
        },
    }


def _kendall_tau_b(ref_order: list[int], pred_order: list[int]) -> float:
    n = len(ref_order)
    if n < 2:
        return 1.0
    concordant = discordant = ties_ref = ties_pred = 0
    for i in range(n):
        for j in range(i + 1, n):
            ref_diff, pred_diff = ref_order[i] - ref_order[j], pred_order[i] - pred_order[j]
            if ref_diff == 0:
                ties_ref += 1
            if pred_diff == 0:
                ties_pred += 1
            if ref_diff and pred_diff:
                if (ref_diff > 0) == (pred_diff > 0):
                    concordant += 1
                else:
                    discordant += 1
    pairs = n * (n - 1) // 2
    denominator = math.sqrt((pairs - ties_ref) * (pairs - ties_pred))
    return (concordant - discordant) / denominator if denominator else 0.0


def _beam_score(question: dict, prediction: str, judge: Judge | None) -> dict:
    metadata = question.get("metadata") or {}
    rubrics = metadata.get("rubric") or question.get("rubric")
    if (
        not isinstance(rubrics, list)
        or not rubrics
        or not all(isinstance(r, str) and r.strip() for r in rubrics)
    ):
        raise ValueError(
            "BEAM scoring requires the original non-empty rubric list in metadata.rubric"
        )
    category = str(question.get("category") or "")
    details = {
        "protocol": SCORE_PROTOCOLS["beam"],
        "officially_comparable": False,
        "category": category,
    }
    system = "You are a precise and fair evaluation judge."
    if category == "event_ordering":
        listing = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(rubrics))
        raw = _judge(
            judge,
            _template("beam_ordering_judge.txt").format(
                question=question["question"],
                reference_ordering=listing,
                response=prediction,
            ),
            system=system,
        )
        entries = _json_object(raw).get("positions")
        if not isinstance(entries, list):
            raise ValueError("BEAM ordering judge must return a positions list")
        detected = []
        for entry in entries[: len(rubrics)]:
            position = entry.get("position", -1) if isinstance(entry, dict) else None
            if type(position) is not int or (position != -1 and position < 1):
                raise ValueError("BEAM event positions must be positive integers or -1")
            detected.append(position)
        detected += [-1] * (len(rubrics) - len(detected))
        found = [(i, position) for i, position in enumerate(detected) if position > 0]
        coverage = len(found) / len(rubrics)
        tau_b = None
        if len(found) < 2:
            score = coverage * 0.5
        else:
            tau_b = _kendall_tau_b([i for i, _ in found], [p for _, p in found])
            score = max(0.0, (tau_b + 1.0) / 2.0) * coverage
        details.update(
            scoring_method="normalized_kendall_tau_b_times_coverage",
            positions=detected,
            coverage=coverage,
            tau_b=tau_b,
            judge_response=raw,
        )
    else:
        item_scores = []
        for rubric in rubrics:
            raw = _judge(
                judge,
                _template("beam_rubric_judge.txt").format(
                    question=question["question"],
                    rubric_item=rubric,
                    response=prediction,
                ),
                system=system,
            )
            payload = _json_object(raw)
            value = payload.get("score")
            if isinstance(value, bool):
                raise ValueError("BEAM rubric score must be numeric, not boolean")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "BEAM rubric judge must provide a numeric score in [0, 1]"
                ) from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("BEAM rubric score is outside [0, 1]")
            binned = 1.0 if value >= 0.75 else 0.5 if value >= 0.25 else 0.0
            item_scores.append(
                {
                    "rubric_item": rubric,
                    "raw_score": value,
                    "score": binned,
                    "reason": _text(payload.get("reason")),
                    "judge_response": raw,
                }
            )
        score = sum(item["score"] for item in item_scores) / len(item_scores)
        details.update(scoring_method="per_rubric_item", rubric_item_scores=item_scores)
    return {"score": score, "metric": "beam_nugget_v2", "details": details}


def score_prediction(
    benchmark: str, question: dict, prediction: str, *, judge: Judge | None = None
) -> dict:
    """Return ``score`` in [0, 1], a named ``metric``, and auditable ``details``.

    Callers must retain raw model responses as well as parsed answers. PersonaMem
    accepts either a raw response or parsed label; raw text preserves diagnostics.
    Lexical metrics use each benchmark's answer projection; judges see full text.
    Missing/invalid judge responses raise
    instead of being counted as wrong. Category 5 is excluded before aggregation.
    """
    benchmark = _benchmark(benchmark)
    prediction = str(prediction or "")
    if benchmark == "personamem":
        choices = question.get("choices")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError("PersonaMem scoring requires four choices")
        index = question.get("correct_choice")
        if index is None:
            label = str(question.get("answer") or "").strip().strip("() ").lower()
            if not re.fullmatch("[a-d]", label):
                raise ValueError("PersonaMem requires a valid gold label or correct_choice")
            index = ord(label) - 97
        if type(index) is not int or not 0 <= index < 4:
            raise ValueError("PersonaMem correct_choice must be an integer in [0, 3]")
        gold = chr(97 + index)
        final_options, all_options = _persona_parts(prediction)
        correct = final_options == {gold} or all_options == {gold}
        return {
            "score": float(correct),
            "metric": "personamem_mcq_accuracy",
            "details": {
                "protocol": SCORE_PROTOCOLS[benchmark],
                "predicted_label": parse_answer(benchmark, prediction),
                "correct_choice": index,
                "final_option_set": sorted(final_options),
                "full_option_set": sorted(all_options),
            },
        }
    if not prediction.strip():
        raise ValueError("Prediction is empty; this question was not scored")
    if "answer" not in question or question["answer"] is None:
        raise ValueError(f"{benchmark} requires a gold answer for scoring")
    if benchmark == "locomo" and str(question.get("category")) == "5":
        raise ValueError(
            "LoCoMo category 5 is excluded from this evaluation; filter it before scoring"
        )
    lexical_answer = project_lexical_answer(benchmark, prediction)
    lexical_metrics = calculate_answer_metrics(str(question["answer"]), lexical_answer)
    if benchmark == "beam":
        result = _beam_score(question, prediction, judge)
    elif benchmark == "longmemeval":
        result = _longmemeval_judge(question, prediction, judge)
    else:
        raw = _judge(
            judge,
            _render_literal(
                _template("locomo_judge.txt"),
                QUESTION=question["question"],
                GOLD_ANSWER=_text(question["answer"]),
                GENERATED_ANSWER=prediction,
            ),
        )
        label = raw.strip().strip('"').upper()
        if label not in {"CORRECT", "WRONG"}:
            raise ValueError("LoCoMo judge must return CORRECT or WRONG; this question was not scored")
        result = {
            "score": float(label == "CORRECT"),
            "metric": "locomo_weavemem_judge_accuracy",
            "details": {
                "protocol": SCORE_PROTOCOLS[benchmark],
                "judge_response": raw,
                "secondary_token_set_f1": lexical_metrics["f1"],
                "secondary_metric_input": "final_answer",
                "secondary_metric_is_official": False,
            },
        }
    result["lexical_metrics"] = lexical_metrics
    result["details"]["lexical_projection"] = {
        "locomo": "single_final_answer_marker",
        "longmemeval": "last_final_answer_marker",
        "beam": "full_response_without_truncation",
    }[benchmark]
    return result
