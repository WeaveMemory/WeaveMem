

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import config
import embedding
import jsonio
from checkpoint import Checkpoint
from data import conversations as corpus
from enriched_embeddings import build_enriched_mid_embeddings
from generation import (
    extract_mid_memories,
    extract_typed_long_memories,
    resolve_typed_long_memory_types,
)
from generation.embedding_text import long_retrieval_content
from generation.util import new_id, now
from relation_graph import build_relation_graph


_TYPES = ("core", "episodic", "knowledge")
_TABLES = ("mid_memories", *(f"long_{kind}" for kind in _TYPES))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _read(table: str) -> list[dict]:
    rows = jsonio.read_json(str(Path(config.DATA_DIR) / f"{table}.json"), default=[])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{table} must contain an array of objects")
    return rows


def _write(table: str, rows: list[dict]) -> None:
    jsonio.atomic_write_json(str(Path(config.DATA_DIR) / f"{table}.json"), rows)


def _valid_embedding(vector: object) -> bool:
    return (
        isinstance(vector, list)
        and bool(vector)
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in vector
        )
        and any(value != 0 for value in vector)
    )


def _attach_embeddings(records: list[dict], text_of, label: str) -> None:
    if not records:
        return
    vectors = embedding.embed_texts([text_of(record) for record in records])
    if (
        not isinstance(vectors, list)
        or len(vectors) != len(records)
        or any(not _valid_embedding(vector) for vector in vectors)
        or len({len(vector) for vector in vectors}) != 1
    ):
        raise RuntimeError(f"{label} embeddings are missing, partial or invalid")
    for record, vector in zip(records, vectors):
        record["embedding"] = vector


def _normalize_mid_source_chat_ids(mids: list[dict], *, session_id: str, turns: list[dict]) -> None:
    available = {str(turn["dia_id"]).strip() for turn in turns}
    aliases = {}
    for value in available:
        prefix, separator, suffix = value.partition(":")
        if separator and prefix == session_id and suffix.isdigit():
            alias = str(int(suffix))
            aliases[alias] = value if alias not in aliases else None
    for mid in mids:
        ids = mid.get("chat_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("Every Mid must reference non-empty source chat_ids")
        resolved = []
        for value in ids:
            value = str(value).strip()
            candidate = (
                value
                if value in available
                else aliases.get(str(int(value)))
                if value.isdigit()
                else None
            )
            if candidate is None:
                raise ValueError("Mid chat_ids must resolve inside the current source session")
            resolved.append(candidate)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Mid chat_ids contain duplicate source turns")
        mid["chat_ids"] = resolved


def _system_persona_mids(sample: dict, session: dict) -> list[dict]:
    if session["session_id"] != "D0":
        return []
    turns = [turn for turn in session["turns"] if turn.get("source_role") == "system"]
    if not turns:
        return []
    if len(turns) != 1:
        raise ValueError("The initial PersonaMem session must have one system persona")
    turn = turns[0]
    owner = corpus.participants(sample)[0]
    text = str(turn.get("text") or "").strip()
    if not text:
        raise ValueError("Initial system persona text is empty")
    return [
        {
            "id": new_id(),
            "user_id": owner,
            "conversation_id": sample["sample_id"],
            "session_id": "D0",
            "session_date": session["session_date"],
            "chat_ids": [turn["dia_id"]],
            "topic_subject": f"Initial system persona profile for {owner}",
            "summary": text,
            "tags": ["initial persona", "system persona", owner],
            "confidence": 1.0,
            "created_at": now(),
            "extraction_scope": "system_persona_baseline",
            "source_role": "system",
        }
    ]


def _extract_session(sample: dict, session: dict, memory_types: tuple, mid_prompt: str) -> dict:
    cid, sid = sample["sample_id"], session["session_id"]
    mids = extract_mid_memories(
        session_history=corpus.format_session_history(
            session["turns"], start_time=session["session_date"]
        ),
        conversation_id=cid,
        session_id=sid,
        session_date=session["session_date"],
        participants=corpus.participants(sample),
        require_valid_response=True,
        prompt_name=mid_prompt,
    )
    mids.extend(_system_persona_mids(sample, session))
    _normalize_mid_source_chat_ids(mids, session_id=sid, turns=session["turns"])
    # The archived base Mid vector uses topic + summary (the enriched sidecar is
    # separately built from the formal Mid/child projection).
    _attach_embeddings(
        mids,
        lambda row: f"{row.get('topic_subject') or ''} {row.get('summary') or ''}".strip(),
        "Mid",
    )
    tables = {table: [] for table in _TABLES}
    tables["mid_memories"] = mids
    for mid in mids:
        selected = mid["chat_ids"]
        dialogue = corpus.format_dialogue_for_ids(
            session["turns"], selected, start_time=session["session_date"]
        )
        if not dialogue.strip():
            raise ValueError("Mid selected no raw dialogue")
        # Only this two-field provenance anchor reaches the extraction function;
        # neither value is used to render the model prompt.
        extracted = extract_typed_long_memories(
            {"id": mid["id"], "user_id": mid["user_id"]},
            dialogue,
            require_valid_response=True,
            memory_types=memory_types,
        )
        for kind in _TYPES:
            for raw in extracted[kind]:
                evidence = raw.get("sourceEvidence")
                if (
                    not isinstance(evidence, list)
                    or not evidence
                    or any(
                        not isinstance(item, dict)
                        or str(item.get("chatId") or "").strip() not in selected
                        for item in evidence
                    )
                ):
                    raise ValueError(
                        "Long sourceEvidence must reference only its parent Mid's raw turns"
                    )
                tables[f"long_{kind}"].append(
                    {
                        **raw,
                        "conversation_id": cid,
                        "session_id": sid,
                        "session_date": session["session_date"],
                        "user_id": mid["user_id"],
                        "mid_id": mid["id"],
                        "type": kind,
                        "extraction_scope": "mid_source_original_dialogue_only",
                        "source_chat_ids": list(selected),
                    }
                )
    longs = [row for kind in _TYPES for row in tables[f"long_{kind}"]]
    _attach_embeddings(longs, long_retrieval_content, "Typed Long")
    for row in longs:
        row["embedding_model"] = config.EMBEDDING_MODEL
        row["embedding_text_sha256"] = hashlib.sha256(
            long_retrieval_content(row).encode()
        ).hexdigest()
    return tables


def _session_rows(cid: str, sid: str) -> dict:
    return {
        table: [
            row
            for row in _read(table)
            if row.get("conversation_id") == cid and row.get("session_id") == sid
        ]
        for table in _TABLES
    }


def _persist_session(cid: str, sid: str, tables: dict) -> None:
    for table in _TABLES:
        kept = [
            row
            for row in _read(table)
            if not (row.get("conversation_id") == cid and row.get("session_id") == sid)
        ]
        # Preserve complete generated Mid state and graph evidence fields; the
        # historical generic store's column projection drops several of them.
        _write(table, [*kept, *tables[table]])


def _build_contract(samples: list[dict], settings: dict) -> None:
    path = Path(config.DATA_DIR) / "formal_build_contract.json"
    contract = jsonio.read_json(str(path), default={}) or {}
    if contract and contract.get("settings") != settings:
        raise ValueError("Formal build settings changed; use a new memory directory")
    sources = contract.get("source_sha256", {})
    for sample in samples:
        cid = sample["sample_id"]
        fingerprint = _digest(sample)
        if cid in sources and sources[cid] != fingerprint:
            raise ValueError("Source conversation changed; use a new memory directory")
        sources[cid] = fingerprint
    jsonio.atomic_write_json(
        str(path), {"schema_version": 1, "settings": settings, "source_sha256": sources}
    )


def ingest_corpus(
    path: str,
    *,
    sample: str | None = None,
    session: int | None = None,
    restart: bool = False,
    strict: bool = True,
    memory_types: tuple[str, ...] = _TYPES,
    graph_candidate_top_n: int = 10,
    graph_min_confidence: float = 0.95,
    graph_bm25_lemmatization: bool = False,
    mid_prompt: str = "mid_extraction",
) -> dict:
    """Construct selected conversations and publish complete, scoped memory graphs.

    ``memory_types`` defaults to the formal three typed long memories and may be
    narrowed by a benchmark configuration. ``mid_prompt='mid_extraction_beam'``
    preserves BEAM's additional dialogue state. Raw session timestamps are never
    invented. A session checkpoint covers validated Mid/Long vectors and persisted
    rows; graph and enriched sidecar completion are independent, resumable final
    steps.
    """
    if not strict:
        raise ValueError("Formal construction requires strict=True")
    if type(graph_candidate_top_n) is not int or graph_candidate_top_n <= 0:
        raise ValueError("graph_candidate_top_n must be a positive integer")
    if (
        isinstance(graph_min_confidence, bool)
        or not isinstance(graph_min_confidence, (int, float))
        or not math.isfinite(graph_min_confidence)
        or not 0.95 <= graph_min_confidence <= 1
    ):
        raise ValueError("graph_min_confidence must be finite and in [0.95, 1]")
    if type(graph_bm25_lemmatization) is not bool:
        raise ValueError("graph_bm25_lemmatization must be boolean")
    if session is not None and sample is None:
        raise ValueError("A single session requires sample")
    if mid_prompt not in {"mid_extraction", "mid_extraction_beam"}:
        raise ValueError("Unknown formal Mid extraction prompt")
    memory_types = resolve_typed_long_memory_types(memory_types)
    samples = corpus.load_samples(path)
    if not isinstance(samples, list) or not samples:
        raise ValueError("The corpus must be a non-empty sample array")
    ids = [row.get("sample_id") for row in samples]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Conversation IDs must be non-empty and unique")
    if sample is not None:
        samples = [row for row in samples if row["sample_id"] == sample]
        if not samples:
            raise ValueError(f"sample not found: {sample}")
    if graph_bm25_lemmatization:
        import spacy

        # A requested formal tokenizer must not silently fall back to regex.
        spacy.load(config.BM25_LEMMATIZATION_MODEL)
    root = Path(config.DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    for table in _TABLES:
        if not (root / f"{table}.json").exists():
            _write(table, [])
    prompt_names = [
        mid_prompt,
        *(f"memory_extraction_{kind}" for kind in memory_types),
        "long_relation_projectable_pair",
    ]
    settings = {
        "model": config.EXTRACTION_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "memory_types": list(memory_types),
        "mid_prompt": mid_prompt,
        "graph_candidate_top_n": graph_candidate_top_n,
        "graph_min_confidence": graph_min_confidence,
        "graph_bm25_lemmatization": graph_bm25_lemmatization,
        "long_input": "selected_raw_dialogue_only",
        "secondary_review": False,
        "implementation_sha256": _digest(
            {
                path.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(
                    [
                        Path(__file__),
                        Path(__file__).parent / "relation_graph.py",
                        Path(__file__).parent / "enriched_embeddings.py",
                        Path(__file__).parent / "data/conversations.py",
                        *Path(__file__).parent.joinpath("generation").glob("*.py"),
                    ]
                )
            }
        ),
        "prompt_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / "llm/prompts" / f"{name}.txt").read_bytes()
            ).hexdigest()
            for name in prompt_names
        },
    }
    _build_contract(samples, settings)
    checkpoint = Checkpoint(config.INGEST_PROGRESS_PATH, "formal_memory_build")
    commits_path = root / "formal_session_commits.json"
    commits = jsonio.read_json(str(commits_path), default={}) or {}
    totals: Counter = Counter()
    for item in samples:
        cid = item["sample_id"]
        sessions = [
            row
            for row in corpus.iter_sessions(item)
            if session is None or row["session_id"] == f"D{session}"
        ]
        if not sessions:
            raise ValueError("The selected conversation has no matching sessions")
        for current in sessions:
            sid = current["session_id"]
            number = int(sid[1:])
            key = f"{cid}/{sid}"
            if not restart and checkpoint.done(cid, number):
                saved = _session_rows(cid, sid)
                if commits.get(key) != _digest(saved):
                    raise ValueError("Completed session rows changed or are incomplete")
                totals["cached_sessions"] += 1
                continue
            generated = _extract_session(item, current, memory_types, mid_prompt)
            _persist_session(cid, sid, generated)
            commits[key] = _digest(generated)
            jsonio.atomic_write_json(str(commits_path), commits)
            checkpoint.mark(cid, number)
            totals["built_sessions"] += 1
    mids = _read("mid_memories")
    longs = [row for kind in _TYPES for row in _read(f"long_{kind}")]
    _write(
        "long_embeddings",
        [
            {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "embedding_model": config.EMBEDDING_MODEL,
                "text_sha256": row["embedding_text_sha256"],
                "embedding": row["embedding"],
            }
            for row in longs
        ],
    )
    sidecar = build_enriched_mid_embeddings(prune_orphans=True, strict=True)
    graph = build_relation_graph(
        mids,
        longs,
        output_dir=root,
        candidate_top_n=graph_candidate_top_n,
        min_confidence=graph_min_confidence,
        bm25_lemmatization=graph_bm25_lemmatization,
    )
    return {
        **dict(totals),
        "mid": len(mids),
        "long": len(longs),
        **{kind: len(_read(f"long_{kind}")) for kind in _TYPES},
        "mid_rel": graph["projected_mid_relation_count"],
        "long_rel": graph["accepted_long_relation_count"],
        "enriched_mid_embeddings": sidecar["sidecar_rows"],
    }
