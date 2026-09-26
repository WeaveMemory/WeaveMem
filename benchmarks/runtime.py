
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()
_MODEL_ENV = {
    "extraction": "WEAVE_MEM_EXTRACTION_MODEL",
    "embedding": "WEAVE_MEM_EMBEDDING_MODEL",
    "rerank": "WEAVE_MEM_RERANK_MODEL",
}
_OVERRIDES = (
    "MID_MEMORIES_OVERRIDE_PATH",
    "LONG_RELATIONS_OVERRIDE_PATH",
    "MID_RELATIONS_OVERRIDE_PATH",
)
_SOURCE_OVERRIDE_ENV = (
    "WEAVE_MEM_MID_MEMORIES_PATH",
    "WEAVE_MEM_LONG_RELATIONS_PATH",
    "WEAVE_MEM_MID_RELATIONS_PATH",
    "WEAVE_MEM_SOURCE_CORPUS_PATH",
)


class BenchmarkRuntimeError(RuntimeError):
    """An execution failure safe to include in a public benchmark run log."""


def _conversation_id(conversation: dict) -> str:
    value = (
        conversation.get("sample_id")
        or conversation.get("conversation_id")
        or conversation.get("id")
    )
    if value is None or not str(value).strip():
        raise BenchmarkRuntimeError("Conversation is missing its stable identifier.")
    return str(value)


def _memory_dir(run_dir: Path, conversation_id: str) -> Path:
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:20]
    return run_dir / "memory" / digest


def _read_rows(path: Path, label: str) -> list[dict]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BenchmarkRuntimeError(f"{label} is missing or is not valid JSON.") from None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise BenchmarkRuntimeError(f"{label} must contain a JSON array of records.")
    return rows


def _valid_embedding(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
        and any(v != 0 for v in value)
    )


def _attach_mention_span(
    destination: dict, record: dict, times: dict, *, parent: dict | None = None
) -> None:
    """Persona timestamps describe source order, not inferred real event dates."""
    if parent is None:
        ids = record.get("chat_ids") or []
    else:
        evidence = record.get("sourceEvidence", record.get("source_evidence"))
        if evidence is not None and not isinstance(evidence, list):
            raise BenchmarkRuntimeError("Persona child sourceEvidence must be an array.")
        ids = [
            item.get("chatId") or item.get("chat_id")
            for item in (evidence or [])
            if isinstance(item, dict)
        ]
        ids = [value for value in ids if value]
        if evidence and not ids:
            raise BenchmarkRuntimeError(
                "Persona child sourceEvidence has no usable source turn IDs."
            )
        if not ids:
            ids = (
                record.get("sourceChatIds")
                or record.get("source_chat_ids")
                or record.get("chat_ids")
                or []
            )
    if not isinstance(ids, (list, tuple)):
        raise BenchmarkRuntimeError("Source mentions must use a list of source turn IDs.")
    resolved = []
    for value in ids:
        key = str(value)
        if key not in times and ":" not in key:
            parent_ids = {str(item) for item in (parent or record).get("chat_ids", [])}
            aliases = [
                item
                for item in times
                if item.rsplit(":", 1)[-1] == key and (not parent_ids or item in parent_ids)
            ]
            if len(aliases) == 1:
                key = aliases[0]
        if key not in times:
            raise BenchmarkRuntimeError(
                "Selected source evidence is outside the visible Persona prefix."
            )
        resolved.append(times[key])
    if not resolved:
        raise BenchmarkRuntimeError("Persona answer memory has no source-mention timestamps.")
    destination["synthetic_source_mention_start"] = min(resolved)
    destination["synthetic_source_mention_end"] = max(resolved)


@contextmanager
def _quiet_service_logs():
    # Provider exceptions may embed endpoint details that should not land in
    # public artifacts; the wrapper raises a fixed, actionable failure instead.
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _load_live_core(profile: dict, run_dir: Path) -> dict:
    dotenv = importlib.import_module("dotenv")
    dotenv.load_dotenv(ROOT / ".env", override=False)
    environment = profile.get("environment", {})
    if not isinstance(environment, dict):
        raise BenchmarkRuntimeError("Profile environment must be a mapping.")
    from benchmarks.run import ENVIRONMENT_KEYS

    for name in ENVIRONMENT_KEYS | {"GRAPH_NAVIGATION_MODEL", "GRAPH_PATH_MODEL"}:
        os.environ.pop(name, None)
    for name, value in environment.items():
        if not isinstance(name, str) or not (
            name.startswith("WEAVE_MEM_") or name in {"GRAPH_NAVIGATION_MODEL", "GRAPH_PATH_MODEL"}
        ):
            raise BenchmarkRuntimeError("Profile environment contains an unsupported setting.")
        os.environ[name] = str(value) if value is not None else ""
    for role, name in _MODEL_ENV.items():
        model = profile.get("models", {}).get(role)
        if not isinstance(model, str) or not model.strip():
            raise BenchmarkRuntimeError(f"Profile must explicitly name the {role} model.")
        os.environ[name] = model.strip()
    extraction = profile["models"]["extraction"].strip()
    # The dedicated extraction provider is registered under this name. A stale
    # local value must not silently reroute the configured model to CHAT.
    os.environ["WEAVE_MEM_GLM_MODEL"] = extraction
    for name in ("GRAPH_NAVIGATION_MODEL", "GRAPH_PATH_MODEL"):
        os.environ[name] = str(environment.get(name) or extraction)
    # Never let local memory paths enter a benchmark, even during core import.
    for name in _SOURCE_OVERRIDE_ENV:
        os.environ[name] = ""
    os.environ["WEAVE_MEM_DATA_DIR"] = str(run_dir / "memory")
    os.environ["WEAVE_MEM_MID_ENRICHED_EMBEDDINGS_PATH"] = str(
        run_dir / "memory" / "mid_enriched_embeddings.json"
    )
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    config = importlib.import_module("config")
    # A notebook or another sequential run may have imported the core already.
    importlib.reload(config)
    providers = importlib.import_module("llm.providers")
    importlib.reload(providers)
    chat = importlib.import_module("llm.client")
    chat._clients.clear()
    encoder = importlib.import_module("embedding.encoder")
    encoder._client = None
    retrieval = importlib.import_module("retrieval")
    retrieval.clear_retrieval_cache()
    return {
        "config": config,
        "ingest_corpus": importlib.import_module("ingest").ingest_corpus,
        "retrieval": retrieval,
    }


def _environment_pair(prefix: str) -> tuple[str, str] | None:
    key = os.getenv(f"{prefix}_API_KEY", "").strip()
    url = os.getenv(f"{prefix}_BASE_URL", "").strip()
    if not key and not url:
        return None
    if not key or not url:
        raise BenchmarkRuntimeError(
            f"Set both {prefix}_API_KEY and {prefix}_BASE_URL for this role."
        )
    return key, url


class LiveBackend:
    """Sequential, conversation-scoped execution using the actual WeaveMem core."""

    synthetic = False

    def __init__(self, profile: dict, run_dir: Path):
        self.profile = deepcopy(profile)
        self.run_dir = Path(run_dir).resolve()
        self._scopes: dict[str, Path] = {}
        self._current_id: str | None = None
        self._clients: dict[str, object] = {}
        self._answer_source_cache: tuple[str, dict] | None = None
        try:
            with _quiet_service_logs():
                self._core = _load_live_core(self.profile, self.run_dir)
        except BenchmarkRuntimeError:
            raise
        except Exception:
            raise BenchmarkRuntimeError(
                "Could not initialize the live backend; check dependencies and profile settings."
            ) from None

    @contextmanager
    def _scope(self, conversation_id: str, corpus_path: Path):
        config = self._core["config"]
        memory = _memory_dir(self.run_dir, conversation_id)
        values = {
            "DATA_DIR": str(memory),
            "MID_ENRICHED_EMBEDDINGS_PATH": str(memory / "mid_enriched_embeddings.json"),
            "SOURCE_CORPUS_PATH": str(corpus_path.resolve()),
            "INGEST_PROGRESS_PATH": str(memory / "progress_ingest.json"),
        }
        values.update({name: None for name in _OVERRIDES})
        values.update({name: None for name in vars(config) if name.endswith("_OVERRIDE_PATH")})
        previous = {name: getattr(config, name, _MISSING) for name in values}
        for name, value in values.items():
            setattr(config, name, value)
        try:
            self._core["retrieval"].clear_retrieval_cache()
            yield memory
        finally:
            for name, value in previous.items():
                if value is _MISSING:
                    delattr(config, name)
                else:
                    setattr(config, name, value)
            self._core["retrieval"].clear_retrieval_cache()

    def _validate_memories(self, memory: Path, conversation_id: str) -> dict:
        mids = _read_rows(memory / "mid_memories.json", "Mid-memory output")
        if not mids or any(row.get("conversation_id") != conversation_id for row in mids):
            raise BenchmarkRuntimeError(
                "Memory construction produced no valid conversation-scoped mids."
            )
        ids = [str(row.get("id") or "") for row in mids]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise BenchmarkRuntimeError(
                "Mid-memory output contains missing or duplicate identifiers."
            )
        if any(not _valid_embedding(row.get("embedding")) for row in mids):
            raise BenchmarkRuntimeError(
                "Mid embeddings are missing; lexical fallback is not a complete live build."
            )
        sidecar = _read_rows(memory / "mid_enriched_embeddings.json", "Enriched-embedding output")
        by_id = {str(row.get("id") or ""): row for row in sidecar}
        model = self._core["config"].EMBEDDING_MODEL
        if (
            len(by_id) != len(sidecar)
            or set(by_id) != set(ids)
            or any(
                not _valid_embedding(row.get("embedding")) or row.get("embedding_model") != model
                for row in sidecar
            )
        ):
            raise BenchmarkRuntimeError(
                "Enriched embeddings are incomplete or use the wrong embedding model."
            )
        for table in ("long_core", "long_episodic", "long_knowledge"):
            path = memory / f"{table}.json"
            if path.exists():
                rows = _read_rows(path, "Typed-memory output")
                if any(not _valid_embedding(row.get("embedding")) for row in rows):
                    raise BenchmarkRuntimeError(
                        "Typed-memory embeddings are incomplete; rebuild before evaluating."
                    )
        if self.profile.get("build"):
            try:
                metrics = json.loads((memory / "relation_graph_metrics.json").read_text())
            except (OSError, ValueError):
                raise BenchmarkRuntimeError(
                    "Complete graph metrics are missing; finish build before retrieval."
                ) from None
            judged, candidates = (
                metrics.get("judged_candidate_count"),
                metrics.get("candidate_count"),
            )
            failed = metrics.get("failure_count")
            if (
                type(candidates) is not int
                or candidates < 0
                or type(judged) is not int
                or judged != candidates
                or failed != 0
                or metrics.get("secondary_review_count") != 0
                or metrics.get("dry_run") is True
            ):
                raise BenchmarkRuntimeError(
                    "The relation graph is incomplete or does not use the configured proposer-only protocol."
                )
            mid_ids = set(ids)
            long_ids = {
                str(row["id"])
                for kind in ("core", "episodic", "knowledge")
                for row in _read_rows(memory / f"long_{kind}.json", "Typed-memory output")
            }
            for table, allowed, count_key in (
                ("mid_relations", mid_ids, "projected_mid_relation_count"),
                ("long_relations", long_ids, "accepted_long_relation_count"),
            ):
                edges = _read_rows(memory / f"{table}.json", "Relation output")
                if len(edges) != metrics.get(count_key) or any(
                    edge.get("source_id") not in allowed
                    or edge.get("target_id") not in allowed
                    or edge.get("conversation_id") != conversation_id
                    for edge in edges
                ):
                    raise BenchmarkRuntimeError(
                        "The relation graph count or conversation boundary is invalid."
                    )
        return {"mid_count": len(mids), "enriched_embedding_count": len(sidecar)}

    def build(self, conversation: dict, corpus_path: Path) -> dict:
        cid = _conversation_id(conversation)
        corpus_path = Path(corpus_path).resolve()
        if not corpus_path.is_file():
            raise BenchmarkRuntimeError("Prepared conversation corpus is missing.")
        try:
            with self._scope(cid, corpus_path) as memory, _quiet_service_logs():
                stats = dict(
                    self._core["ingest_corpus"](
                        str(corpus_path), sample=cid, strict=True, **self.profile.get("build", {})
                    )
                )
                stats.update(self._validate_memories(memory, cid))
        except BenchmarkRuntimeError:
            raise
        except Exception:
            raise BenchmarkRuntimeError(
                "Memory construction failed; check model access, profile settings, and prepared input."
            ) from None
        self._scopes[cid] = corpus_path
        self._current_id = cid
        return {**stats, "synthetic": False}

    def retrieve(self, question: dict) -> dict:
        cid = str(question.get("conversation_id") or self._current_id or "")
        corpus_path = self._scopes.get(cid)
        restored = corpus_path is None
        if restored:
            corpus_path = self.run_dir / "prepared" / "corpus.json"
            samples = _read_rows(corpus_path, "Prepared conversation corpus")
            if not cid or sum(_conversation_id(sample) == cid for sample in samples) != 1:
                raise BenchmarkRuntimeError("Question does not identify one prepared conversation.")
        query = question.get("retrieval_query") or question.get("question")
        if not isinstance(query, str) or not query.strip():
            raise BenchmarkRuntimeError("Question is missing its public retrieval text.")
        try:
            with self._scope(cid, corpus_path) as memory, _quiet_service_logs():
                if restored:
                    self._validate_memories(memory, cid)
                result = self._core["retrieval"].search_memory_result(query, conversation_id=cid)
        except BenchmarkRuntimeError:
            raise
        except Exception:
            raise BenchmarkRuntimeError(
                "Live retrieval failed; check model access and the conversation memory artifacts."
            ) from None
        self._scopes[cid] = corpus_path
        self._current_id = cid
        memories = self._project_answer_sources(result.memories, result.trace, cid, corpus_path)
        return {"memories": memories, "trace": result.trace}

    def _answer_sources(self, cid: str, corpus_path: Path) -> dict:
        """Recover only selected-record fields; gold QA never enters this boundary."""
        if self._answer_source_cache is not None and self._answer_source_cache[0] == cid:
            return self._answer_source_cache[1]
        memory = _memory_dir(self.run_dir, cid)
        records = {}
        for table in (
            "mid_memories",
            "long_core",
            "long_episodic",
            "long_knowledge",
        ):
            path = memory / f"{table}.json"
            if path.exists():
                for row in _read_rows(path, "Answer source memory"):
                    records[str(row["id"])] = {
                        key: value
                        for key, value in row.items()
                        if key not in {"embedding", "embedding_model", "embedding_text_sha256"}
                    }
        turns = {}
        for sample in _read_rows(corpus_path, "Prepared source corpus"):
            if _conversation_id(sample) != cid:
                continue
            for session, rows in sample["conversation"].items():
                if re.fullmatch(r"session_\d+", session) and isinstance(rows, list):
                    for turn in rows:
                        if turn.get("synthetic_source_mention_timestamp"):
                            turns[str(turn["dia_id"])] = turn["synthetic_source_mention_timestamp"]
        result = {"records": records, "turns": turns}
        self._answer_source_cache = (cid, result)
        return result

    def _project_answer_sources(
        self, memories: list[dict], trace: dict, cid: str, corpus_path: Path
    ) -> list[dict]:
        benchmark = self.profile.get("benchmark")
        if benchmark not in {"personamem", "beam"}:
            return memories
        sources = self._answer_sources(cid, corpus_path)
        projected = deepcopy(memories)
        for mid in projected:
            original = sources["records"].get(str(mid.get("id")))
            if original is None:
                raise BenchmarkRuntimeError("Selected Mid is missing from the scoped memory store.")
            mid["source_record"] = deepcopy(original)
            for child in mid.get("query_relevant_child_facts") or []:
                raw = sources["records"].get(str(child.get("id")))
                if raw is None:
                    raise BenchmarkRuntimeError(
                        "Selected child is missing from the scoped memory store."
                    )
                child["source_record"] = deepcopy(raw)
                if benchmark == "personamem":
                    _attach_mention_span(child, raw, sources["turns"], parent=original)
            if benchmark == "personamem":
                _attach_mention_span(mid, original, sources["turns"])
        if benchmark == "beam" and projected:
            projected[0]["selected_graph_relations"] = deepcopy(
                trace.get("selected_graph_relations", [])
            )
        return projected

    def _role_pair(self, role: str) -> tuple[str, str]:
        if role == "judge":
            pair = _environment_pair("BENCHMARK_JUDGE")
            if pair is not None:
                return pair
        pair = _environment_pair("BENCHMARK_ANSWER") or _environment_pair("CHAT")
        if pair is None:
            raise BenchmarkRuntimeError(
                "Set BENCHMARK_ANSWER_API_KEY and BENCHMARK_ANSWER_BASE_URL (or the CHAT pair). "
                "Embedding credentials are not reused for benchmark answers."
            )
        return pair

    def complete(self, role: str, messages: list) -> dict:
        if role not in {"answer", "judge"}:
            raise BenchmarkRuntimeError("Completion role must be answer or judge.")
        model = self.profile.get("models", {}).get(role)
        if not isinstance(model, str) or not model.strip():
            raise BenchmarkRuntimeError(f"Profile must explicitly name the {role} model.")
        key, url = self._role_pair(role)
        options = self.profile.get(role, {})
        try:
            with _quiet_service_logs():
                if role not in self._clients:
                    factory = importlib.import_module("openai").OpenAI
                    self._clients[role] = factory(
                        api_key=key, base_url=url, timeout=120.0, max_retries=0
                    )
                request = {
                    "model": model.strip(),
                    "messages": messages,
                    "max_tokens": options.get("max_tokens", 2048),
                }
                if options.get("temperature") is not None:
                    request["temperature"] = options["temperature"]
                if options.get("extra_body"):
                    request["extra_body"] = options["extra_body"]
                result = self._clients[role].chat.completions.create(**request)
                content = result.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty model completion")
                usage = {}
                for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = getattr(getattr(result, "usage", None), name, None)
                    if isinstance(value, int) and not isinstance(value, bool):
                        usage[name] = value
        except Exception:
            raise BenchmarkRuntimeError(
                f"The {role} model request failed; check its configured model, endpoint, and credentials."
            ) from None
        return {"response": content, "usage": usage}


class SmokeBackend:
    """Synthetic wiring check: session text plus lexical lookup, with no model calls."""

    synthetic = True

    def __init__(self, profile: dict, run_dir: Path):
        self.profile = deepcopy(profile)
        self.run_dir = Path(run_dir).resolve()
        self._memories: dict[str, list[dict]] = {}
        self._current_id: str | None = None

    def build(self, conversation: dict, corpus_path: Path) -> dict:
        cid = _conversation_id(conversation)
        samples = _read_rows(Path(corpus_path), "Prepared conversation corpus")
        matches = [sample for sample in samples if _conversation_id(sample) == cid]
        if len(matches) != 1:
            raise BenchmarkRuntimeError(
                "Prepared corpus must contain this conversation exactly once."
            )
        source = matches[0].get("conversation")
        if not isinstance(source, dict):
            raise BenchmarkRuntimeError("Prepared conversation has no session-text mapping.")
        memories = []
        sessions = sorted(
            (key for key in source if re.fullmatch(r"session_\d+", key)),
            key=lambda key: int(key.split("_")[-1]),
        )
        for session in sessions:
            turns = source[session]
            if not isinstance(turns, list):
                raise BenchmarkRuntimeError("Prepared session must contain a list of turns.")
            text = "\n".join(
                f"{turn.get('speaker', '')}: {turn.get('text', '')}"
                for turn in turns
                if isinstance(turn, dict)
            )
            if text.strip():
                session_id = "D" + session.split("_", 1)[1]
                session_date = source.get(f"{session}_date_time", "")
                memories.append(
                    {
                        "id": f"smoke-{hashlib.sha256((cid + session).encode()).hexdigest()[:20]}",
                        "conversation_id": cid,
                        "topic_subject": session,
                        "summary": text,
                        "session_id": session_id,
                        "session_date": session_date,
                        "chat_ids": [str(turn.get("dia_id")) for turn in turns],
                        "synthetic_source_mention_start": min(
                            (
                                turn["synthetic_source_mention_timestamp"]
                                for turn in turns
                                if turn.get("synthetic_source_mention_timestamp")
                            ),
                            default="",
                        ),
                        "synthetic_source_mention_end": max(
                            (
                                turn["synthetic_source_mention_timestamp"]
                                for turn in turns
                                if turn.get("synthetic_source_mention_timestamp")
                            ),
                            default="",
                        ),
                        "source_record": {
                            "topic_subject": session,
                            "summary": text,
                            "session_id": session_id,
                            "session_date": session_date,
                            "dialogue_phase": "synthetic_smoke",
                            "user_attitude": "synthetic smoke fixture",
                        },
                        "synthetic": True,
                    }
                )
        if not memories:
            raise BenchmarkRuntimeError("Prepared conversation contains no usable session text.")
        memory = _memory_dir(self.run_dir, cid)
        memory.mkdir(parents=True, exist_ok=True)
        (memory / "mid_memories.json").write_text(
            json.dumps(memories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self._memories[cid] = memories
        self._current_id = cid
        return {"mid_count": len(memories), "synthetic": True, "backend": "SMOKE_ONLY"}

    def retrieve(self, question: dict) -> dict:
        cid = str(question.get("conversation_id") or self._current_id or "")
        if cid not in self._memories:
            rows = _read_rows(
                _memory_dir(self.run_dir, cid) / "mid_memories.json", "Synthetic mid-memory output"
            )
            if (
                not cid
                or not rows
                or any(
                    row.get("conversation_id") != cid
                    or row.get("synthetic") is not True
                    or not isinstance(row.get("summary"), str)
                    for row in rows
                )
            ):
                raise BenchmarkRuntimeError(
                    "Smoke retrieval requires matching synthetic memory artifacts."
                )
            self._memories[cid] = rows
        self._current_id = cid
        query = question.get("retrieval_query") or question.get("question")
        if not isinstance(query, str):
            raise BenchmarkRuntimeError("Question is missing its public retrieval text.")
        terms = set(re.findall(r"\w+", query.lower()))
        ranked = sorted(
            self._memories[cid],
            key=lambda row: -len(terms & set(re.findall(r"\w+", row["summary"].lower()))),
        )
        return {
            "memories": deepcopy(ranked[:5]),
            "trace": {
                "synthetic": True,
                "backend": "SMOKE_ONLY",
                "retrieval": "lexical",
                "query": query,
            },
        }

    def complete(self, role: str, messages: list) -> dict:
        if role not in {"answer", "judge"}:
            raise BenchmarkRuntimeError("Completion role must be answer or judge.")
        return {
            "response": "SMOKE_ONLY: synthetic pipeline check; no model was called.",
            "usage": {},
        }


__all__ = ["BenchmarkRuntimeError", "LiveBackend", "SmokeBackend"]
