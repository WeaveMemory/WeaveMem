

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve .env relative to the project, regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env_str(name: str, default: str = "") -> str:
    """Treat an unset or blank optional setting as its documented default."""
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    """Read an integer setting; reject malformed values."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    """Read a finite floating-point setting from the environment."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not float("-inf") < value < float("inf"):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    return value


# Model roles. llm/providers.py resolves endpoints and request options.
EXTRACTION_MODEL = _env_str(
    "WEAVE_MEM_EXTRACTION_MODEL", "glm-5.2"
)  # memory extraction, relation judging, and graph navigation
GLM_MODEL = _env_str("WEAVE_MEM_GLM_MODEL", "glm-5.2")

# OpenAI-compatible endpoints, consumed by llm/providers.py.
OPENAI_BASE_URL = _env_str("OPENAI_BASE_URL", "")
OPENAI_API_KEY = _env_str("OPENAI_API_KEY") or None
# Chat defaults to the embedding endpoint and key.
CHAT_BASE_URL = _env_str("CHAT_BASE_URL", OPENAI_BASE_URL)
CHAT_API_KEY = _env_str("CHAT_API_KEY") or OPENAI_API_KEY
# Dedicated GLM route. The *2 settings take precedence over the primary values.
GLM_BASE_URL = _env_str("GLM_BASE_URL2") or _env_str("GLM_BASE_URL", CHAT_BASE_URL)
GLM_API_KEY = _env_str("GLM_API_KEY2") or _env_str("GLM_API_KEY") or CHAT_API_KEY

# Per-model request options, consumed by llm/providers.py.
# Disable Qwen reasoning output through extra_body.
ENABLE_THINKING = False
# GLM reasoning output is configured separately.
GLM_ENABLE_THINKING = _env_str("WEAVE_MEM_GLM_ENABLE_THINKING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Shared chat request policy.
LLM_MAX_TOKENS = _env_int("WEAVE_MEM_LLM_MAX_TOKENS", 20000)
LLM_TIMEOUT_SECONDS = _env_int("WEAVE_MEM_LLM_TIMEOUT_SECONDS", 400)
LLM_MAX_RETRIES = _env_int("WEAVE_MEM_LLM_MAX_RETRIES", 5)
LLM_RETRY_BACKOFF_SECONDS = _env_float("WEAVE_MEM_LLM_RETRY_BACKOFF_SECONDS", 2.0)

# Embedding: similarity is computed locally (brute-force cosine); only vectors are remote.
EMBEDDING_MODEL = _env_str("WEAVE_MEM_EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_BATCH_SIZE = 10  # inputs per embedding request
EMBEDDING_TIMEOUT_SECONDS = 30


# Dense and BM25 retrieval fusion.
BM25_K1 = 1.5  # BM25 term-frequency saturation
BM25_B = 0.75  # BM25 length normalization

BM25_LEMMATIZATION_ENABLED = _env_str("WEAVE_MEM_BM25_LEMMATIZATION", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
BM25_LEMMATIZATION_AUTO_DOWNLOAD = _env_str(
    "WEAVE_MEM_BM25_LEMMATIZATION_AUTO_DOWNLOAD", "0"
).strip().lower() not in {"0", "false", "no", "off"}
BM25_LEMMATIZATION_MODEL = _env_str("WEAVE_MEM_BM25_LEMMATIZATION_MODEL", "en_core_web_sm")
RRF_K = 60  # reciprocal-rank-fusion damping constant



RERANK_MODEL = _env_str("WEAVE_MEM_RERANK_MODEL", "qwen3-rerank")
RERANK_API_KEY = _env_str("RERANK_API_KEY") or OPENAI_API_KEY
RERANK_MAX_RETRIES = 3
RERANK_RETRY_BACKOFF_SECONDS = 1

# Maximum candidates per rerank request.
RERANK_BATCH_SIZE = 100


# Graph navigation.
GRAPH_AGENT_WORKERS = 3
GRAPH_NAVIGATION_MODEL = _env_str("GRAPH_NAVIGATION_MODEL", EXTRACTION_MODEL)
GRAPH_NAVIGATION_MAX_ATTEMPTS = 3
GRAPH_PATH_MODEL = _env_str("GRAPH_PATH_MODEL", EXTRACTION_MODEL)
GRAPH_PATH_MAX_ATTEMPTS = 3
GRAPH_MAX_PATHS_PER_SEED = _env_int("WEAVE_MEM_GRAPH_MAX_PATHS_PER_SEED", 5)
# Retry with wider candidate and seed pools only when the initial evidence is insufficient.
GRAPH_INITIAL_SEED_TOP_N = _env_int("WEAVE_MEM_GRAPH_INITIAL_SEED_TOP_N", 5)
GRAPH_INITIAL_RRF_CANDIDATE_TOP_N = _env_int("WEAVE_MEM_GRAPH_INITIAL_RRF_TOP_N", 20)
GRAPH_FALLBACK_SEED_TOP_N = _env_int("WEAVE_MEM_GRAPH_FALLBACK_SEED_TOP_N", 10)
GRAPH_FALLBACK_RRF_CANDIDATE_TOP_N = _env_int("WEAVE_MEM_GRAPH_FALLBACK_RRF_TOP_N", 50)
GRAPH_MAX_HOPS = _env_int("WEAVE_MEM_GRAPH_MAX_HOPS", 3)
# Admit whole paths while their union fits the unique-Mid budget.
GRAPH_COMPLETE_PATH_MAX_UNIQUE_NODES = _env_int("WEAVE_MEM_GRAPH_COMPLETE_PATH_BUDGET", 10)
# Attach relevant child facts under their owning Mid after navigation.
GRAPH_CHILD_FACT_MIN_SCORE = (
    None
    if _env_str("WEAVE_MEM_GRAPH_CHILD_FACT_MIN_SCORE").lower() == "none"
    else _env_float("WEAVE_MEM_GRAPH_CHILD_FACT_MIN_SCORE", 0.5)
)
GRAPH_PATH_WEIGHT_NODE = _env_float("WEAVE_MEM_PATH_W_NODE", 0.6)
GRAPH_PATH_WEIGHT_TYPE = _env_float("WEAVE_MEM_PATH_W_TYPE", 0.2)
GRAPH_PATH_WEIGHT_DESC = _env_float("WEAVE_MEM_PATH_W_DESC", 0.1)
GRAPH_PATH_WEIGHT_CONTROLLER = _env_float("WEAVE_MEM_PATH_W_CONTROLLER", 0.1)
_PATH_WEIGHTS = (
    GRAPH_PATH_WEIGHT_NODE,
    GRAPH_PATH_WEIGHT_TYPE,
    GRAPH_PATH_WEIGHT_DESC,
    GRAPH_PATH_WEIGHT_CONTROLLER,
)
if any(weight < 0 for weight in _PATH_WEIGHTS) or abs(sum(_PATH_WEIGHTS) - 1) > 1e-8:
    raise ValueError("Path weights must be nonnegative and sum to one")
GRAPH_CHILD_FACT_GLOBAL_TOP_N = _env_int("WEAVE_MEM_GRAPH_CHILD_FACT_TOP_N", 10)


def _validate_graph_parameters() -> None:
    """Reject an internally inconsistent graph configuration before any API call."""
    checks = (
        (GRAPH_MAX_HOPS >= 0, "H must be >= 0"),
        (GRAPH_MAX_PATHS_PER_SEED >= 1, "P must be >= 1"),
        (
            GRAPH_COMPLETE_PATH_MAX_UNIQUE_NODES >= 1,
            "B must be >= 1",
        ),
        (GRAPH_INITIAL_SEED_TOP_N >= 1, "K_i must be >= 1"),
        (
            GRAPH_INITIAL_RRF_CANDIDATE_TOP_N >= GRAPH_INITIAL_SEED_TOP_N,
            "C_i must be >= K_i",
        ),
        (
            GRAPH_FALLBACK_SEED_TOP_N > GRAPH_INITIAL_SEED_TOP_N,
            "K_f^seed must be > K_i",
        ),
        (
            GRAPH_FALLBACK_RRF_CANDIDATE_TOP_N >= GRAPH_FALLBACK_SEED_TOP_N,
            "C_f must be >= K_f^seed",
        ),
        (
            GRAPH_CHILD_FACT_GLOBAL_TOP_N >= 1,
            "K_fact must be >= 1",
        ),
        (
            GRAPH_CHILD_FACT_MIN_SCORE is None or 0.0 <= GRAPH_CHILD_FACT_MIN_SCORE <= 1.0,
            "tau_fact must be in [0, 1]",
        ),
    )
    errors = [message for valid, message in checks if not valid]
    if errors:
        raise ValueError("invalid graph retrieval configuration: " + "; ".join(errors))


_validate_graph_parameters()


# Memory tables and ingest checkpoints default to DATA_DIR.
DATA_DIR = _env_str("WEAVE_MEM_DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))
MID_MEMORIES_OVERRIDE_PATH = (
    os.path.abspath(os.path.expanduser(os.getenv("WEAVE_MEM_MID_MEMORIES_PATH", "")))
    if os.getenv("WEAVE_MEM_MID_MEMORIES_PATH", "").strip()
    else None
)
# Source corpus used for raw-turn quotes; see data/conversations.py for its schema.
SOURCE_CORPUS_PATH = (
    os.path.abspath(os.path.expanduser(os.getenv("WEAVE_MEM_SOURCE_CORPUS_PATH", "")))
    if os.getenv("WEAVE_MEM_SOURCE_CORPUS_PATH", "").strip()
    else None
)
# Optional graph overrides are included in retrieval snapshot fingerprints.
LONG_RELATIONS_OVERRIDE_PATH = (
    os.path.abspath(os.path.expanduser(os.getenv("WEAVE_MEM_LONG_RELATIONS_PATH", "")))
    if os.getenv("WEAVE_MEM_LONG_RELATIONS_PATH", "").strip()
    else None
)
# Optional replacement for mid_relations.json.
MID_RELATIONS_OVERRIDE_PATH = (
    os.path.abspath(os.path.expanduser(os.getenv("WEAVE_MEM_MID_RELATIONS_PATH", "")))
    if os.getenv("WEAVE_MEM_MID_RELATIONS_PATH", "").strip()
    else None
)
MID_ENRICHED_EMBEDDINGS_PATH = os.path.abspath(
    os.path.expanduser(
        _env_str("WEAVE_MEM_MID_ENRICHED_EMBEDDINGS_PATH")
        or os.path.join(DATA_DIR, "mid_enriched_embeddings.json")
    )
)
INGEST_PROGRESS_PATH = os.path.join(DATA_DIR, "progress_ingest.json")
