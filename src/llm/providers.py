

from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible backend: where to send requests and with which key."""

    base_url: str | None
    api_key: str | None


@dataclass(frozen=True)
class Provider:
    """Protocol, endpoint, and default request options for a model.

    protocol selects the calling convention ("openai" for the standard chat route,
    "openai_embedding" for embeddings, "dashscope_rerank" for the native rerank SDK).
    extra_body carries request options such as thinking suppression.
    """

    protocol: str
    endpoint: Endpoint
    extra_body: dict = field(default_factory=dict)


# Default chat route.
CHAT = Provider(
    "openai",
    Endpoint(config.CHAT_BASE_URL, config.CHAT_API_KEY),
    {"enable_thinking": config.ENABLE_THINKING},
)
EMBEDDING = Provider(
    "openai_embedding", Endpoint(config.OPENAI_BASE_URL, config.OPENAI_API_KEY)
)
RERANK = Provider("dashscope_rerank", Endpoint(None, config.RERANK_API_KEY))
# Dedicated GLM route.
GLM = Provider(
    "openai",
    Endpoint(config.GLM_BASE_URL, config.GLM_API_KEY),
    {"enable_thinking": config.GLM_ENABLE_THINKING},
)

# Model-to-provider routing.
_REGISTRY: dict[str, Provider] = {
    config.GLM_MODEL: GLM,
    config.EMBEDDING_MODEL: EMBEDDING,
    config.RERANK_MODEL: RERANK,
}

# Unregistered model names use the default chat endpoint.
_DEFAULT = CHAT


def provider_for(model: str) -> Provider:
    """Resolve the Provider for a model name, falling back to the default chat endpoint."""
    return _REGISTRY.get(model, _DEFAULT)
