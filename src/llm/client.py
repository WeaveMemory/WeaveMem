

from __future__ import annotations

import logging
import math
import os
import threading
import time

from openai import OpenAI

import config

from . import providers

logger = logging.getLogger("llm")

_clients: dict[tuple[str | None, str | None], OpenAI] = {}
_request_start_condition = threading.Condition()
_last_request_start_by_endpoint: dict[providers.Endpoint, float] = {}
_rate_limit_cooldown_until_by_endpoint: dict[providers.Endpoint, float] = {}
_RATE_LIMIT_SIGNALS = (
    "limit_requests",
    "limit_burst_rate",
    "rate limit",
    "429005",
    "too many requests",
)
_TERMINAL_AUTH_SIGNALS = (
    "invalid_api_key",
    "api-key is blocked",
    "api-key blocked",
    "api key is blocked",
    "api key blocked",
)


class TerminalLLMAuthenticationError(RuntimeError):
    """A credential/authorization failure that must not be retried."""


def _status_code(error: Exception) -> int | None:
    """Best-effort HTTP status extraction across OpenAI-compatible clients."""
    raw = getattr(error, "status_code", None)
    if raw is None:
        raw = getattr(getattr(error, "response", None), "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _is_terminal_auth_error(error: Exception) -> bool:
    """Return whether retrying cannot repair this request's credentials."""
    if _status_code(error) in {401, 403}:
        return True
    details = " ".join(
        str(value)
        for value in (
            error,
            getattr(error, "body", None),
            getattr(error, "code", None),
        )
        if value is not None
    ).casefold()
    return any(signal in details for signal in _TERMINAL_AUTH_SIGNALS)


def _minimum_request_interval_seconds() -> float:
    """Return the optional chat-start spacing without routing it through config."""
    raw = os.getenv("WEAVE_MEM_LLM_MIN_REQUEST_INTERVAL_SECONDS", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "invalid WEAVE_MEM_LLM_MIN_REQUEST_INTERVAL_SECONDS=%r; disabling throttle",
            raw,
        )
        return 0.0


def _retry_backoff_seconds() -> float:
    """Return the execution-only retry delay, falling back to configured policy."""
    fallback = max(0.0, float(config.LLM_RETRY_BACKOFF_SECONDS))
    raw = os.getenv("WEAVE_MEM_LLM_RETRY_BACKOFF_SECONDS")
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        logger.warning(
            "invalid WEAVE_MEM_LLM_RETRY_BACKOFF_SECONDS=%r; using configured %.3fs",
            raw,
            fallback,
        )
        return fallback
    return value


def _rate_limit_cooldown_seconds() -> float:
    """Return the optional endpoint-wide cooldown applied after rate-limit errors."""
    raw = os.getenv("WEAVE_MEM_LLM_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        logger.warning(
            "invalid WEAVE_MEM_LLM_RATE_LIMIT_COOLDOWN_SECONDS=%r; disabling cooldown",
            raw,
        )
        return 0.0
    return value


def _apply_rate_limit_cooldown(
    endpoint: providers.Endpoint,
    error: Exception,
) -> bool:
    """Push an endpoint's schedule after a recognized rate-limit exception."""
    if not any(signal in str(error).casefold() for signal in _RATE_LIMIT_SIGNALS):
        return False
    cooldown = _rate_limit_cooldown_seconds()
    if cooldown <= 0:
        return False

    with _request_start_condition:
        cooldown_until = time.monotonic() + cooldown
        cooldown_until = max(
            cooldown_until,
            _rate_limit_cooldown_until_by_endpoint.get(endpoint, 0.0),
        )
        _rate_limit_cooldown_until_by_endpoint[endpoint] = cooldown_until
        _request_start_condition.notify_all()
    return True


def _wait_for_request_start_slot(endpoint: providers.Endpoint) -> None:
    """Reserve and wait for one endpoint-local request-start slot.

    Waiting releases the lock so rate-limit responses can extend the shared
    deadline and wake callers to recompute their slots.
    """
    interval = _minimum_request_interval_seconds()
    with _request_start_condition:
        while True:
            cooldown_until = _rate_limit_cooldown_until_by_endpoint.get(endpoint, 0.0)
            if interval <= 0 and cooldown_until <= 0:
                return

            now = time.monotonic()
            if cooldown_until and cooldown_until <= now:
                _rate_limit_cooldown_until_by_endpoint.pop(endpoint, None)
                cooldown_until = 0.0
                if interval <= 0:
                    return
            last_start = _last_request_start_by_endpoint.get(endpoint)
            interval_ready = (
                last_start + interval if last_start is not None else now
            )
            eligible_at = max(interval_ready, cooldown_until)
            if eligible_at <= now:
                _last_request_start_by_endpoint[endpoint] = now
                return
            _request_start_condition.wait(timeout=eligible_at - now)


def _client_for(endpoint: providers.Endpoint) -> OpenAI:
    """Return a lazily created and cached client for one endpoint."""
    key = (endpoint.api_key, endpoint.base_url)
    if key not in _clients:
        # Disable SDK retries so all attempts honor shared spacing and 429 cooldowns.
        _clients[key] = OpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            max_retries=0,
        )
    return _clients[key]


def _call_openai(
    messages: list[dict],
    *,
    provider: providers.Provider,
    model: str,
    max_tokens: int,
    timeout: int,
    temperature: float | None,
    response_format: dict | None = None,
):
    """Standard OpenAI-compatible chat completion (one attempt)."""
    client = _client_for(provider.endpoint)
    # extra_body carries non-OpenAI-standard request options, e.g. thinking suppression.
    extra = {"extra_body": provider.extra_body} if provider.extra_body else {}
    request = {
        "model": model,
        "messages": messages,
        "stream": False,
        "timeout": timeout,
        **extra,
    }

    if model == "gpt-4o":
        request["max_completion_tokens"] = max_tokens
    else:
        request["max_tokens"] = max_tokens
    if temperature is not None:
        request["temperature"] = temperature
    if response_format is not None:
        request["response_format"] = response_format
    return client.chat.completions.create(**request)


# Single-attempt handlers selected by the provider's protocol.
_CHAT_HANDLERS = {"openai": _call_openai}


def _field(value: object, name: str, default: object = None) -> object:
    """Read one field from either an SDK object or a decoded JSON mapping."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_content(response: object | None) -> str | None:
    """Extract visible answer text from Chat Completions or Responses payloads.

    Some OpenAI-compatible gateways accept requests on ``/chat/completions`` but
    return a Responses-API-shaped object.  The OpenAI SDK preserves its ``output``
    fields while leaving ``choices`` as ``None``.  Only ``output_text`` parts are
    answer-visible; reasoning items remain excluded.
    """
    if response is None:
        return None

    choices = _field(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        message = _field(choices[0], "message")
        content = _field(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            parts = []
            for item in content:
                text = _field(item, "text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)

    output = _field(response, "output")
    if not isinstance(output, (list, tuple)):
        return None
    parts = []
    for item in output:
        content = _field(item, "content")
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            if str(_field(part, "type", "") or "") != "output_text":
                continue
            text = _field(part, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts) if parts else None


def _response_usage_details(response: object) -> dict[str, int]:
    """Normalize token counters from both supported response schemas."""
    usage = _field(response, "usage")
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(
        _field(usage, "prompt_tokens", 0)
        or _field(usage, "input_tokens", 0)
        or 0
    )
    completion_tokens = int(
        _field(usage, "completion_tokens", 0)
        or _field(usage, "output_tokens", 0)
        or 0
    )
    total_tokens = int(_field(usage, "total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _response_finish_reason(response: object) -> str | None:
    """Normalize Chat Completions finish_reason and Responses status."""
    choices = _field(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        finish_reason = _field(choices[0], "finish_reason")
        if finish_reason is not None:
            return str(finish_reason)

    status = _field(response, "status")
    if status is None:
        return None
    normalized = str(status)
    if normalized == "completed":
        return "stop"
    if normalized == "incomplete":
        details = _field(response, "incomplete_details")
        reason = _field(details, "reason")
        if str(reason or "") == "max_output_tokens":
            return "length"
        if reason is not None:
            return str(reason)
    return normalized


def _request(
    messages: list[dict],
    *,
    model: str | None,
    max_tokens: int | None,
    timeout: int | None,
    max_retries: int | None,
    temperature: float | None = None,
    response_format: dict | None = None,
):
    """Send a chat request with retries.

    Return the raw response, or ``None`` after retryable failures are exhausted.
    Authentication and authorization failures raise immediately.
    """
    # Read defaults at call time: benchmark profiles may reload config in one
    # process while generation modules retain imported aliases to these functions.
    model = config.EXTRACTION_MODEL if model is None else model
    max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    timeout = config.LLM_TIMEOUT_SECONDS if timeout is None else timeout
    max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    provider = providers.provider_for(model)
    handler = _CHAT_HANDLERS.get(provider.protocol)
    if handler is None:
        raise ValueError(
            f"model {model!r} maps to non-chat protocol {provider.protocol!r}"
        )
    for attempt in range(1, max_retries + 1):
        _wait_for_request_start_slot(provider.endpoint)
        start = time.time()
        try:
            logger.info(
                "request (model=%s, attempt %d/%d, timeout=%ds)...",
                model, attempt, max_retries, timeout,
            )
            response = handler(
                messages,
                provider=provider,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                temperature=temperature,
                response_format=response_format,
            )
            content = _response_content(response)
            logger.info(
                "response in %.1fs (%d chars)",
                time.time() - start, len(content or ""),
            )
            return response
        except Exception as e:  # the backend may raise various exceptions
            if _is_terminal_auth_error(e):
                logger.error(
                    "non-retryable authentication/authorization failure "
                    "(model=%s, status=%s): %s",
                    model,
                    _status_code(e),
                    e,
                )
                raise TerminalLLMAuthenticationError(
                    "non-retryable authentication/authorization failure "
                    f"for model {model!r}"
                ) from e
            _apply_rate_limit_cooldown(provider.endpoint, e)
            logger.warning(
                "request failed in %.1fs (attempt %d/%d): %s",
                time.time() - start, attempt, max_retries, e,
            )
            if attempt >= max_retries:
                logger.error("reached max retries (%d): %s", max_retries, e)
                return None
            time.sleep(_retry_backoff_seconds())
    return None


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
) -> str | None:
    """Send a chat request with retries and return the message content.

    Returns ``None`` if retryable failures exhaust all attempts. Terminal
    authentication/authorization failures raise immediately.
    Omitted model and request limits use the current configuration on every call.
    """
    response = _request(
        messages, model=model, max_tokens=max_tokens,
        timeout=timeout, max_retries=max_retries, temperature=temperature,
    )
    return _response_content(response)


def complete_with_usage(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
) -> tuple[str | None, int]:
    """Like complete(), but also return the prompt-token count (0 if unavailable)."""
    response = _request(
        messages, model=model, max_tokens=max_tokens,
        timeout=timeout, max_retries=max_retries, temperature=temperature,
    )
    if response is None:
        return None, 0
    content = _response_content(response)
    usage = _response_usage_details(response)
    return content, usage["prompt_tokens"]


def complete_with_usage_details(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> tuple[str | None, dict[str, object]]:
    """Like :func:`complete`, returning prompt/completion/total token counters.

    OpenAI-compatible endpoints do not all expose every counter, so absent values are
    normalized to zero.  Keeping this separate preserves the compact historical
    ``complete_with_usage`` API used by the QA evaluator.
    """
    response = _request(
        messages,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        response_format=response_format,
    )
    if response is None:
        return None, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": None,
        }
    content = _response_content(response)
    usage = _response_usage_details(response)
    finish_reason = _response_finish_reason(response)
    return content, {
        **usage,
        "finish_reason": finish_reason,
    }


def get_response(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
) -> str | None:
    """Convenience wrapper for a single prompt; delegates to complete()."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return complete(
        messages,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
    )
