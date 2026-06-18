"""Build OpenAI-format request body from Anthropic request for FreeLLMAPI."""
from __future__ import annotations

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for FreeLLMAPI."""
    logger.debug(
        "Converting Anthropic request to OpenAI format for FreeLLMAPI: "
        "model={}, messages={}",
        getattr(request_data, "model", "unknown"),
        len(getattr(request_data, "messages", [])),
    )

    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=(
                ReasoningReplayMode.REASONING_CONTENT
                if thinking_enabled
                else ReasoningReplayMode.DISABLED
            ),
        )
    except OpenAIConversionError as e:
        raise InvalidRequestError(str(e)) from e

    # FreeLLMAPI uses 'auto' to let its internal router pick the best model
    body["model"] = "auto"

    request_extra = getattr(request_data, "extra_body", None)
    if isinstance(request_extra, dict) and request_extra:
        body["extra_body"] = dict(request_extra)

    logger.debug(
        "Conversion complete: model={}, messages={}, tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )

    return body
