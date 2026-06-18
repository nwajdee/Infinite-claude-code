"""FreeLLMAPI provider client."""
from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import FREELLMAPI_DEFAULT_BASE
from providers.transports.openai_chat import OpenAIChatTransport
from .request import build_request_body


class FreeLLMAPIProvider(OpenAIChatTransport):
    """FreeLLMAPI using ``http://localhost:3001/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(
            config=config,
            provider_name="FREELLMAPI",
            base_url=config.base_url or FREELLMAPI_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, *, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
