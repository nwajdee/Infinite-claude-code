"""FreeLLMAPI (OpenAI-compat) adapter."""
from providers.defaults import FREELLMAPI_DEFAULT_BASE
from .client import FreeLLMAPIProvider

__all__ = [
    "FREELLMAPI_DEFAULT_BASE",
    "FreeLLMAPIProvider",
]
