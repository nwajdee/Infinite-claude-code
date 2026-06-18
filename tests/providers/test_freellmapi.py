"""Tests for the FreeLLMAPI (OpenAI-compatible) provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.freellmapi import FREELLMAPI_DEFAULT_BASE, FreeLLMAPIProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "freellmapi/auto"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def freellmapi_config():
    return ProviderConfig(
        api_key="test_freellmapi_key",
        base_url=FREELLMAPI_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture
def freellmapi_config_no_thinking():
    return ProviderConfig(
        api_key="test_freellmapi_key",
        base_url=FREELLMAPI_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=False,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def freellmapi_provider(freellmapi_config):
    return FreeLLMAPIProvider(freellmapi_config)


def test_init(freellmapi_config):
    """Test provider initialization."""
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = FreeLLMAPIProvider(freellmapi_config)
        assert provider._api_key == "test_freellmapi_key"
        assert provider._base_url == FREELLMAPI_DEFAULT_BASE
        assert provider._provider_name == "FREELLMAPI"
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert FREELLMAPI_DEFAULT_BASE == "http://localhost:3001/v1"


def test_build_request_body_sets_model_auto(freellmapi_provider):
    """FreeLLMAPI always sends model='auto' for upstream routing."""
    req = MockRequest()
    body = freellmapi_provider._build_request_body(req)

    assert body["model"] == "auto"
    assert body["messages"][0]["role"] == "system"
    assert "max_tokens" in body


def test_build_request_body_model_auto_even_with_explicit_model(freellmapi_provider):
    """Even when the request has a specific model, body uses 'auto'."""
    req = MockRequest(model="claude-sonnet-4-20250514")

    body = freellmapi_provider._build_request_body(req)

    assert body["model"] == "auto"


def test_build_request_body_global_disable_blocks_reasoning_mapping(freellmapi_config_no_thinking):
    provider = FreeLLMAPIProvider(freellmapi_config_no_thinking)
    req = MockRequest()
    body = provider._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


def test_build_request_body_preserves_caller_extra_body(freellmapi_provider):
    """extra_body passthrough still works."""
    req = MockRequest(extra_body={"metadata": {"user": "u1"}})

    body = freellmapi_provider._build_request_body(req)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}


def test_custom_base_url():
    """Provider accepts an overridden base URL."""
    custom_url = "http://192.168.1.50:3001/v1"
    config = ProviderConfig(api_key="test-key", base_url=custom_url)
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = FreeLLMAPIProvider(config)
    assert provider._base_url == custom_url
    assert provider._provider_name == "FREELLMAPI"


def test_api_key_stored():
    """Provider stores the credential as its API key."""
    config = ProviderConfig(api_key="freellmapi-my-secret-key")
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = FreeLLMAPIProvider(config)
    assert provider._api_key == "freellmapi-my-secret-key"


def test_build_request_body_direct_function():
    """Test build_request_body directly from the request module."""
    from providers.freellmapi.request import build_request_body

    request = MockRequest()
    body = build_request_body(request, thinking_enabled=False)
    assert body["model"] == "auto"


def test_build_request_body_with_thinking():
    """When thinking is enabled, the reasoning_replay mode uses REASONING_CONTENT."""
    from providers.freellmapi.request import build_request_body

    request = MockRequest()
    body = build_request_body(request, thinking_enabled=True)
    assert body["model"] == "auto"


@pytest.mark.asyncio
async def test_stream_response_text(freellmapi_provider):
    """Text content deltas are emitted as text blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        freellmapi_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in freellmapi_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(freellmapi_provider):
    freellmapi_provider._client = AsyncMock()

    await freellmapi_provider.cleanup()

    freellmapi_provider._client.close.assert_called_once()
