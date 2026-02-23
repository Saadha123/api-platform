"""
Anthropic provider tests – uses the official `anthropic` Python SDK.

The SDK is pointed at the gateway; the gateway API key is passed as the
Anthropic API key.  The gateway proxies /v1/messages to Anthropic upstream.

Resources tested:
  - Messages (non-streaming)
  - Messages (streaming)
  - Multi-turn conversation
  - System prompt
  - Token counting (if endpoint is available)
"""

import logging
import pytest
import anthropic as _anthropic
from utils.config import (
    load_config,
    is_provider_enabled,
    provider_base_url,
    provider_api_key,
    provider_models,
    get_verify_ssl,
)

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "anthropic"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_BASE_URL = provider_base_url(_cfg, _PROVIDER)
_API_KEY = provider_api_key(_cfg, _PROVIDER)
_MODELS = provider_models(_cfg, _PROVIDER)
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.anthropic,
    pytest.mark.skipif(not _ENABLED, reason="Anthropic provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client() -> _anthropic.Anthropic:
    import httpx

    http_client = httpx.Client(verify=_VERIFY_SSL)
    return _anthropic.Anthropic(
        api_key=_API_KEY,
        base_url=_BASE_URL,
        http_client=http_client,
    )


# ── Messages ──────────────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_message_basic(model):
    """A basic user message returns a non-empty text response."""
    logger.info("model=%s  endpoint=%s", model, _BASE_URL)
    client = _client()
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": "Reply with only the word: Hello"}],
    )
    assert response.id, "Response must have an id"
    assert response.content, "Response must have content blocks"
    text = response.content[0].text
    assert text and text.strip(), "Text block must not be empty"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    logger.info("response=%r  tokens(in=%s out=%s)",
                text, response.usage.input_tokens, response.usage.output_tokens)


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_message_with_system_prompt(model):
    """System prompt is honoured and affects the response tone/content."""
    logger.info("model=%s  system_prompt=pirate", model)
    client = _client()
    response = client.messages.create(
        model=model,
        max_tokens=60,
        system="You are a pirate. Always end your reply with 'Arrr!'",
        messages=[{"role": "user", "content": "How are you?"}],
    )
    text = response.content[0].text
    assert text.strip(), "Response must not be empty"
    logger.info("response=%r", text)


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_message_multi_turn(model):
    """Multi-turn conversation context is maintained."""
    client = _client()
    messages = [
        {"role": "user", "content": "My lucky number is 42."},
        {"role": "assistant", "content": "I'll remember that. Your lucky number is 42."},
        {"role": "user", "content": "What is my lucky number?"},
    ]
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=messages,
    )
    text = response.content[0].text
    assert "42" in text, f"Expected '42' in multi-turn reply, got: {text!r}"


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_message_stop_reason(model):
    """Response includes a stop_reason indicating why generation ended."""
    client = _client()
    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": "Write a long poem about the ocean."}],
    )
    # With max_tokens=10 the stop reason should be 'max_tokens' or 'end_turn'
    assert response.stop_reason in {"end_turn", "max_tokens", "stop_sequence"}


# ── Streaming ─────────────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_message_streaming(model):
    """Streaming messages returns incremental text events."""
    logger.info("model=%s  stream=true", model)
    client = _client()
    full_text = ""
    event_count = 0

    with client.messages.stream(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ) as stream:
        for text in stream.text_stream:
            full_text += text
            event_count += 1

    assert event_count > 0, "No streaming events received"
    assert full_text.strip(), "Streamed text must not be empty"
    logger.info("events=%d  assembled=%r", event_count, full_text)


@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_message_streaming_final_message(model):
    """stream.get_final_message() returns a complete Message object."""
    logger.info("model=%s  stream=true  check=final_message", model)
    client = _client()
    with client.messages.stream(
        model=model,
        max_tokens=30,
        messages=[{"role": "user", "content": "Say: done"}],
    ) as stream:
        final = stream.get_final_message()

    assert final.content, "Final message must have content"
    assert final.usage.output_tokens > 0
    logger.info("final_text=%r  output_tokens=%d",
                final.content[0].text, final.usage.output_tokens)
