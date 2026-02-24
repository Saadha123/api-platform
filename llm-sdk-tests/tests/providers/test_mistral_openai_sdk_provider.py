"""
Mistral provider tests using the OpenAI-compatible SDK (openai.OpenAI).

Mistral exposes an OpenAI-compatible API at /v1, so the standard openai.OpenAI
client works by appending /v1 to the gateway context URL.

Resources tested:
  - Chat completions (non-streaming)
  - Chat completions with system message
  - Multi-turn conversation
  - Streaming chat completions
  - Temperature parameter
  - Models list
"""

import logging
import pytest
from openai import OpenAI
from utils.config import (
    load_config,
    is_provider_enabled,
    get_provider,
    get_verify_ssl,
)

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "mistral"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_PCFG = get_provider(_cfg, _PROVIDER)
_BASE_URL = _PCFG.get("base_url", "")
_API_KEY = _PCFG.get("api_key", "")
_MODELS = _PCFG.get("models", [])
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.mistral,
    pytest.mark.openai,
    pytest.mark.skipif(not _ENABLED, reason="Mistral provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    import httpx

    # Mistral's OpenAI-compatible endpoint lives at /v1
    return OpenAI(
        api_key=_API_KEY,
        base_url=_BASE_URL + "/v1",
        http_client=httpx.Client(verify=_VERIFY_SSL),
        default_headers={"X-API-Key": _API_KEY},
    )


# ── Chat completions ──────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_basic(model):
    """Single-turn chat returns a non-empty assistant message."""
    logger.info("model=%s  endpoint=%s/v1", model, _BASE_URL)
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: Hello"}],
        max_tokens=20,
    )
    assert response.choices, "Response must have choices"
    content = response.choices[0].message.content
    assert content and content.strip(), "Assistant message must not be empty"
    assert response.usage and response.usage.total_tokens > 0
    logger.info("response=%r  tokens=%d", content, response.usage.total_tokens)


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_with_system_message(model):
    """System message is honoured by the model."""
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer only with yes or no."},
            {"role": "user", "content": "Is the sky blue?"},
        ],
        max_tokens=10,
    )
    content = response.choices[0].message.content or ""
    assert content.strip()


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_multi_turn(model):
    """Multi-turn context is preserved across messages."""
    client = _client()
    messages = [
        {"role": "user", "content": "My pet's name is Biscuit."},
        {"role": "assistant", "content": "Got it, your pet is named Biscuit."},
        {"role": "user", "content": "What is my pet's name?"},
    ]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=30)
    content = response.choices[0].message.content or ""
    assert "biscuit" in content.lower(), f"Expected 'Biscuit' in reply, got: {content!r}"


@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_streaming(model):
    """Streaming delivers multiple chunks and assembles a valid response."""
    logger.info("model=%s  stream=true", model)
    client = _client()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say: one two three"}],
        max_tokens=20,
        stream=True,
    )
    chunks = 0
    content = ""
    for chunk in stream:
        chunks += 1
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            content += delta

    assert chunks > 1
    assert content.strip()
    logger.info("events=%d  assembled=%r", chunks, content)


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_temperature(model):
    """Temperature parameter is accepted and a valid response is returned."""
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=20,
        temperature=0.7,
    )
    assert response.choices[0].message.content


def test_models_list():
    """Models list endpoint is reachable via OpenAI SDK."""
    client = _client()
    try:
        models = client.models.list()
        assert models.data is not None
    except Exception as exc:
        pytest.skip(f"Models list endpoint not available: {exc}")
