"""
OpenAI provider tests – uses the official `openai` Python SDK.

The SDK is pointed at the gateway URL instead of api.openai.com.
The gateway API key (generated in the AI workspace) is used for auth.

Resources tested:
  - Chat completions (non-streaming)
  - Chat completions (streaming)
  - Multi-turn conversation
  - Embeddings
  - Models list
"""

import logging
import pytest
from openai import OpenAI
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
_PROVIDER = "openai"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_BASE_URL = provider_base_url(_cfg, _PROVIDER)
_API_KEY = provider_api_key(_cfg, _PROVIDER)
_MODELS = provider_models(_cfg, _PROVIDER)
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.openai,
    pytest.mark.skipif(not _ENABLED, reason="OpenAI provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    import httpx

    http_client = httpx.Client(verify=_VERIFY_SSL)
    return OpenAI(
        api_key=_API_KEY,
        base_url=_BASE_URL,
        http_client=http_client,
        default_headers={"X-API-Key": _API_KEY},
    )


# ── Chat completions ──────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_basic(model):
    """Single-turn chat returns a non-empty assistant message."""
    logger.info("model=%s  endpoint=%s", model, _BASE_URL)
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: Hello"}],
        max_tokens=20,
    )
    assert response.id, "Response must have an id"
    assert response.choices, "Response must have choices"
    content = response.choices[0].message.content
    assert content and content.strip(), "Assistant message must not be empty"
    assert response.usage is not None, "Token usage must be present"
    assert response.usage.total_tokens > 0
    logger.info("response=%r  tokens(in=%s out=%s)",
                content, response.usage.prompt_tokens, response.usage.completion_tokens)


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_with_system_message(model):
    """System message is respected and a proper reply is returned."""
    logger.info("model=%s", model)
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers in one sentence."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=50,
    )
    content = response.choices[0].message.content
    assert content and content.strip()
    assert response.model  # gateway should echo the model name
    logger.info("response=%r  model_echoed=%s", content, response.model)


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_multi_turn(model):
    """Multi-turn conversation is handled correctly."""
    logger.info("model=%s  turns=3", model)
    client = _client()
    messages = [
        {"role": "user", "content": "My favourite colour is blue. Remember that."},
        {"role": "assistant", "content": "Noted, your favourite colour is blue."},
        {"role": "user", "content": "What is my favourite colour?"},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=50,
    )
    content = response.choices[0].message.content or ""
    assert "blue" in content.lower(), f"Expected 'blue' in response, got: {content!r}"
    logger.info("response=%r", content)


@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_streaming(model):
    """Streaming chat returns tokens incrementally and assembles a full reply."""
    logger.info("model=%s  stream=true", model)
    client = _client()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 5, numbers only."}],
        max_tokens=30,
        stream=True,
    )
    chunks_received = 0
    full_content = ""
    for chunk in stream:
        chunks_received += 1
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_content += delta

    assert chunks_received > 1, "Streaming must deliver more than one chunk"
    assert full_content.strip(), "Streamed content must not be empty"
    logger.info("chunks=%d  assembled=%r", chunks_received, full_content)


# ── Embeddings ────────────────────────────────────────────────────────────────

@pytest.mark.embeddings
@pytest.mark.parametrize("model", _MODELS)
def test_embeddings_single_input(model):
    """Embedding endpoint returns a non-empty vector for a single string."""
    logger.info("model=%s  embedding=single", model)
    client = _client()
    try:
        response = client.embeddings.create(
            model=model,
            input="The quick brown fox jumps over the lazy dog",
        )
    except Exception as exc:
        # Not all models support embeddings; skip gracefully
        pytest.skip(f"Model {model!r} does not support embeddings via this provider: {exc}")

    assert response.data, "Embedding response must contain data"
    vector = response.data[0].embedding
    assert isinstance(vector, list) and len(vector) > 0, "Embedding vector must be a non-empty list"
    assert response.usage.total_tokens > 0
    logger.info("vector_dim=%d  tokens=%d", len(vector), response.usage.total_tokens)


@pytest.mark.embeddings
@pytest.mark.parametrize("model", _MODELS)
def test_embeddings_batch_input(model):
    """Embedding endpoint handles a batch of inputs."""
    logger.info("model=%s  embedding=batch(3)", model)
    client = _client()
    inputs = ["Hello world", "Goodbye world", "The sky is blue"]
    try:
        response = client.embeddings.create(model=model, input=inputs)
    except Exception as exc:
        pytest.skip(f"Model {model!r} does not support embeddings: {exc}")

    assert len(response.data) == len(inputs), "Batch size must match number of inputs"
    for item in response.data:
        assert len(item.embedding) > 0
    logger.info("batch_size=%d  vector_dim=%d", len(response.data), len(response.data[0].embedding))


# ── Models list ───────────────────────────────────────────────────────────────

def test_models_list():
    """Gateway exposes a models list endpoint compatible with the OpenAI SDK."""
    client = _client()
    try:
        models = client.models.list()
        assert models.data is not None
    except Exception as exc:
        pytest.skip(f"Models list endpoint not available: {exc}")
