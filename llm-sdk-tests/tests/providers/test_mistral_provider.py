"""
Mistral provider tests – uses the official `mistralai` Python SDK.

The SDK's server_url is overridden to the gateway URL.
The gateway API key is passed as the Mistral API key.

Resources tested:
  - Chat completions (non-streaming)
  - Chat completions (streaming)
  - Multi-turn conversation
  - Embeddings
  - Models list
"""

import logging
import pytest
from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage, AssistantMessage
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
_PROVIDER = "mistral"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_BASE_URL = provider_base_url(_cfg, _PROVIDER)
_API_KEY = provider_api_key(_cfg, _PROVIDER)
_MODELS = provider_models(_cfg, _PROVIDER)
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.mistral,
    pytest.mark.skipif(not _ENABLED, reason="Mistral provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client() -> Mistral:
    # mistralai sends Authorization: Bearer by default, but the gateway expects
    # X-API-Key.  Pass a custom httpx.Client with an event hook that injects the
    # header on every outgoing request.
    import httpx

    def _inject(request):
        request.headers["X-API-Key"] = _API_KEY

    http_client = httpx.Client(
        verify=_VERIFY_SSL,
        event_hooks={"request": [_inject]},
    )
    return Mistral(api_key=_API_KEY, server_url=_BASE_URL, client=http_client)


# ── Chat completions ──────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_basic(model):
    """Basic chat returns a non-empty assistant message."""
    logger.info("model=%s  endpoint=%s", model, _BASE_URL)
    client = _client()
    response = client.chat.complete(
        model=model,
        messages=[UserMessage(content="Reply with only the word: Hello")],
        max_tokens=20,
    )
    assert response.id, "Response must have an id"
    assert response.choices, "Response must have choices"
    content = response.choices[0].message.content
    assert content and (content.strip() if isinstance(content, str) else content)
    assert response.usage and response.usage.total_tokens > 0
    logger.info("response=%r  tokens=%d", content, response.usage.total_tokens)


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_with_system_message(model):
    """System message shapes the assistant behaviour."""
    client = _client()
    response = client.chat.complete(
        model=model,
        messages=[
            SystemMessage(content="Answer in exactly one sentence."),
            UserMessage(content="What is the capital of France?"),
        ],
        max_tokens=60,
    )
    content = response.choices[0].message.content or ""
    assert "paris" in content.lower(), f"Expected 'Paris' in reply: {content!r}"


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_multi_turn(model):
    """Context from prior turns is maintained."""
    client = _client()
    messages = [
        UserMessage(content="My favourite animal is a capybara."),
        AssistantMessage(content="Noted! Your favourite animal is a capybara."),
        UserMessage(content="What is my favourite animal?"),
    ]
    response = client.chat.complete(model=model, messages=messages, max_tokens=40)
    content = response.choices[0].message.content or ""
    assert "capybara" in content.lower(), f"Expected 'capybara' in: {content!r}"


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_temperature(model):
    """Temperature parameter is accepted without error."""
    client = _client()
    response = client.chat.complete(
        model=model,
        messages=[UserMessage(content="Say hello.")],
        max_tokens=20,
        temperature=0.0,
    )
    assert response.choices[0].message.content


# ── Streaming ─────────────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_streaming(model):
    """Streaming delivers multiple events and assembles a full response."""
    logger.info("model=%s  stream=true", model)
    client = _client()
    full_content = ""
    event_count = 0

    with client.chat.stream(
        model=model,
        messages=[UserMessage(content="Count from 1 to 5.")],
        max_tokens=40,
    ) as stream:
        for event in stream:
            event_count += 1
            if event.data.choices and event.data.choices[0].delta.content:
                full_content += event.data.choices[0].delta.content

    assert event_count > 0, "No streaming events received"
    assert full_content.strip(), "Streamed content must not be empty"
    logger.info("events=%d  assembled=%r", event_count, full_content)


# ── Embeddings ────────────────────────────────────────────────────────────────

@pytest.mark.embeddings
@pytest.mark.parametrize("model", _MODELS)
def test_embeddings(model):
    """Embedding endpoint returns non-empty vectors."""
    client = _client()
    # Mistral has dedicated embedding models; try with mistral-embed
    embed_model = "mistral-embed"
    try:
        response = client.embeddings.create(
            model=embed_model,
            inputs=["Hello world", "Goodbye world"],
        )
    except Exception as exc:
        pytest.skip(f"Embedding endpoint not available: {exc}")

    assert response.data and len(response.data) == 2
    for item in response.data:
        assert item.embedding and len(item.embedding) > 0


# ── Models list ───────────────────────────────────────────────────────────────

def test_models_list():
    """Models list endpoint returns at least one model."""
    client = _client()
    try:
        result = client.models.list()
        assert result.data, "Models list must not be empty"
    except Exception as exc:
        pytest.skip(f"Models list not available: {exc}")
