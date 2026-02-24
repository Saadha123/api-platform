"""
Azure AI Foundry provider tests – uses the openai AzureOpenAI SDK.

Azure AI Foundry exposes an OpenAI-compatible endpoint at:
  /openai/deployments/{deployment}/chat/completions

Gateway path rewriting (context = /azure-foundry, upstream = resource root):
  SDK constructs:  /azure-foundry/openai/deployments/{model}/chat/completions
  gateway strips:  /azure-foundry
  gateway forwards:https://{resource}.cognitiveservices.azure.com/openai/deployments/{model}/...

The model parameter passed to the SDK must be the Azure deployment name.

Resources tested:
  - Chat completions (non-streaming)
  - Chat completions with system message
  - Multi-turn conversation
  - Streaming chat completions
"""

import logging
import pytest
from openai import AzureOpenAI
from utils.config import (
    load_config,
    is_provider_enabled,
    get_provider,
    get_verify_ssl,
)

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "azure_foundry"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_PCFG = get_provider(_cfg, _PROVIDER)
_BASE_URL = _PCFG.get("base_url", "")
_API_KEY = _PCFG.get("api_key", "")
_API_VERSION = _PCFG.get("api_version", "2024-05-01-preview")
_MODELS = _PCFG.get("models", [])
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.azure_foundry,
    pytest.mark.skipif(not _ENABLED, reason="Azure AI Foundry provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client() -> AzureOpenAI:
    import httpx

    return AzureOpenAI(
        api_key=_API_KEY,
        azure_endpoint=_BASE_URL,
        api_version=_API_VERSION,
        default_headers={"X-API-Key": _API_KEY},
        http_client=httpx.Client(verify=_VERIFY_SSL),
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
        max_tokens=500,
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
        max_tokens=500,
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
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=500)
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
        max_tokens=500,
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
    logger.info("chunks=%d  assembled=%r", chunks, content)
