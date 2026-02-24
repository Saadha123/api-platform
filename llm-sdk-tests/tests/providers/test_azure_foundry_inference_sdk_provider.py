"""
Azure AI Foundry provider tests – uses the azure-ai-inference SDK.

Requires: pip install azure-ai-inference

The ChatCompletionsClient uses the Azure AI Foundry inference endpoint format:
  POST {endpoint}/models/chat/completions

Gateway path rewriting (context = /azure-foundry, upstream = resource root):
  endpoint passed: /azure-foundry/models  (base_url + /models)
  SDK appends:     /chat/completions
  SDK sends to:    /azure-foundry/models/chat/completions
  gateway strips:  /azure-foundry
  gateway forwards:https://{resource}.cognitiveservices.azure.com/models/chat/completions

The model is specified per-request in client.complete(model=...).

Resources tested:
  - Chat completions (non-streaming)
  - Chat completions with system message
  - Multi-turn conversation
  - Streaming chat completions
"""

import logging
import pytest
from utils.config import (
    load_config,
    is_provider_enabled,
    get_provider,
    get_verify_ssl,
)

# Skip entire module if azure-ai-inference is not installed
pytest.importorskip(
    "azure.ai.inference",
    reason="azure-ai-inference not installed; run: pip install azure-ai-inference",
)

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "azure_foundry"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_PCFG = get_provider(_cfg, _PROVIDER)
_BASE_URL = _PCFG.get("base_url", "")
_API_KEY = _PCFG.get("api_key", "")
_MODELS = _PCFG.get("models", [])
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.azure_foundry,
    pytest.mark.skipif(not _ENABLED, reason="Azure AI Foundry provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client() -> ChatCompletionsClient:
    # The inference SDK uses Azure Core's HTTP pipeline (not httpx), so SSL
    # verification must be disabled via RequestsTransport, not a verify kwarg.
    #
    # ChatCompletionsClient appends /chat/completions to the endpoint, so we
    # append /models here so the final path becomes /models/chat/completions,
    # which the gateway then strips the context prefix from and forwards to the
    # upstream Azure resource at /models/chat/completions.
    from azure.core.pipeline.transport import RequestsTransport
    transport = RequestsTransport(connection_verify=_VERIFY_SSL)
    return ChatCompletionsClient(
        endpoint=_BASE_URL + "/models",
        credential=AzureKeyCredential(_API_KEY),
        headers={"X-API-Key": _API_KEY},
        transport=transport,
    )


# ── Chat completions ──────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_basic(model):
    """Single-turn chat returns a non-empty assistant message."""
    logger.info("model=%s  endpoint=%s", model, _BASE_URL)
    client = _client()
    response = client.complete(
        model=model,
        messages=[UserMessage(content="Reply with the single word: Hello")],
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
    response = client.complete(
        model=model,
        messages=[
            SystemMessage(content="Answer only with yes or no."),
            UserMessage(content="Is the sky blue?"),
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
        UserMessage(content="My pet's name is Biscuit."),
        AssistantMessage(content="Got it, your pet is named Biscuit."),
        UserMessage(content="What is my pet's name?"),
    ]
    response = client.complete(model=model, messages=messages, max_tokens=30)
    content = response.choices[0].message.content or ""
    assert "biscuit" in content.lower(), f"Expected 'Biscuit' in reply, got: {content!r}"


@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_chat_completion_streaming(model):
    """Streaming delivers multiple chunks and assembles a valid response."""
    logger.info("model=%s  stream=true", model)
    client = _client()
    chunks = 0
    content = ""
    with client.complete(
        model=model,
        messages=[UserMessage(content="Say: one two three")],
        max_tokens=20,
        stream=True,
    ) as stream:
        for update in stream:
            if update.choices:
                delta = update.choices[0].delta.content
                if delta:
                    content += delta
                    chunks += 1

    assert chunks > 1
    assert content.strip()
    logger.info("chunks=%d  assembled=%r", chunks, content)
