"""
Azure AI Foundry provider tests using the LangChain Azure AI SDK (langchain-azure-ai).

Requires: pip install langchain-azure-ai

Uses AzureAIChatCompletionsModel which drives the Azure AI Model Inference endpoint:
  POST {endpoint}/chat/completions

The endpoint must include /models so the full path becomes /models/chat/completions.
Gateway path rewriting (context = /azure-foundry, upstream = resource root):
  endpoint passed: {base_url}/models
  SDK sends to:    /azure-foundry/models/chat/completions
  gateway strips:  /azure-foundry
  gateway forwards:https://{resource}.cognitiveservices.azure.com/models/chat/completions

Resources tested:
  - Basic invoke (single-turn)
  - Invoke with system message
  - Multi-turn conversation
  - Streaming
"""

import logging
import pytest
from utils.config import (
    load_config,
    is_provider_enabled,
    get_provider,
    get_verify_ssl,
)

# Skip entire module if langchain-azure-ai is not installed
pytest.importorskip(
    "langchain_azure_ai",
    reason="langchain-azure-ai not installed; run: pip install langchain-azure-ai",
)

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from azure.core.credentials import AzureKeyCredential
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    pytest.mark.langchain,
    pytest.mark.skipif(not _ENABLED, reason="Azure AI Foundry provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _llm(model: str) -> AzureAIChatCompletionsModel:
    # client_kwargs are forwarded to ChatCompletionsClient:
    #   transport  – use RequestsTransport to control SSL verification
    #   headers    – pass X-API-Key for WSO2 gateway subscription auth
    from azure.core.pipeline.transport import RequestsTransport

    return AzureAIChatCompletionsModel(
        endpoint=_BASE_URL + "/models",
        credential=AzureKeyCredential(_API_KEY),
        model=model,
        api_version=_API_VERSION,
        client_kwargs={
            "transport": RequestsTransport(connection_verify=_VERIFY_SSL),
            "headers": {"X-API-Key": _API_KEY},
        },
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_invoke_basic(model):
    """Basic single-turn invoke returns non-empty content."""
    logger.info("model=%s  endpoint=%s/models", model, _BASE_URL)
    llm = _llm(model)
    response = llm.invoke([HumanMessage(content="Reply with the single word: Hello")])
    assert response.content and str(response.content).strip()
    logger.info("response=%r", str(response.content)[:80])


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_invoke_with_system_message(model):
    """System message shapes the model's behaviour."""
    llm = _llm(model)
    messages = [
        SystemMessage(content="You are a helpful assistant that answers in one sentence."),
        HumanMessage(content="What is 2 + 2?"),
    ]
    response = llm.invoke(messages)
    assert response.content and str(response.content).strip()


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_invoke_multi_turn(model):
    """Multi-turn history is preserved across messages."""
    llm = _llm(model)
    messages = [
        HumanMessage(content="My pet's name is Biscuit."),
        AIMessage(content="Got it, your pet is named Biscuit."),
        HumanMessage(content="What is my pet's name?"),
    ]
    response = llm.invoke(messages)
    assert "biscuit" in str(response.content).lower(), (
        f"Expected 'Biscuit' in reply, got: {response.content!r}"
    )


@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_stream(model):
    """Streaming yields multiple chunks that assemble into a valid response."""
    logger.info("model=%s  stream=true", model)
    llm = _llm(model)
    chunks = list(llm.stream([HumanMessage(content="Count from 1 to 5, numbers only.")]))
    assert len(chunks) > 0
    content = "".join(str(c.content) for c in chunks if c.content)
    assert content.strip()
    logger.info("chunks=%d  assembled=%r", len(chunks), content)
