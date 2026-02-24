"""
Azure AI Foundry proxy tests using the LangChain Azure AI SDK (langchain-azure-ai).

Requires: pip install langchain-azure-ai

Each enabled proxy with provider_type=azure_foundry in config.yaml gets its own
parametrised run.

Uses AzureAIChatCompletionsModel which drives the Azure AI Model Inference endpoint:
  POST {endpoint}/chat/completions

The endpoint must include /models so the full path becomes /models/chat/completions.
Gateway path rewriting (context = /azure-foundry-proxy, upstream = resource root):
  endpoint passed: {base_url}/models
  SDK sends to:    /azure-foundry-proxy/models/chat/completions
  gateway strips:  /azure-foundry-proxy
  gateway forwards:https://{resource}.cognitiveservices.azure.com/models/chat/completions
"""

import logging
import pytest
from typing import Any, Dict, List
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

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
_VERIFY_SSL = get_verify_ssl(_cfg)
_PROXIES = enabled_proxies_by_type(_cfg, "azure_foundry")

logger = logging.getLogger(__name__)


def _proxy_model_params() -> List[pytest.param]:
    params = []
    for proxy in _PROXIES:
        for model in proxy.get("models", []):
            label = f"{proxy.get('name', '?')} / {model}"
            params.append(pytest.param(proxy, model, id=label))
    return params


_PROXY_MODEL_PARAMS = _proxy_model_params()

_skip = pytest.mark.skipif(
    len(_PROXY_MODEL_PARAMS) == 0,
    reason="No Azure AI Foundry proxies enabled in config.yaml",
)


def _llm(proxy: Dict[str, Any], model: str) -> AzureAIChatCompletionsModel:
    # client_kwargs are forwarded to ChatCompletionsClient:
    #   transport  – use RequestsTransport to control SSL verification
    #   headers    – pass X-API-Key for WSO2 gateway subscription auth
    from azure.core.pipeline.transport import RequestsTransport

    api_key = proxy["api_key"]
    return AzureAIChatCompletionsModel(
        endpoint=proxy["base_url"] + "/models",
        credential=AzureKeyCredential(api_key),
        model=model,
        api_version=proxy.get("api_version", "2024-05-01-preview"),
        client_kwargs={
            "transport": RequestsTransport(connection_verify=_VERIFY_SSL),
            "headers": {"X-API-Key": api_key},
        },
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_invoke_basic(proxy, model):
    """Basic single-turn invoke returns non-empty content."""
    logger.info("proxy=%r  model=%s  endpoint=%s/models", proxy.get("name"), model, proxy.get("base_url"))
    llm = _llm(proxy, model)
    response = llm.invoke([HumanMessage(content="Reply with the single word: Hello")])
    assert response.content and str(response.content).strip()
    logger.info("response=%r", str(response.content)[:80])


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_invoke_with_system_message(proxy, model):
    """System message shapes the model's behaviour."""
    llm = _llm(proxy, model)
    messages = [
        SystemMessage(content="You are a helpful assistant that answers in one sentence."),
        HumanMessage(content="What is 2 + 2?"),
    ]
    response = llm.invoke(messages)
    assert response.content and str(response.content).strip()


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_invoke_multi_turn(proxy, model):
    """Multi-turn history is preserved across messages."""
    llm = _llm(proxy, model)
    messages = [
        HumanMessage(content="My pet's name is Biscuit."),
        AIMessage(content="Got it, your pet is named Biscuit."),
        HumanMessage(content="What is my pet's name?"),
    ]
    response = llm.invoke(messages)
    assert "biscuit" in str(response.content).lower(), (
        f"Expected 'Biscuit' in reply, got: {response.content!r}"
    )


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_stream(proxy, model):
    """Streaming yields multiple chunks that assemble into a valid response."""
    logger.info("proxy=%r  model=%s  stream=true", proxy.get("name"), model)
    llm = _llm(proxy, model)
    chunks = list(llm.stream([HumanMessage(content="Count from 1 to 5, numbers only.")]))
    assert len(chunks) > 0
    content = "".join(str(c.content) for c in chunks if c.content)
    assert content.strip()
    logger.info("chunks=%d  assembled=%r", len(chunks), content)
