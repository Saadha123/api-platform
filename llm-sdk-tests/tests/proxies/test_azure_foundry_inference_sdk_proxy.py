"""
Azure AI Foundry proxy tests – uses the azure-ai-inference SDK.

Requires: pip install azure-ai-inference

Each enabled proxy with provider_type=azure_foundry in config.yaml gets its own
parametrised run.  Tests mirror test_azure_foundry_inference_sdk_provider.py exactly.

The ChatCompletionsClient uses the Azure AI Foundry inference endpoint format:
  POST {endpoint}/models/chat/completions

Gateway path rewriting (context = /azure-foundry-proxy, upstream = resource root):
  SDK sends to:    /azure-foundry-proxy/models/chat/completions
  gateway strips:  /azure-foundry-proxy
  gateway forwards:https://{resource}.cognitiveservices.azure.com/models/chat/completions
"""

import logging
import pytest
from typing import Any, Dict, List
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

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


def _client(proxy: Dict[str, Any]) -> ChatCompletionsClient:
    # The inference SDK uses Azure Core's HTTP pipeline (not httpx), so SSL
    # verification must be disabled via RequestsTransport, not a verify kwarg.
    #
    # ChatCompletionsClient appends /chat/completions to the endpoint, so we
    # append /models here so the final path becomes /models/chat/completions,
    # which the gateway then strips the context prefix from and forwards to the
    # upstream Azure resource at /models/chat/completions.
    from azure.core.pipeline.transport import RequestsTransport
    api_key = proxy["api_key"]
    transport = RequestsTransport(connection_verify=_VERIFY_SSL)
    return ChatCompletionsClient(
        endpoint=proxy["base_url"] + "/models",
        credential=AzureKeyCredential(api_key),
        headers={"X-API-Key": api_key},
        transport=transport,
    )


# ── Chat completions ──────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_basic(proxy, model):
    """Single-turn chat returns a non-empty assistant message."""
    logger.info("proxy=%r  model=%s  endpoint=%s", proxy.get("name"), model, proxy.get("base_url"))
    client = _client(proxy)
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


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_with_system_message(proxy, model):
    """System message is honoured by the model."""
    client = _client(proxy)
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


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_multi_turn(proxy, model):
    """Multi-turn context is preserved across messages."""
    client = _client(proxy)
    messages = [
        UserMessage(content="My pet's name is Biscuit."),
        AssistantMessage(content="Got it, your pet is named Biscuit."),
        UserMessage(content="What is my pet's name?"),
    ]
    response = client.complete(model=model, messages=messages, max_tokens=30)
    content = response.choices[0].message.content or ""
    assert "biscuit" in content.lower(), f"Expected 'Biscuit' in reply, got: {content!r}"


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_streaming(proxy, model):
    """Streaming delivers multiple chunks and assembles a valid response."""
    logger.info("proxy=%r  model=%s  stream=true", proxy.get("name"), model)
    client = _client(proxy)
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
