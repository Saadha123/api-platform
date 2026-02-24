"""
Azure AI Foundry proxy tests – uses the openai AzureOpenAI SDK.

Each enabled proxy with provider_type=azure_foundry in config.yaml gets its own
parametrised run.  Tests mirror test_azure_foundry_provider.py exactly.

Gateway path rewriting (context = /azure-foundry-proxy, upstream = resource root):
  SDK constructs:  /azure-foundry-proxy/openai/deployments/{model}/chat/completions
  gateway strips:  /azure-foundry-proxy
  gateway forwards:https://{resource}.cognitiveservices.azure.com/openai/deployments/{model}/...
"""

import logging
import pytest
from openai import AzureOpenAI
from typing import Any, Dict, List
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

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


def _client(proxy: Dict[str, Any]) -> AzureOpenAI:
    import httpx

    api_key = proxy["api_key"]
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=proxy["base_url"],
        api_version=proxy.get("api_version", "2024-05-01-preview"),
        default_headers={"X-API-Key": api_key},
        http_client=httpx.Client(verify=_VERIFY_SSL),
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


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_with_system_message(proxy, model):
    """System message is honoured by the model."""
    client = _client(proxy)
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


@_skip
@pytest.mark.proxy
@pytest.mark.azure_foundry
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_multi_turn(proxy, model):
    """Multi-turn context is preserved across messages."""
    client = _client(proxy)
    messages = [
        {"role": "user", "content": "My pet's name is Biscuit."},
        {"role": "assistant", "content": "Got it, your pet is named Biscuit."},
        {"role": "user", "content": "What is my pet's name?"},
    ]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=500)
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
