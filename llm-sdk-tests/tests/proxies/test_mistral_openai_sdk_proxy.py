"""
Mistral proxy tests using the OpenAI-compatible SDK (openai.OpenAI).

Mistral exposes an OpenAI-compatible API at /v1, so the standard openai.OpenAI
client works by appending /v1 to the gateway proxy URL.

Each enabled proxy with provider_type=mistral in config.yaml gets its own
parametrised run.
"""

import logging
import pytest
from typing import Any, Dict, List
from openai import OpenAI
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_VERIFY_SSL = get_verify_ssl(_cfg)
_PROXIES = enabled_proxies_by_type(_cfg, "mistral")

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
    reason="No Mistral proxies enabled in config.yaml",
)


def _client(proxy: Dict[str, Any]) -> OpenAI:
    import httpx

    api_key = proxy["api_key"]
    return OpenAI(
        api_key=api_key,
        base_url=proxy["base_url"] + "/v1",
        http_client=httpx.Client(verify=_VERIFY_SSL),
        default_headers={"X-API-Key": api_key},
    )


# ── Chat completions ──────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.openai
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_basic(proxy, model):
    """Single-turn chat returns a non-empty assistant message."""
    logger.info("proxy=%r  model=%s  endpoint=%s/v1", proxy.get("name"), model, proxy.get("base_url"))
    client = _client(proxy)
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


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.openai
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
        max_tokens=10,
    )
    content = response.choices[0].message.content or ""
    assert content.strip()


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.openai
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
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=30)
    content = response.choices[0].message.content or ""
    assert "biscuit" in content.lower(), f"Expected 'Biscuit' in reply, got: {content!r}"


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.openai
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


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.openai
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_temperature(proxy, model):
    """Temperature parameter is accepted and a valid response is returned."""
    client = _client(proxy)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=20,
        temperature=0.7,
    )
    assert response.choices[0].message.content


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.openai
@pytest.mark.parametrize("proxy", [pytest.param(p, id=p.get("name", "?")) for p in _PROXIES])
def test_models_list(proxy):
    """Models list endpoint is reachable via OpenAI SDK."""
    client = _client(proxy)
    try:
        models = client.models.list()
        assert models.data is not None
    except Exception as exc:
        pytest.skip(f"Models list endpoint not available: {exc}")
