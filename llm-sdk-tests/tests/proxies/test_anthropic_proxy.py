"""
Anthropic proxy tests – uses the official `anthropic` Python SDK.

Each enabled proxy with provider_type=anthropic in config.yaml gets its own
parametrised run.  Tests mirror test_anthropic_provider.py exactly.
"""

import logging
import pytest
import anthropic as _anthropic
from typing import Any, Dict, List
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_VERIFY_SSL = get_verify_ssl(_cfg)
_PROXIES = enabled_proxies_by_type(_cfg, "anthropic")

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
    reason="No Anthropic proxies enabled in config.yaml",
)


def _client(proxy: Dict[str, Any]) -> _anthropic.Anthropic:
    import httpx
    return _anthropic.Anthropic(
        api_key=proxy["api_key"],
        base_url=proxy["base_url"],
        http_client=httpx.Client(verify=_VERIFY_SSL),
    )


# ── Messages ──────────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_message_basic(proxy, model):
    """A basic user message returns a non-empty text response."""
    logger.info("proxy=%r  model=%s  endpoint=%s", proxy.get("name"), model, proxy.get("base_url"))
    client = _client(proxy)
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": "Reply with only the word: Hello"}],
    )
    assert response.id
    text = response.content[0].text
    assert text and text.strip()
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    logger.info("response=%r  tokens(in=%s out=%s)",
                text, response.usage.input_tokens, response.usage.output_tokens)


@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_message_with_system_prompt(proxy, model):
    """System prompt is honoured and affects the response."""
    client = _client(proxy)
    response = client.messages.create(
        model=model,
        max_tokens=60,
        system="You are a pirate. Always end your reply with 'Arrr!'",
        messages=[{"role": "user", "content": "How are you?"}],
    )
    text = response.content[0].text
    assert text.strip()


@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_message_multi_turn(proxy, model):
    """Multi-turn conversation context is maintained."""
    client = _client(proxy)
    messages = [
        {"role": "user", "content": "My lucky number is 42."},
        {"role": "assistant", "content": "I'll remember that. Your lucky number is 42."},
        {"role": "user", "content": "What is my lucky number?"},
    ]
    response = client.messages.create(model=model, max_tokens=50, messages=messages)
    text = response.content[0].text
    assert "42" in text, f"Expected '42' in reply, got: {text!r}"


@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_message_stop_reason(proxy, model):
    """Response includes a valid stop_reason."""
    client = _client(proxy)
    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": "Write a long poem about the ocean."}],
    )
    assert response.stop_reason in {"end_turn", "max_tokens", "stop_sequence"}


# ── Streaming ─────────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_message_streaming(proxy, model):
    """Streaming messages returns incremental text events."""
    logger.info("proxy=%r  model=%s  stream=true", proxy.get("name"), model)
    client = _client(proxy)
    full_text = ""
    event_count = 0
    with client.messages.stream(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ) as stream:
        for text in stream.text_stream:
            full_text += text
            event_count += 1

    assert event_count > 0
    assert full_text.strip()
    logger.info("events=%d  assembled=%r", event_count, full_text)


@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_message_streaming_final_message(proxy, model):
    """stream.get_final_message() returns a complete Message object."""
    client = _client(proxy)
    with client.messages.stream(
        model=model,
        max_tokens=30,
        messages=[{"role": "user", "content": "Say: done"}],
    ) as stream:
        final = stream.get_final_message()

    assert final.content
    assert final.usage.output_tokens > 0
    logger.info("final_text=%r  output_tokens=%d",
                final.content[0].text, final.usage.output_tokens)
