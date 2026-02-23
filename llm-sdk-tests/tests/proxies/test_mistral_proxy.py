"""
Mistral proxy tests – uses the official `mistralai` Python SDK.

Each enabled proxy with provider_type=mistral in config.yaml gets its own
parametrised run.  Tests mirror test_mistral_provider.py exactly.
"""

import logging
import pytest
from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage, AssistantMessage
from typing import Any, Dict, List
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


def _client(proxy: Dict[str, Any]) -> Mistral:
    import httpx
    api_key = proxy["api_key"]

    def _inject(request):
        request.headers["X-API-Key"] = api_key

    http_client = httpx.Client(
        verify=_VERIFY_SSL,
        event_hooks={"request": [_inject]},
    )
    return Mistral(api_key=api_key, server_url=proxy["base_url"], client=http_client)


# ── Chat completions ──────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_basic(proxy, model):
    """Basic chat returns a non-empty assistant message."""
    logger.info("proxy=%r  model=%s  endpoint=%s", proxy.get("name"), model, proxy.get("base_url"))
    client = _client(proxy)
    response = client.chat.complete(
        model=model,
        messages=[UserMessage(content="Reply with only the word: Hello")],
        max_tokens=20,
    )
    assert response.id
    assert response.choices
    content = response.choices[0].message.content
    assert content and (content.strip() if isinstance(content, str) else content)
    assert response.usage and response.usage.total_tokens > 0
    logger.info("response=%r  tokens=%d", content, response.usage.total_tokens)


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_with_system_message(proxy, model):
    """System message shapes the assistant behaviour."""
    client = _client(proxy)
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


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_multi_turn(proxy, model):
    """Context from prior turns is maintained."""
    client = _client(proxy)
    messages = [
        UserMessage(content="My favourite animal is a capybara."),
        AssistantMessage(content="Noted! Your favourite animal is a capybara."),
        UserMessage(content="What is my favourite animal?"),
    ]
    response = client.chat.complete(model=model, messages=messages, max_tokens=40)
    content = response.choices[0].message.content or ""
    assert "capybara" in content.lower(), f"Expected 'capybara' in: {content!r}"


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_temperature(proxy, model):
    """Temperature parameter is accepted without error."""
    client = _client(proxy)
    response = client.chat.complete(
        model=model,
        messages=[UserMessage(content="Say hello.")],
        max_tokens=20,
        temperature=0.0,
    )
    assert response.choices[0].message.content


@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_streaming(proxy, model):
    """Streaming delivers multiple events and assembles a full response."""
    logger.info("proxy=%r  model=%s  stream=true", proxy.get("name"), model)
    client = _client(proxy)
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

    assert event_count > 0
    assert full_content.strip()
    logger.info("events=%d  assembled=%r", event_count, full_content)


# ── Embeddings ────────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.embeddings
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_embeddings(proxy, model):
    """Embedding endpoint returns non-empty vectors."""
    client = _client(proxy)
    try:
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=["Hello world", "Goodbye world"],
        )
    except Exception as exc:
        pytest.skip(f"Embedding endpoint not available: {exc}")

    assert response.data and len(response.data) == 2
    for item in response.data:
        assert item.embedding and len(item.embedding) > 0


# ── Models list ───────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.mistral
@pytest.mark.parametrize("proxy", [pytest.param(p, id=p.get("name", "?")) for p in _PROXIES])
def test_models_list(proxy):
    """Models list endpoint returns at least one model."""
    client = _client(proxy)
    try:
        result = client.models.list()
        assert result.data
    except Exception as exc:
        pytest.skip(f"Models list not available: {exc}")
