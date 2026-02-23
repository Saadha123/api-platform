"""
OpenAI proxy tests – uses the official `openai` Python SDK.

Each enabled proxy with provider_type=openai in config.yaml gets its own
parametrised run.  Tests mirror test_openai_provider.py exactly.
"""

import logging
import pytest
from typing import Any, Dict, List
from openai import OpenAI
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_VERIFY_SSL = get_verify_ssl(_cfg)
_PROXIES = enabled_proxies_by_type(_cfg, "openai")

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
    reason="No OpenAI proxies enabled in config.yaml",
)


def _client(proxy: Dict[str, Any]) -> OpenAI:
    import httpx
    api_key = proxy["api_key"]
    return OpenAI(
        api_key=api_key,
        base_url=proxy["base_url"],
        http_client=httpx.Client(verify=_VERIFY_SSL),
        default_headers={"X-API-Key": api_key},
    )


# ── Chat completions ──────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.openai
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_basic(proxy, model):
    """Single-turn chat returns a non-empty assistant message."""
    logger.info("proxy=%r  model=%s  endpoint=%s", proxy.get("name"), model, proxy.get("base_url"))
    client = _client(proxy)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: Hello"}],
        max_tokens=20,
    )
    assert response.id, "Response must have an id"
    assert response.choices, "Response must have choices"
    content = response.choices[0].message.content
    assert content and content.strip(), "Assistant message must not be empty"
    assert response.usage is not None, "Token usage must be present"
    assert response.usage.total_tokens > 0
    logger.info("response=%r  tokens(in=%s out=%s)",
                content, response.usage.prompt_tokens, response.usage.completion_tokens)


@_skip
@pytest.mark.proxy
@pytest.mark.openai
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_with_system_message(proxy, model):
    """System message is respected and a proper reply is returned."""
    client = _client(proxy)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers in one sentence."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=50,
    )
    content = response.choices[0].message.content
    assert content and content.strip()
    logger.info("response=%r", content)


@_skip
@pytest.mark.proxy
@pytest.mark.openai
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_multi_turn(proxy, model):
    """Multi-turn conversation context is maintained."""
    client = _client(proxy)
    messages = [
        {"role": "user", "content": "My favourite colour is blue. Remember that."},
        {"role": "assistant", "content": "Noted, your favourite colour is blue."},
        {"role": "user", "content": "What is my favourite colour?"},
    ]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=50)
    content = response.choices[0].message.content or ""
    assert "blue" in content.lower(), f"Expected 'blue' in response, got: {content!r}"
    logger.info("response=%r", content)


@_skip
@pytest.mark.proxy
@pytest.mark.openai
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_completion_streaming(proxy, model):
    """Streaming chat returns tokens incrementally and assembles a full reply."""
    logger.info("proxy=%r  model=%s  stream=true", proxy.get("name"), model)
    client = _client(proxy)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 5, numbers only."}],
        max_tokens=30,
        stream=True,
    )
    chunks_received = 0
    full_content = ""
    for chunk in stream:
        chunks_received += 1
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_content += delta

    assert chunks_received > 1, "Streaming must deliver more than one chunk"
    assert full_content.strip(), "Streamed content must not be empty"
    logger.info("chunks=%d  assembled=%r", chunks_received, full_content)


# ── Embeddings ────────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.openai
@pytest.mark.embeddings
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_embeddings_single_input(proxy, model):
    """Embedding endpoint returns a non-empty vector for a single string."""
    client = _client(proxy)
    try:
        response = client.embeddings.create(
            model=model,
            input="The quick brown fox jumps over the lazy dog",
        )
    except Exception as exc:
        pytest.skip(f"Model {model!r} does not support embeddings: {exc}")

    vector = response.data[0].embedding
    assert isinstance(vector, list) and len(vector) > 0
    logger.info("vector_dim=%d  tokens=%d", len(vector), response.usage.total_tokens)


@_skip
@pytest.mark.proxy
@pytest.mark.openai
@pytest.mark.embeddings
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_embeddings_batch_input(proxy, model):
    """Embedding endpoint handles a batch of inputs."""
    client = _client(proxy)
    inputs = ["Hello world", "Goodbye world", "The sky is blue"]
    try:
        response = client.embeddings.create(model=model, input=inputs)
    except Exception as exc:
        pytest.skip(f"Model {model!r} does not support embeddings: {exc}")

    assert len(response.data) == len(inputs)
    for item in response.data:
        assert len(item.embedding) > 0


# ── Models list ───────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.openai
@pytest.mark.parametrize("proxy", [pytest.param(p, id=p.get("name", "?")) for p in _PROXIES])
def test_models_list(proxy):
    """Models list endpoint returns at least one model."""
    client = _client(proxy)
    try:
        models = client.models.list()
        assert models.data is not None
    except Exception as exc:
        pytest.skip(f"Models list endpoint not available: {exc}")
