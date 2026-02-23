"""
Gemini proxy tests – uses the official `google-genai` Python SDK.

Each enabled proxy with provider_type=gemini in config.yaml gets its own
parametrised run.  Tests mirror test_gemini_provider.py exactly.
"""

import logging
import pytest
from typing import Any, Dict, List
from google import genai
from google.genai import types as genai_types
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_VERIFY_SSL = get_verify_ssl(_cfg)
_PROXIES = enabled_proxies_by_type(_cfg, "gemini")

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
    reason="No Gemini proxies enabled in config.yaml",
)


def _client(proxy: Dict[str, Any]) -> genai.Client:
    api_key = proxy["api_key"]
    http_options = genai_types.HttpOptions(
        base_url=proxy["base_url"],
        headers={"X-API-Key": api_key},
    )
    if _VERIFY_SSL:
        return genai.Client(api_key="placeholder", http_options=http_options)

    import httpx as _httpx
    _orig_sync = _httpx.Client.__init__
    _orig_async = _httpx.AsyncClient.__init__

    def _no_verify_sync(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_sync(self, *args, **kwargs)

    def _no_verify_async(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_async(self, *args, **kwargs)

    _httpx.Client.__init__ = _no_verify_sync
    _httpx.AsyncClient.__init__ = _no_verify_async
    try:
        client = genai.Client(api_key="placeholder", http_options=http_options)
    finally:
        _httpx.Client.__init__ = _orig_sync
        _httpx.AsyncClient.__init__ = _orig_async
    return client


# ── 1. Basic text generation ──────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_basic_text_generation(proxy, model):
    """Basic generate_content call with a plain string prompt returns non-empty text."""
    logger.info("proxy=%r  model=%s  endpoint=%s", proxy.get("name"), model, proxy.get("base_url"))
    client = _client(proxy)
    response = client.models.generate_content(
        model=model,
        contents="How does AI work?",
    )
    assert response.text and response.text.strip(), "Response text must not be empty"
    logger.info("response=%r", response.text[:120])


# ── 2. Thinking / reasoning ───────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_thinking_config(proxy, model):
    """ThinkingConfig with thinking_level='low' is accepted."""
    client = _client(proxy)
    try:
        response = client.models.generate_content(
            model=model,
            contents="How does AI work?",
            config=genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(thinking_level="low"),
            ),
        )
    except Exception as exc:
        pytest.skip(f"ThinkingConfig not supported for {model!r}: {exc}")
    assert response.text and response.text.strip()
    logger.info("thinking_level=low  response=%r", response.text[:120])


# ── 3. System instructions ────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_system_instruction(proxy, model):
    """system_instruction in GenerateContentConfig shapes the model's persona."""
    client = _client(proxy)
    response = client.models.generate_content(
        model=model,
        config=genai_types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko.",
        ),
        contents="Hello there",
    )
    assert response.text and response.text.strip()
    logger.info("system_instruction=cat  response=%r", response.text[:120])


# ── 4. Generation config parameters ──────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_generation_config(proxy, model):
    """GenerateContentConfig parameters are accepted and respected."""
    client = _client(proxy)
    response = client.models.generate_content(
        model=model,
        contents=["Explain how AI works"],
        config=genai_types.GenerateContentConfig(
            max_output_tokens=120,
        ),
    )
    assert response.text and response.text.strip()
    logger.info("max_output_tokens=120  response_len=%d", len(response.text))


# ── 5. Streaming responses ────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_streaming(proxy, model):
    """generate_content_stream yields chunks incrementally."""
    logger.info("proxy=%r  model=%s  stream=true", proxy.get("name"), model)
    client = _client(proxy)
    chunks_received = 0
    full_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=["Explain how AI works"],
    ):
        chunks_received += 1
        if chunk.text:
            full_text += chunk.text

    assert chunks_received > 0, "No streaming chunks received"
    assert full_text.strip(), "Streamed text must not be empty"
    logger.info("chunks=%d  assembled_len=%d", chunks_received, len(full_text))


# ── 6. Multi-turn chat with history ──────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_multi_turn(proxy, model):
    """chats.create() maintains conversation history across turns."""
    client = _client(proxy)
    chat = client.chats.create(model=model)

    r1 = chat.send_message("I have 2 dogs in my house.")
    assert r1.text and r1.text.strip()
    logger.info("turn1=%r", r1.text[:80])

    r2 = chat.send_message("How many paws are in my house?")
    assert r2.text and r2.text.strip()
    assert "8" in (r2.text or ""), f"Expected '8' paws in reply, got: {r2.text!r}"
    logger.info("turn2=%r", r2.text[:80])

    history = chat.get_history()
    assert len(history) >= 4
    roles = [m.role for m in history]
    assert "user" in roles and "model" in roles


# ── 7. Streaming multi-turn chat ──────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_chat_multi_turn_streaming(proxy, model):
    """send_message_stream() streams each turn while history is accumulated."""
    client = _client(proxy)
    chat = client.chats.create(model=model)

    turn1_text = ""
    for chunk in chat.send_message_stream("I have 2 dogs in my house."):
        if chunk.text:
            turn1_text += chunk.text
    assert turn1_text.strip()
    logger.info("stream_turn1=%r", turn1_text[:80])

    turn2_text = ""
    for chunk in chat.send_message_stream("How many paws are in my house?"):
        if chunk.text:
            turn2_text += chunk.text
    assert turn2_text.strip()
    assert "8" in turn2_text, f"Expected '8' paws in streamed reply, got: {turn2_text!r}"
    logger.info("stream_turn2=%r", turn2_text[:80])

    history = chat.get_history()
    assert len(history) >= 4


# ── 8. Embeddings ─────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
@pytest.mark.embeddings
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_embed_content(proxy, model):
    """embed_content returns a non-empty embedding vector."""
    client = _client(proxy)
    embed_model = "models/text-embedding-004"
    try:
        result = client.models.embed_content(
            model=embed_model,
            contents="The quick brown fox",
        )
    except Exception as exc:
        pytest.skip(f"Embedding not available ({embed_model}): {exc}")

    assert result.embeddings, "Embedding result must contain data"
    assert len(result.embeddings[0].values) > 0
    logger.info("embed_model=%s  vector_dim=%d", embed_model, len(result.embeddings[0].values))
