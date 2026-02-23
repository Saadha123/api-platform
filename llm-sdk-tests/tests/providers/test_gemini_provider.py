"""
Google Gemini provider tests – uses the official `google-genai` Python SDK.

Covers text-generation SDK patterns from the Gemini API documentation:
  1. Basic text generation
  2. System instructions
  3. Generation config parameters (max_output_tokens)
  4. Streaming responses (generate_content_stream)
  5. Multi-turn chat with history verification
  6. Streaming multi-turn chat with history verification
"""

import logging
import pytest
from google import genai
from google.genai import types as genai_types
from utils.config import (
    load_config,
    is_provider_enabled,
    provider_base_url,
    provider_api_key,
    provider_models,
    get_verify_ssl,
)

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "gemini"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_BASE_URL = provider_base_url(_cfg, _PROVIDER)
_API_KEY = provider_api_key(_cfg, _PROVIDER)
_MODELS = provider_models(_cfg, _PROVIDER)
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.gemini,
    pytest.mark.skipif(not _ENABLED, reason="Gemini provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client() -> genai.Client:
    # The gateway authenticates clients via the X-API-Key header (configured in
    # the Security tab of the provider).  The google-genai SDK normally sends the
    # key as x-goog-api-key, so we inject it as a custom header instead and pass
    # a placeholder for the SDK's own api_key parameter.
    http_options = genai_types.HttpOptions(
        base_url=_BASE_URL,
        headers={"X-API-Key": _API_KEY},
    )
    if _VERIFY_SSL:
        return genai.Client(api_key="placeholder", http_options=http_options)

    # google-genai creates its own internal httpx.Client and AsyncClient with no
    # direct API to pass verify=False through HttpOptions.  Monkeypatch both
    # constructors for the duration of genai.Client.__init__ only.
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

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_basic_text_generation(model):
    """
    Basic generate_content call with a plain string prompt returns non-empty text.

    SDK pattern:
        response = client.models.generate_content(model=..., contents="...")
        print(response.text)
    """
    logger.info("model=%s  endpoint=%s", model, _BASE_URL)
    client = _client()
    response = client.models.generate_content(
        model=model,
        contents="How does AI work?",
    )
    assert response.text and response.text.strip(), "Response text must not be empty"
    logger.info("response=%r", response.text[:120])


# ── 2. System instructions ────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_system_instruction(model):
    """
    system_instruction in GenerateContentConfig shapes the model's persona.

    SDK pattern:
        config=types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko.")
    """
    client = _client()
    response = client.models.generate_content(
        model=model,
        config=genai_types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko.",
        ),
        contents="Hello there",
    )
    assert response.text and response.text.strip()
    logger.info("system_instruction=cat  response=%r", response.text[:120])


# ── 3. Generation config parameters ──────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_generation_config(model):
    """
    GenerateContentConfig parameters (max_output_tokens) are accepted and
    respected.  Temperature is intentionally left at the default (1.0) because
    the Gemini docs note that changing it on Gemini 3 models can cause
    unexpected behaviour.

    SDK pattern:
        config=types.GenerateContentConfig(temperature=0.1)
    """
    client = _client()
    response = client.models.generate_content(
        model=model,
        contents=["Explain how AI works"],
        config=genai_types.GenerateContentConfig(
            max_output_tokens=120,
        ),
    )
    assert response.text and response.text.strip()
    logger.info("max_output_tokens=120  response_len=%d", len(response.text))


# ── 4. Streaming responses ────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_streaming(model):
    """
    generate_content_stream yields GenerateContentResponse chunks incrementally.

    SDK pattern:
        response = client.models.generate_content_stream(model=..., contents=[...])
        for chunk in response:
            print(chunk.text, end="")
    """
    logger.info("model=%s  stream=true", model)
    client = _client()
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


# ── 5. Multi-turn chat with history ──────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_chat_multi_turn(model):
    """
    chats.create() maintains conversation history across turns.
    get_history() returns all user and model messages in order.

    SDK pattern:
        chat = client.chats.create(model=...)
        response = chat.send_message("I have 2 dogs in my house.")
        response = chat.send_message("How many paws are in my house?")
        for message in chat.get_history():
            print(f'role - {message.role}', end=': ')
            print(message.parts[0].text)
    """
    client = _client()
    chat = client.chats.create(model=model)

    r1 = chat.send_message("I have 2 dogs in my house.")
    assert r1.text and r1.text.strip(), "First reply must not be empty"
    logger.info("turn1=%r", r1.text[:80])

    r2 = chat.send_message("How many paws are in my house?")
    assert r2.text and r2.text.strip(), "Second reply must not be empty"
    assert "8" in (r2.text or ""), f"Expected '8' paws in reply, got: {r2.text!r}"
    logger.info("turn2=%r", r2.text[:80])

    history = chat.get_history()
    assert len(history) >= 4, (
        f"History should have ≥4 messages (2 user + 2 model), got {len(history)}"
    )
    roles = [m.role for m in history]
    assert "user" in roles and "model" in roles, f"Unexpected roles in history: {roles}"
    logger.info("history_length=%d  roles=%s", len(history), roles)


# ── 6. Streaming multi-turn chat ──────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_chat_multi_turn_streaming(model):
    """
    send_message_stream() streams each turn while the chat object still
    accumulates the full conversation history.

    SDK pattern:
        chat = client.chats.create(model=...)
        response = chat.send_message_stream("I have 2 dogs in my house.")
        for chunk in response:
            print(chunk.text, end="")
        response = chat.send_message_stream("How many paws are in my house?")
        for chunk in response:
            print(chunk.text, end="")
        for message in chat.get_history():
            print(f'role - {message.role}', end=': ')
            print(message.parts[0].text)
    """
    client = _client()
    chat = client.chats.create(model=model)

    # First streaming turn
    turn1_text = ""
    for chunk in chat.send_message_stream("I have 2 dogs in my house."):
        if chunk.text:
            turn1_text += chunk.text
    assert turn1_text.strip(), "First streaming turn must return text"
    logger.info("stream_turn1=%r", turn1_text[:80])

    # Second streaming turn — model should recall context from turn 1
    turn2_text = ""
    for chunk in chat.send_message_stream("How many paws are in my house?"):
        if chunk.text:
            turn2_text += chunk.text
    assert turn2_text.strip(), "Second streaming turn must return text"
    assert "8" in turn2_text, f"Expected '8' paws in streamed reply, got: {turn2_text!r}"
    logger.info("stream_turn2=%r", turn2_text[:80])

    # History should contain all turns
    history = chat.get_history()
    assert len(history) >= 4, (
        f"History should have ≥4 messages after 2 streaming turns, got {len(history)}"
    )
    roles = [m.role for m in history]
    logger.info("history_length=%d  roles=%s", len(history), roles)

