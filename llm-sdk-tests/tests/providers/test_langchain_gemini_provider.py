"""
Gemini provider tests using the LangChain SDK (langchain-google-genai).

Requires: pip install langchain-google-genai

Uses ChatGoogleGenerativeAI with client_options to point at the gateway.

Gateway path rewriting (context = /gemini, upstream = Gemini API root):
  SDK constructs:  POST /gemini/v1beta/models/{model}:generateContent
  gateway strips:  /gemini
  gateway forwards: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent

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
    provider_base_url,
    provider_api_key,
    provider_models,
    get_verify_ssl,
)

# Skip entire module if langchain_google_genai is not installed
pytest.importorskip("langchain_google_genai", reason="langchain-google-genai not installed; run: pip install langchain-google-genai")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    pytest.mark.langchain,
    pytest.mark.skipif(not _ENABLED, reason="Gemini provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _llm(model: str) -> ChatGoogleGenerativeAI:
    kwargs = {}
    if not _VERIFY_SSL:
        # The google-genai SDK overwrites verify=False (falsy) with a real SSL
        # context. Pass an ssl.SSLContext with verification disabled instead —
        # it is truthy so the library keeps it, and httpx accepts it as verify=.
        import ssl
        _ctx = ssl.create_default_context()
        _ctx.check_hostname = False
        _ctx.verify_mode = ssl.CERT_NONE
        kwargs["client_args"] = {"verify": _ctx}
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=_API_KEY,
        client_options={"api_endpoint": _BASE_URL},
        additional_headers={"X-API-Key": _API_KEY},
        **kwargs,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_invoke_basic(model):
    """Basic single-turn invoke returns non-empty content."""
    logger.info("model=%s  endpoint=%s", model, _BASE_URL)
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
