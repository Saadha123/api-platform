"""
Anthropic provider tests using the LangChain SDK (langchain-anthropic).

Requires: pip install langchain-anthropic

Uses ChatAnthropic with anthropic_api_url to point at the gateway.

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

# Skip entire module if langchain_anthropic is not installed
pytest.importorskip("langchain_anthropic", reason="langchain-anthropic not installed; run: pip install langchain-anthropic")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "anthropic"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_BASE_URL = provider_base_url(_cfg, _PROVIDER)
_API_KEY = provider_api_key(_cfg, _PROVIDER)
_MODELS = provider_models(_cfg, _PROVIDER)
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.anthropic,
    pytest.mark.langchain,
    pytest.mark.skipif(not _ENABLED, reason="Anthropic provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _llm(model: str) -> ChatAnthropic:
    if _VERIFY_SSL:
        return ChatAnthropic(
            model=model,
            api_key=_API_KEY,
            anthropic_api_url=_BASE_URL,
            default_headers={"X-API-Key": _API_KEY},
            max_tokens=1024,
        )

    # ChatAnthropic builds its own httpx client internally via a cached factory
    # and doesn't expose a verify= parameter. anthropic.DefaultHttpxClient.__init__
    # creates the HTTPTransport (with verify) before calling super().__init__, so
    # we must patch at that level — not httpx.Client.__init__.
    import anthropic as _anthropic
    import langchain_anthropic._client_utils as _cu

    _cu._get_default_httpx_client.cache_clear()
    _orig = _anthropic.DefaultHttpxClient.__init__

    def _no_verify(self, **kwargs):
        kwargs["verify"] = False
        _orig(self, **kwargs)

    _anthropic.DefaultHttpxClient.__init__ = _no_verify
    try:
        llm = ChatAnthropic(
            model=model,
            api_key=_API_KEY,
            anthropic_api_url=_BASE_URL,
            default_headers={"X-API-Key": _API_KEY},
            max_tokens=1024,
        )
        _ = llm._client  # force client creation while patch is active
    finally:
        _anthropic.DefaultHttpxClient.__init__ = _orig
    return llm


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
