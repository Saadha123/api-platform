"""
Mistral provider tests using the LangChain SDK via ChatOpenAI.

Requires: pip install langchain-openai

Mistral exposes an OpenAI-compatible API at /v1. LangChain's ChatOpenAI can
target it by setting base_url to the gateway context + /v1.

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
    get_provider,
    get_verify_ssl,
)

# Skip entire module if langchain_openai is not installed
pytest.importorskip("langchain_openai", reason="langchain-openai not installed; run: pip install langchain-openai")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "mistral"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_PCFG = get_provider(_cfg, _PROVIDER)
_BASE_URL = _PCFG.get("base_url", "")
_API_KEY = _PCFG.get("api_key", "")
_MODELS = _PCFG.get("models", [])
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.mistral,
    pytest.mark.langchain,
    pytest.mark.skipif(not _ENABLED, reason="Mistral provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _llm(model: str) -> ChatOpenAI:
    import httpx

    return ChatOpenAI(
        model=model,
        api_key=_API_KEY,
        base_url=_BASE_URL + "/v1",
        http_client=httpx.Client(verify=_VERIFY_SSL),
        default_headers={"X-API-Key": _API_KEY},
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_invoke_basic(model):
    """Basic single-turn invoke returns non-empty content."""
    logger.info("model=%s  endpoint=%s/v1", model, _BASE_URL)
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
        SystemMessage(content="Answer only with yes or no."),
        HumanMessage(content="Is the sky blue?"),
    ]
    response = llm.invoke(messages)
    assert response.content and str(response.content).strip()


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_invoke_multi_turn(model):
    """Multi-turn history is preserved across messages."""
    llm = _llm(model)
    messages = [
        HumanMessage(content="My favourite colour is blue. Remember that."),
        AIMessage(content="Noted, your favourite colour is blue."),
        HumanMessage(content="What is my favourite colour?"),
    ]
    response = llm.invoke(messages)
    assert "blue" in str(response.content).lower(), (
        f"Expected 'blue' in reply, got: {response.content!r}"
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
