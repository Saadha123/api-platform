"""
Anthropic proxy tests using the LangChain SDK (langchain-anthropic).

Requires: pip install langchain-anthropic

Each enabled proxy with provider_type=anthropic in config.yaml gets its own
parametrised run.

Uses ChatAnthropic with anthropic_api_url to point at the gateway proxy.
"""

import logging
import pytest
from typing import Any, Dict, List
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

# Skip entire module if langchain_anthropic is not installed
pytest.importorskip("langchain_anthropic", reason="langchain-anthropic not installed; run: pip install langchain-anthropic")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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


def _llm(proxy: Dict[str, Any], model: str) -> ChatAnthropic:
    api_key = proxy["api_key"]

    if _VERIFY_SSL:
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            anthropic_api_url=proxy["base_url"],
            default_headers={"X-API-Key": api_key},
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
            api_key=api_key,
            anthropic_api_url=proxy["base_url"],
            default_headers={"X-API-Key": api_key},
            max_tokens=1024,
        )
        _ = llm._client  # force client creation while patch is active
    finally:
        _anthropic.DefaultHttpxClient.__init__ = _orig
    return llm


# ── Tests ─────────────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_invoke_basic(proxy, model):
    """Basic single-turn invoke returns non-empty content."""
    logger.info("proxy=%r  model=%s  endpoint=%s", proxy.get("name"), model, proxy.get("base_url"))
    llm = _llm(proxy, model)
    response = llm.invoke([HumanMessage(content="Reply with the single word: Hello")])
    assert response.content and str(response.content).strip()
    logger.info("response=%r", str(response.content)[:80])


@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_invoke_with_system_message(proxy, model):
    """System message shapes the model's behaviour."""
    llm = _llm(proxy, model)
    messages = [
        SystemMessage(content="You are a helpful assistant that answers in one sentence."),
        HumanMessage(content="What is 2 + 2?"),
    ]
    response = llm.invoke(messages)
    assert response.content and str(response.content).strip()


@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_invoke_multi_turn(proxy, model):
    """Multi-turn history is preserved across messages."""
    llm = _llm(proxy, model)
    messages = [
        HumanMessage(content="My pet's name is Biscuit."),
        AIMessage(content="Got it, your pet is named Biscuit."),
        HumanMessage(content="What is my pet's name?"),
    ]
    response = llm.invoke(messages)
    assert "biscuit" in str(response.content).lower(), (
        f"Expected 'Biscuit' in reply, got: {response.content!r}"
    )


@_skip
@pytest.mark.proxy
@pytest.mark.anthropic
@pytest.mark.langchain
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_stream(proxy, model):
    """Streaming yields multiple chunks that assemble into a valid response."""
    logger.info("proxy=%r  model=%s  stream=true", proxy.get("name"), model)
    llm = _llm(proxy, model)
    chunks = list(llm.stream([HumanMessage(content="Count from 1 to 5, numbers only.")]))
    assert len(chunks) > 0
    content = "".join(str(c.content) for c in chunks if c.content)
    assert content.strip()
    logger.info("chunks=%d  assembled=%r", len(chunks), content)
