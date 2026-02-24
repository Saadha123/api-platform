"""
Gemini proxy tests using the LangChain SDK (langchain-google-genai).

Requires: pip install langchain-google-genai

Each enabled proxy with provider_type=gemini in config.yaml gets its own
parametrised run.

Uses ChatGoogleGenerativeAI with client_options to point at the gateway proxy.

Gateway path rewriting (context = /gemini-proxy, upstream = Gemini API root):
  SDK constructs:  POST /gemini-proxy/v1beta/models/{model}:generateContent
  gateway strips:  /gemini-proxy
  gateway forwards: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
"""

import logging
import pytest
from typing import Any, Dict, List
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

# Skip entire module if langchain_google_genai is not installed
pytest.importorskip("langchain_google_genai", reason="langchain-google-genai not installed; run: pip install langchain-google-genai")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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


def _llm(proxy: Dict[str, Any], model: str) -> ChatGoogleGenerativeAI:
    api_key = proxy["api_key"]
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
        google_api_key=api_key,
        client_options={"api_endpoint": proxy["base_url"]},
        additional_headers={"X-API-Key": api_key},
        **kwargs,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.gemini
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
@pytest.mark.gemini
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
@pytest.mark.gemini
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
@pytest.mark.gemini
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
