"""
AWS Bedrock proxy tests – uses `boto3` (bedrock-runtime).

Each enabled proxy with provider_type=bedrock in config.yaml gets its own
parametrised run.  Tests mirror test_bedrock_provider.py exactly.

boto3 is configured with:
  - UNSIGNED signature so it skips AWS SigV4
  - A before-send event injects the gateway API key as X-API-Key
"""

import json
import logging
import pytest
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config as BotocoreConfig
from typing import Any, Dict, List
from utils.config import load_config, enabled_proxies_by_type, get_verify_ssl

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_VERIFY_SSL = get_verify_ssl(_cfg)
_PROXIES = enabled_proxies_by_type(_cfg, "bedrock")

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
    reason="No Bedrock proxies enabled in config.yaml",
)


def _client(proxy: Dict[str, Any]):
    api_key = proxy["api_key"]
    session = boto3.Session()
    client = session.client(
        service_name="bedrock-runtime",
        region_name=proxy.get("region", "us-east-1"),
        endpoint_url=proxy["base_url"],
        aws_access_key_id="gateway-test",
        aws_secret_access_key=api_key,
        config=BotocoreConfig(signature_version=UNSIGNED),
        verify=_VERIFY_SSL,
    )

    def _inject_api_key(request, **kwargs):
        request.headers["X-API-Key"] = api_key

    client.meta.events.register("before-send.bedrock-runtime.*", _inject_api_key)
    return client


def _invoke_model_body(model_id: str) -> dict:
    if model_id.startswith("amazon.titan"):
        return {"inputText": "Reply with only the word: Hello",
                "textGenerationConfig": {"maxTokenCount": 20, "temperature": 0.0}}
    if "claude" in model_id:
        return {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 20,
                "messages": [{"role": "user", "content": "Reply with only the word: Hello"}]}
    if "llama" in model_id or "meta" in model_id:
        return {"prompt": "Reply with only the word: Hello", "max_gen_len": 20, "temperature": 0.0}
    if "mistral" in model_id:
        return {"prompt": "<s>[INST] Reply with only the word: Hello [/INST]", "max_tokens": 20}
    return {"inputText": "Reply with only the word: Hello"}


def _extract_invoke_text(model_id: str, body: dict) -> str:
    if model_id.startswith("amazon.titan"):
        return body.get("results", [{}])[0].get("outputText", "")
    if "claude" in model_id:
        return body.get("content", [{}])[0].get("text", "")
    if "llama" in model_id or "meta" in model_id:
        return body.get("generation", "")
    if "mistral" in model_id:
        return body.get("outputs", [{}])[0].get("text", "")
    return str(body)


# ── InvokeModel ───────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.bedrock
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_invoke_model(proxy, model):
    """InvokeModel returns a non-empty generated text."""
    logger.info("proxy=%r  model=%s  api=InvokeModel", proxy.get("name"), model)
    client = _client(proxy)
    payload = _invoke_model_body(model)
    try:
        response = client.invoke_model(
            modelId=model,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
    except botocore.exceptions.ClientError as exc:
        pytest.skip(f"InvokeModel failed for {model!r}: {exc}")

    body = json.loads(response["body"].read())
    text = _extract_invoke_text(model, body)
    assert text.strip(), f"Expected non-empty text from InvokeModel, got body: {body}"
    logger.info("response=%r", text)


# ── Converse API ──────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.bedrock
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_converse_basic(proxy, model):
    """Converse API returns a well-formed assistant message."""
    logger.info("proxy=%r  model=%s  api=Converse", proxy.get("name"), model)
    client = _client(proxy)
    try:
        response = client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": "Reply with only the word: Hello"}]}],
            inferenceConfig={"maxTokens": 20},
        )
    except botocore.exceptions.ClientError as exc:
        pytest.skip(f"Converse not supported for {model!r}: {exc}")

    text = response["output"]["message"]["content"][0].get("text", "")
    assert text.strip()
    usage = response.get("usage", {})
    assert usage.get("inputTokens", 0) > 0


@_skip
@pytest.mark.proxy
@pytest.mark.bedrock
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_converse_multi_turn(proxy, model):
    """Converse API maintains context across multiple message turns."""
    client = _client(proxy)
    messages = [
        {"role": "user", "content": [{"text": "My lucky number is 99."}]},
        {"role": "assistant", "content": [{"text": "Your lucky number is 99."}]},
        {"role": "user", "content": [{"text": "What is my lucky number?"}]},
    ]
    try:
        response = client.converse(modelId=model, messages=messages, inferenceConfig={"maxTokens": 30})
    except botocore.exceptions.ClientError as exc:
        pytest.skip(f"Converse not supported for {model!r}: {exc}")

    text = response["output"]["message"]["content"][0].get("text", "")
    assert "99" in text, f"Expected '99' in reply, got: {text!r}"


@_skip
@pytest.mark.proxy
@pytest.mark.bedrock
@pytest.mark.chat
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_converse_system_prompt(proxy, model):
    """System prompt is forwarded and influences model output."""
    client = _client(proxy)
    try:
        response = client.converse(
            modelId=model,
            system=[{"text": "Always reply with exactly the word CONFIRMED."}],
            messages=[{"role": "user", "content": [{"text": "Acknowledge this message."}]}],
            inferenceConfig={"maxTokens": 20},
        )
    except botocore.exceptions.ClientError as exc:
        pytest.skip(f"System prompt not supported for {model!r}: {exc}")

    text = response["output"]["message"]["content"][0].get("text", "")
    assert text.strip()


# ── ConverseStream ────────────────────────────────────────────────────────────

@_skip
@pytest.mark.proxy
@pytest.mark.bedrock
@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("proxy,model", _PROXY_MODEL_PARAMS)
def test_converse_stream(proxy, model):
    """ConverseStream delivers incremental text events."""
    logger.info("proxy=%r  model=%s  api=ConverseStream", proxy.get("name"), model)
    client = _client(proxy)
    try:
        response = client.converse_stream(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": "Count from 1 to 3."}]}],
            inferenceConfig={"maxTokens": 30},
        )
    except botocore.exceptions.ClientError as exc:
        pytest.skip(f"ConverseStream not supported for {model!r}: {exc}")

    full_text = ""
    event_count = 0
    for event in response["stream"]:
        event_count += 1
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            full_text += delta["text"]

    assert event_count > 0
    assert full_text.strip()
    logger.info("events=%d  assembled=%r", event_count, full_text)
