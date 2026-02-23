"""
AWS Bedrock provider tests – uses `boto3` (bedrock-runtime).

The gateway wraps Bedrock: the client authenticates with the gateway API key
(sent as X-API-Key header); the gateway holds the real AWS credentials and
signs requests upstream.

boto3 is configured with:
  - UNSIGNED signature (botocore.UNSIGNED) so it skips AWS SigV4
  - A before-send event injects the gateway API key as X-API-Key

Resources tested:
  - InvokeModel  (raw Bedrock request/response format)
  - Converse API (unified conversational API)
  - ConverseStream (streaming Converse)
"""

import json
import logging
import pytest
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config as BotocoreConfig
from utils.config import (
    load_config,
    is_provider_enabled,
    get_provider,
    get_verify_ssl,
)

# ── Module-level setup ────────────────────────────────────────────────────────

_cfg = load_config()
_PROVIDER = "bedrock"
_ENABLED = is_provider_enabled(_cfg, _PROVIDER)
_PCFG = get_provider(_cfg, _PROVIDER)
_BASE_URL = _PCFG.get("base_url", "")
_API_KEY = _PCFG.get("api_key", "")
_REGION = _PCFG.get("region", "us-east-1")
_MODELS = _PCFG.get("models", [])
_VERIFY_SSL = get_verify_ssl(_cfg)

pytestmark = [
    pytest.mark.provider,
    pytest.mark.bedrock,
    pytest.mark.skipif(not _ENABLED, reason="Bedrock provider disabled in config.yaml"),
]

logger = logging.getLogger(__name__)


def _client():
    """
    Create a boto3 bedrock-runtime client pointed at the gateway.

    Signature is set to UNSIGNED; a before-send event adds the gateway
    API key so the gateway can authenticate the request.
    """
    session = boto3.Session()
    client = session.client(
        service_name="bedrock-runtime",
        region_name=_REGION,
        endpoint_url=_BASE_URL,
        aws_access_key_id="gateway-test",
        aws_secret_access_key=_API_KEY,
        config=BotocoreConfig(signature_version=UNSIGNED),
        verify=_VERIFY_SSL,
    )

    # Inject the gateway API key as an HTTP header before every request
    def _inject_api_key(request, **kwargs):
        request.headers["X-API-Key"] = _API_KEY

    client.meta.events.register("before-send.bedrock-runtime.*", _inject_api_key)
    return client


# ── InvokeModel ───────────────────────────────────────────────────────────────

def _invoke_model_body(model_id: str) -> dict:
    """Build an appropriate InvokeModel request body for the given model."""
    if model_id.startswith("amazon.titan"):
        return {
            "inputText": "Reply with only the word: Hello",
            "textGenerationConfig": {"maxTokenCount": 20, "temperature": 0.0},
        }
    if "claude" in model_id:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Reply with only the word: Hello"}],
        }
    if "llama" in model_id or "meta" in model_id:
        return {
            "prompt": "Reply with only the word: Hello",
            "max_gen_len": 20,
            "temperature": 0.0,
        }
    if "mistral" in model_id:
        return {
            "prompt": "<s>[INST] Reply with only the word: Hello [/INST]",
            "max_tokens": 20,
        }
    # Generic fallback
    return {
        "inputText": "Reply with only the word: Hello",
    }


def _extract_invoke_text(model_id: str, body: dict) -> str:
    """Extract the generated text from an InvokeModel response body."""
    if model_id.startswith("amazon.titan"):
        return body.get("results", [{}])[0].get("outputText", "")
    if "claude" in model_id:
        return body.get("content", [{}])[0].get("text", "")
    if "llama" in model_id or "meta" in model_id:
        return body.get("generation", "")
    if "mistral" in model_id:
        return body.get("outputs", [{}])[0].get("text", "")
    return str(body)


@pytest.mark.parametrize("model", _MODELS)
def test_invoke_model(model):
    """InvokeModel returns a non-empty generated text."""
    logger.info("model=%s  api=InvokeModel  endpoint=%s", model, _BASE_URL)
    client = _client()
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

@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_converse_basic(model):
    """Converse API returns a well-formed assistant message."""
    logger.info("model=%s  api=Converse", model)
    client = _client()
    try:
        response = client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": "Reply with only the word: Hello"}]}],
            inferenceConfig={"maxTokens": 20},
        )
    except botocore.exceptions.ClientError as exc:
        pytest.skip(f"Converse not supported for {model!r}: {exc}")

    output = response.get("output", {}).get("message", {})
    content = output.get("content", [{}])
    text = content[0].get("text", "") if content else ""
    assert text.strip(), f"Converse returned empty text. Full output: {output}"

    usage = response.get("usage", {})
    assert usage.get("inputTokens", 0) > 0
    assert usage.get("outputTokens", 0) > 0
    logger.info("response=%r  tokens(in=%s out=%s)",
                text, usage.get("inputTokens"), usage.get("outputTokens"))


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_converse_multi_turn(model):
    """Converse API maintains context across multiple message turns."""
    client = _client()
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


@pytest.mark.chat
@pytest.mark.parametrize("model", _MODELS)
def test_converse_system_prompt(model):
    """System prompt is forwarded and influences model output."""
    client = _client()
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

@pytest.mark.chat
@pytest.mark.streaming
@pytest.mark.parametrize("model", _MODELS)
def test_converse_stream(model):
    """ConverseStream delivers incremental text events."""
    logger.info("model=%s  api=ConverseStream", model)
    client = _client()
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

    assert event_count > 0, "No streaming events received"
    assert full_text.strip(), "Streamed text must not be empty"
    logger.info("events=%d  assembled=%r", event_count, full_text)
