# LLM SDK Test Suite

Integration tests for LLM providers and proxies created in the **AI workspace**.
Each service provider is tested through its own native Python SDK, pointed at the
gateway URL instead of the real provider endpoint.

---

## Providers & SDKs

| Provider | SDK | Chat | Streaming | Embeddings |
|---|---|:---:|:---:|:---:|
| OpenAI | `openai` | ✓ | ✓ | ✓ |
| Azure OpenAI | `openai` (AzureOpenAI) | ✓ | ✓ | ✓ |
| Anthropic | `anthropic` | ✓ | ✓ | – |
| Google Gemini | `google-genai` | ✓ | ✓ | ✓ |
| Mistral | `mistralai` | ✓ | ✓ | ✓ |
| AWS Bedrock | `boto3` | ✓ (Converse) | ✓ (ConverseStream) | – |
| Meta (Llama) | `openai` (OpenAI-compatible) | ✓ | ✓ | – |

Proxy tests run for every proxy listed in `config.yaml` and pick the right
SDK automatically based on the proxy's `provider_type` field.

---

## Quick start

### 1 – Prerequisites

```bash
python -m pip install -r requirements.txt
```

### 2 – Create your config file

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` and fill in the details for every provider/proxy you want
to test.  Set `enabled: true` for the ones you want to run.

**Where to get the values from:**

| Field | Where to find it |
|---|---|
| `base_url` | Your gateway host + the context path you set when creating the provider/proxy in the AI workspace. For OpenAI-compatible providers add `/v1` at the end. |
| `api_key` | The key generated in the AI workspace after clicking "Generate API Key" for that provider/proxy. |
| `models` | The model IDs you configured in the provider/proxy (e.g. `gpt-4o-mini`). |

### 3 – Run all tests

```bash
cd api-platform/llm-sdk-tests
pytest
```

### 4 – Run a specific provider

```bash
pytest -m openai          # OpenAI provider tests only
pytest -m anthropic       # Anthropic tests only
pytest -m gemini          # Gemini tests only
pytest -m mistral         # Mistral tests only
pytest -m bedrock         # AWS Bedrock tests only
pytest -m meta            # Meta/Llama tests only
pytest -m azure_openai    # Azure OpenAI tests only
pytest -m proxy           # All proxy tests
```

### 5 – Run only certain resource types

```bash
pytest -m chat            # Chat/message completion tests
pytest -m streaming       # Streaming tests
pytest -m embeddings      # Embedding tests
```

### 6 – Use a different config file

```bash
LLM_TEST_CONFIG=/path/to/my-config.yaml pytest
```

---

## Config reference

```yaml
# Skip TLS certificate verification for dev gateways with self-signed certs.
verify_ssl: true          # set to false for local/dev environments

providers:
  openai:
    enabled: true
    # Full base URL for the OpenAI SDK — include /v1
    base_url: "https://<gateway-host>/<provider-context>/v1"
    api_key: "gw_..."       # gateway-generated API key
    models:
      - "gpt-4o-mini"

  azure_openai:
    enabled: false
    base_url: "https://<gateway-host>/<provider-context>"
    api_key: "gw_..."
    deployment: "gpt-4"         # Azure deployment name
    api_version: "2024-05-01-preview"
    models:
      - "gpt-4"

  anthropic:
    enabled: false
    base_url: "https://<gateway-host>/<provider-context>"
    api_key: "gw_..."
    models:
      - "claude-3-5-sonnet-20241022"

  gemini:
    enabled: false
    base_url: "https://<gateway-host>/<provider-context>"
    api_key: "gw_..."
    models:
      - "gemini-1.5-flash"

  mistral:
    enabled: false
    base_url: "https://<gateway-host>/<provider-context>"
    api_key: "gw_..."
    models:
      - "mistral-small-latest"

  bedrock:
    enabled: false
    base_url: "https://<gateway-host>/<provider-context>"
    api_key: "gw_..."     # gateway API key (injected as X-API-Key)
    region: "us-east-1"
    models:
      - "amazon.titan-text-premier-v1:0"

  meta:
    enabled: false
    # Gateway exposes Meta/Llama through an OpenAI-compatible endpoint
    base_url: "https://<gateway-host>/<provider-context>/v1"
    api_key: "gw_..."
    models:
      - "us.meta.llama3-3-70b-instruct-v1:0"

proxies:
  - name: "My OpenAI Proxy"
    enabled: false
    proxy_id: "my-proxy-id"
    provider_type: "openai"   # openai | azure_openai | anthropic | gemini | mistral | bedrock | meta
    base_url: "https://<gateway-host>/<proxy-context>/v1"
    api_key: "gw_..."
    models:
      - "gpt-4o-mini"
```

---

## AWS Bedrock auth notes

The gateway manages the real AWS credentials on the upstream side.
Client-side (test side) auth uses the gateway API key only.

boto3 is configured with `UNSIGNED` (no AWS SigV4 signing) and a
`before-send` event handler that injects `X-API-Key: <gateway-api-key>`
into every request.  No real AWS credentials are required in the test
environment.

---

## Project layout

```
llm-sdk-tests/
├── config.yaml.example   ← copy to config.yaml and fill in
├── requirements.txt
├── pyproject.toml        ← pytest config and markers
├── conftest.py           ← shared session fixtures
├── utils/
│   ├── __init__.py
│   └── config.py         ← config loading helpers
└── tests/
    ├── providers/
    │   ├── test_openai_provider.py
    │   ├── test_azure_openai_provider.py
    │   ├── test_anthropic_provider.py
    │   ├── test_gemini_provider.py
    │   ├── test_mistral_provider.py
    │   ├── test_bedrock_provider.py
    │   └── test_meta_provider.py
    └── proxies/
        └── test_proxies.py   ← parameterised over all proxies in config.yaml
```
