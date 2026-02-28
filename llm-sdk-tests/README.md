# LLM SDK Test Suite

Integration tests for LLM providers and proxies configured in the **AI workspace**.
Each provider is tested through its native Python SDK (and LangChain where applicable),
pointed at your gateway URL instead of the real provider endpoint.

---

## Providers & SDKs tested

| Provider | Native SDK | LangChain |
|---|---|---|
| OpenAI | `openai` | `langchain-openai` |
| Anthropic | `anthropic` | `langchain-anthropic` |
| Google Gemini | `google-genai` | `langchain-google-genai` |
| Mistral | `mistralai` + `openai` | `langchain-openai` |
| Azure OpenAI | `openai` (AzureOpenAI) | `langchain-openai` |
| Azure AI Foundry | `openai` (AzureOpenAI) | `langchain-openai` |

Provider and proxy tests run independently — both are exercised for every
enabled entry in `config.yaml`.

---

## Quick start

### 1 — Clone / copy this directory

This test suite is self-contained. Copy the `llm-sdk-tests/` directory to any
machine that can reach your gateway and run the steps below.

### 2 — Set up the environment

```bash
./setup.sh
```

This creates a `venv/` virtual environment and installs all dependencies from `requirements.txt`.

### 3 — Configure your gateway details

```bash
cp config.yaml.example config.yaml
# edit config.yaml — fill in base_url, api_key, and models for each provider
```

Set `enabled: true` for the providers/proxies you want to test.
See the **Config reference** section below for field descriptions.

### 4 — Run the tests

```bash
./run.sh                          # all enabled providers and proxies
./run.sh -m openai                # OpenAI tests only
./run.sh -m anthropic             # Anthropic tests only
./run.sh -m gemini                # Gemini tests only
./run.sh -m mistral               # Mistral tests only
./run.sh -m azure_openai          # Azure OpenAI tests only
./run.sh -m azure_foundry         # Azure AI Foundry tests only
./run.sh -m proxy                 # all proxy tests
./run.sh -m langchain             # all LangChain SDK tests
./run.sh -m "chat and not proxy"  # chat tests for providers only
./run.sh -m streaming             # streaming tests only
```

Any extra arguments are forwarded to pytest:

```bash
./run.sh -m openai -v --tb=long   # verbose output with full tracebacks
./run.sh tests/providers/test_openai_provider.py  # a single file
```

### 5 — Use a custom config file

```bash
LLM_TEST_CONFIG=/path/to/other-config.yaml ./run.sh
```

---

## Config reference

```yaml
# Set to false for dev gateways with self-signed TLS certificates.
verify_ssl: true

providers:
  openai:
    enabled: true
    # Gateway URL + the context path set in the AI workspace.
    # Include /v1 at the end for OpenAI-compatible providers.
    base_url: "https://<gateway-host>/<context>/v1"
    api_key: "gw_..."     # key generated in the AI workspace
    models:
      - "gpt-4o-mini"     # model IDs configured in the provider

  anthropic:
    enabled: false
    base_url: "https://<gateway-host>/<context>"
    api_key: "gw_..."
    models:
      - "claude-sonnet-4-5"

  gemini:
    enabled: false
    base_url: "https://<gateway-host>/<context>"
    api_key: "gw_..."
    models:
      - "gemini-2.5-flash"

  mistral:
    enabled: false
    base_url: "https://<gateway-host>/<context>"
    api_key: "gw_..."
    models:
      - "mistral-small-latest"

  azure_openai:
    enabled: false
    base_url: "https://<gateway-host>/<context>"
    api_key: "gw_..."
    api_version: "2024-10-21"
    models:
      - "<azure-deployment-name>"

  azure_foundry:
    enabled: false
    base_url: "https://<gateway-host>/<context>"
    api_key: "gw_..."
    api_version: "2024-10-21"
    models:
      - "<azure-deployment-name>"

proxies:
  - name: "My OpenAI Proxy"
    enabled: false
    provider_type: "openai"   # controls which SDK is used
    base_url: "https://<gateway-host>/<context>/v1"
    api_key: "gw_..."
    models:
      - "gpt-4o-mini"
```

**`provider_type` values:** `openai`, `anthropic`, `gemini`, `mistral`,
`azure_openai`, `azure_foundry`

---

## Project layout

```
llm-sdk-tests/
├── setup.sh                  ← create venv and install dependencies
├── run.sh                    ← run the test suite
├── config.yaml.example       ← copy to config.yaml and fill in
├── config.yaml               ← your local config (git-ignored)
├── requirements.txt          ← pinned dependencies (all SDKs including LangChain)
├── pyproject.toml            ← pytest config and markers
├── conftest.py
├── utils/
│   └── config.py             ← config loading helpers
└── tests/
    ├── providers/            ← one file per provider × SDK
    └── proxies/              ← one file per provider_type × SDK
```

---

## Notes

- **`verify_ssl: false`** — set this when your gateway uses a self-signed TLS
  certificate (common in local/dev deployments). Never set it to false in
  production.

- **LangChain tests skip gracefully** — if a LangChain package is not
  installed, its tests are skipped automatically (no error).

- **Disabled providers/proxies** — tests for any entry with `enabled: false`
  are skipped, so you only pay for what you enable.
