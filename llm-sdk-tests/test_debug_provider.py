import httpx
from openai import AzureOpenAI

# Test with provider config (this works)
api_key = "b91ec50a3abc3cf8305a01cdbd2074072e92f92ed490b43ce931dcb4d2b3e092"
base_url = "https://localhost:8443/azure-openai"

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=base_url,
    api_version="2024-06-01",
    default_headers={"X-API-Key": api_key},
    http_client=httpx.Client(verify=False),
)

print(f"Client created with:")
print(f"  api_key: {api_key[:20]}...")
print(f"  azure_endpoint: {base_url}")

try:
    response = client.chat.completions.create(
        model="apim-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )
    print(f"SUCCESS: {response.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {e}")
