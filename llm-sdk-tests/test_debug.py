import httpx
from openai import AzureOpenAI

# Test with proxy config
api_key = "2ef1ed8c27749caffa174e89c67d816b926296095192de9efbe168c78be2382f"
base_url = "https://localhost:8443/azure-openai-proxy"

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
print(f"  default_headers: {client.default_headers}")

try:
    response = client.chat.completions.create(
        model="apim-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )
    print(f"SUCCESS: {response.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {e}")
