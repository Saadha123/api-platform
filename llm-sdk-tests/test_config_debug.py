from utils.config import load_config, enabled_proxies_by_type

cfg = load_config()
proxies = enabled_proxies_by_type(cfg, "azure_openai")

print(f"Found {len(proxies)} Azure OpenAI proxies:")
for proxy in proxies:
    print(f"\nProxy: {proxy.get('name')}")
    print(f"  base_url: {proxy.get('base_url')}")
    print(f"  api_key: {proxy.get('api_key')[:20]}...")
    print(f"  api_version: {proxy.get('api_version')}")
    print(f"  models: {proxy.get('models')}")
