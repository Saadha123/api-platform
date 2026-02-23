"""
Configuration loader for the LLM SDK test suite.

Reads config.yaml (or the file pointed to by LLM_TEST_CONFIG env var) and
exposes typed helpers used by every test module.
"""

import os
import yaml
from typing import Any, Dict, List, Optional

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config() -> Dict[str, Any]:
    """Load and return the raw config dict (empty dict if file not found or empty)."""
    path = os.environ.get("LLM_TEST_CONFIG", _DEFAULT_CONFIG_PATH)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    # yaml.safe_load returns None for empty files; ensure we always return a dict
    return data if isinstance(data, dict) else {}


def _as_dict(value: Any) -> Dict[str, Any]:
    """Return *value* as a dict, or an empty dict if it is None / not a dict.

    dict.get(key, default) only falls back to *default* when the key is absent.
    In YAML, a key with no value (``key:``) parses to None — not absent — so
    callers must guard with this helper instead of relying on the default arg.
    """
    return value if isinstance(value, dict) else {}


def get_verify_ssl(cfg: Dict[str, Any]) -> bool:
    return bool(_as_dict(cfg).get("verify_ssl", True))


# ── Provider helpers ──────────────────────────────────────────────────────────

def get_provider(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return the provider sub-dict for *name*, or an empty dict."""
    providers = _as_dict(_as_dict(cfg).get("providers"))
    return _as_dict(providers.get(name))


def is_provider_enabled(cfg: Dict[str, Any], name: str) -> bool:
    return bool(get_provider(cfg, name).get("enabled", False))


def provider_base_url(cfg: Dict[str, Any], name: str) -> str:
    return get_provider(cfg, name).get("base_url", "")


def provider_api_key(cfg: Dict[str, Any], name: str) -> str:
    return get_provider(cfg, name).get("api_key", "")


def provider_models(cfg: Dict[str, Any], name: str) -> List[str]:
    models = get_provider(cfg, name).get("models")
    return models if isinstance(models, list) else []


# ── Proxy helpers ─────────────────────────────────────────────────────────────

def get_proxies(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all proxy entries that have enabled=true."""
    proxies = _as_dict(cfg).get("proxies") or []
    return [p for p in proxies if isinstance(p, dict) and p.get("enabled", False)]


def enabled_proxies_by_type(cfg: Dict[str, Any], provider_type: str) -> List[Dict[str, Any]]:
    return [p for p in get_proxies(cfg) if p.get("provider_type") == provider_type]
