from __future__ import annotations
import os

def has_ssl() -> bool:
    try:
        import ssl  # noqa
        return True
    except Exception:
        return False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def check_api_linkage(backend: str, api_vendor: str | None = None) -> tuple[bool, str | None]:
    """Return (is_wired, reason) after attempting a lightweight backend check."""
    backend = (backend or "").lower().strip()
    vendor = (api_vendor or "").lower().strip()

    try:
        if backend == "api":
            if vendor == "openai":
                try:
                    from openai import OpenAI
                except Exception as e:  # pragma: no cover - import error path
                    return False, "openai package is not installed"
                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    return False, "OPENAI_API_KEY is not set"
                try:  # lightweight request
                    OpenAI(api_key=key).models.list()
                    return True, None
                except Exception as e:
                    return False, str(e)
            if vendor == "geogpt":
                import requests  # type: ignore
                key = os.getenv("GEOGPT_API_KEY")
                if not key:
                    return False, "GEOGPT_API_KEY is not set"
                url = os.getenv("GEOGPT_URL", "https://api.geogpt.com").rstrip("/")
                try:
                    r = requests.get(f"{url}/ping", headers={"Authorization": f"Bearer {key}"}, timeout=5)
                    r.raise_for_status()
                    return True, None
                except Exception as e:
                    return False, str(e)
            return False, f"Unknown API vendor: {api_vendor}"
        if backend == "ollama":
            import requests  # type: ignore
            url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
            try:
                r = requests.get(f"{url}/api/tags", timeout=5)
                r.raise_for_status()
                return True, None
            except Exception as e:
                return False, str(e)
        return False, f"Unknown backend: {backend}"
    except Exception as e:  # pragma: no cover - unexpected errors
        return False, str(e)
