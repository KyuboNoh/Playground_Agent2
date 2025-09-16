# Changelog

## 2025-09-12
- **Agent0 wired end-to-end**: reads uploaded docs from `outputs/config.json` and extracts a structured KB using either a local LLM (Ollama) or an API model (OpenAI).
- **Backends**: New `.env.example.local` and `.env.example.api` for quick switching between local and API usage.
- **Requirements**: Replaced incorrect `fitz` with `PyMuPDF`; added `openai`.
- **Agents**: Appended a patched `Agent0Policy` that overrides the original at import time. No changes needed in `orchestrator.py` or `ui.py` to consume docs; Agent0 loads doc paths saved via **Save Config**.

## 2025-09-13
- **OpenAI**: Switched Agent0 PDF extraction to the `responses` API and removed explicit temperature settings to avoid unsupported parameter errors.
