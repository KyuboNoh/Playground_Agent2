from __future__ import annotations
import io
import numpy as np
import pandas as pd
from typing import Any
from pydantic import BaseModel
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

class AgentMsg(BaseModel):
    role: str
    name: str
    content: str
    payload: dict | None = None

class Agent:
    name = "agent"
    def __init__(self, logger: list[AgentMsg]):
        self.logger = logger
    def say(self, txt: str):
        self.logger.append(AgentMsg(role="agent", name=self.name, content=txt))

class Agent0Policy(Agent):
    name = "Agent0 (Policy/RAG)"
    def analyze(self, deposit_type: str) -> dict:
        self.say(f"Loaded deposit KB for {deposit_type} (stub)")
        return {"kb_bullets": [
            "Structures & alteration are key",
            "Use spatially aware CV",
        ]}

class Agent1DataQuery(Agent):
    name = "Agent1 (Data Query)"
    def load_features(self, data_bytes: bytes | None) -> tuple[pd.DataFrame, dict]:
        if data_bytes is None:
            n = 10000
            rng = np.random.RandomState(0)
            X = pd.DataFrame(rng.randn(n, 5), columns=[f"band_{i}" for i in range(5)])
            X["x"], X["y"] = np.meshgrid(np.arange(100), np.arange(100))
            X["x"], X["y"] = X["x"].values.reshape(-1), X["y"].values.reshape(-1)
            self.say("Generated synthetic 100x100 grid with 5 bands")
            return X, {"source": "synthetic"}
        df = pd.read_csv(io.BytesIO(data_bytes))
        self.say(f"Loaded data CSV: {df.shape}")
        return df, {"source": "csv"}

    def load_labels(self, labels_bytes: bytes | None) -> pd.DataFrame | None:
        if labels_bytes is None:
            self.say("No separate labels CSV provided")
            return None
        df = pd.read_csv(io.BytesIO(labels_bytes))
        self.say(f"Loaded labels CSV: {df.shape}")
        return df

    def load_occurrences(self) -> pd.DataFrame:
        pos = pd.DataFrame({
            "x": np.random.randint(0, 100, 40),
            "y": np.random.randint(0, 100, 40),
            "label": 1,
        })
        self.say("Generated 40 synthetic positives")
        return pos

class Agent3Model(Agent):
    name = "Agent3 (Model)"
    @staticmethod
    def pick_features(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
        if requested:
            miss = [c for c in requested if c not in df.columns]
            if miss:
                raise ValueError(f"Missing feature columns: {miss}")
            return requested
        cand = [c for c in df.columns if c not in {"x","y","label","rank","prospectivity"}]
        return [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]

    def train(self, train_df: pd.DataFrame, algo: str, features: list[str]) -> dict:
        X = train_df[features].values
        y = train_df["label"].values
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros(len(train_df))
        mets = []
        for tr, va in kf.split(X):
            if algo == "xgb" and HAS_XGB:
                model = XGBClassifier(n_estimators=200, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                      learning_rate=0.05, eval_metric="logloss", random_state=0, n_jobs=4)
            else:
                model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4)
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[va])[:,1]
            oof[va] = p
            try:
                mets.append((average_precision_score(y[va], p), roc_auc_score(y[va], p)))
            except Exception:
                mets.append((float("nan"), float("nan")))
        import numpy as np
        m_ap = float(np.nanmean([m[0] for m in mets]))
        m_auc = float(np.nanmean([m[1] for m in mets]))
        self.say(f"CV mAP={m_ap:.3f}, AUROC={m_auc:.3f}")
        if algo == "xgb" and HAS_XGB:
            final = XGBClassifier(n_estimators=400, max_depth=5, subsample=0.9, colsample_bytree=0.9,
                                  learning_rate=0.05, eval_metric="logloss", random_state=0, n_jobs=4)
        else:
            final = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=4)
        final.fit(X, y)
        return {"model": final, "features": features, "mAP": m_ap, "AUROC": m_auc}


# === Agent0 patched override (appended) ===
import os, json, requests
from pathlib import Path

def _pdf_bytes_to_text_agent0_override(name: str, b: bytes) -> str:
    nm = (name or "").lower()
    if not nm.endswith(".pdf"):
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    try:
        import fitz  # from PyMuPDF
        doc = fitz.open(stream=b, filetype="pdf")
        parts = []
        for pg in doc:
            try:
                parts.append(pg.get_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception:
        return ""

def _call_ollama_agent0_override(prompt: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").rstrip("/")
    model = os.getenv("LLM_AGENT0_MODEL", "llama3.1:8b-instruct-q4_K_M")
    temperature = float(os.getenv("LLM_AGENT0_TEMPERATURE", "0.2"))
    num_ctx = int(os.getenv("LLM_AGENT0_NUM_CTX", "4096"))
    r = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def _call_openai_agent0_override(prompt: str) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is not installed. pip install openai") from e
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_AGENT0_MODEL", "gpt-4o-mini")
    out = client.responses.create(model=model, input=prompt)
    return out.output_text

def _call_backend_agent0_override(prompt: str) -> str:
    backend = os.getenv("LLM_AGENT0_BACKEND", "ollama").lower()
    if backend == "openai":
        return _call_openai_agent0_override(prompt)
    return _call_ollama_agent0_override(prompt)

class Agent0PolicyPatched(Agent):
    name = "Agent0 (Policy/RAG, patched)"

    def _load_docs_from_config(self) -> list[tuple[str, bytes]]:
        pairs = []
        cfgp = Path("outputs/config.json")
        if not cfgp.exists():
            return pairs
        try:
            cfg = json.loads(cfgp.read_text(encoding="utf-8", errors="ignore"))
            docs = cfg.get("docs") or []
            for p in docs:
                if not p:
                    continue
                try:
                    with open(p, "rb") as fh:
                        pairs.append((os.path.basename(p), fh.read()))
                except Exception:
                    continue
        except Exception:
            return []
        return pairs

    def analyze(
        self,
        deposit_type: str,
        docs: list[tuple[str, bytes]] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict:
        enabled = str(os.getenv("ENABLE_AGENT0_LLM", "1")) != "0"
        if not enabled:
            self.say("Agent0: LLM disabled (ENABLE_AGENT0_LLM=0). Using fallback.")
            return {"kb_bullets": [
                "Structures & alteration are key signals for many deposits.",
                "Prefer spatially aware CV / block-CV to avoid leakage.",
            ]}

        doc_pairs = docs if docs else self._load_docs_from_config()
        texts = []
        for (nm, b) in (doc_pairs or []):
            t = _pdf_bytes_to_text_agent0_override(nm, b)
            if t:
                texts.append(f"### {nm}\n{t}")
        if not texts:
            self.say("Agent0: No documents provided (or unreadable). Proceeding with generic KB.")
            return {"kb_bullets": [
                "Use geoscience priors to define features (structure, lithology, alteration).",
                "Cross-validate with spatial blocks; calibrate thresholds to label scarcity.",
            ]}

        tmpl_path = Path("Agent0_Prompt.txt")
        if not tmpl_path.exists():
            self.say("Agent0: Agent0_Prompt.txt missing. Returning fallback KB.")
            return {"kb_bullets": [
                "Collect domain-specific priors from SMEs.",
                "Validate using withheld areas; avoid spatial leakage.",
            ]}
        template = tmpl_path.read_text(encoding="utf-8", errors="ignore")
        all_text = "\n\n".join(texts)
        all_text = all_text[:100_000]
        prompt = template.replace("{{DOC_TEXT}}", all_text)

        try:
            if overrides:
                raw = _call_backend_agent0_with_overrides(
                    prompt,
                    overrides.get("backend"),
                    overrides.get("model"),
                    overrides.get("api_key"),
                )
            else:
                raw = _call_backend_agent0_override(prompt)
            try:
                parsed = json.loads(raw)
                self.say("Agent0: Extracted structured KB from docs.")
                return {"extracted": parsed}
            except Exception:
                self.say("Agent0: Non-JSON output received; returning raw text.")
                return {"summary_text": raw.strip()[:8000]}
        except Exception as e:
            self.say(f"Agent0: LLM call failed: {e}")
            return {
                "kb_bullets": [
                    "Fallback KB: If LLM unavailable, proceed with standard feature set and RF baseline.",
                    "Tune min/max via IQR; consider robust scalers for heavy-tailed features.",
                ]
            }

# Rebind the public name to use the patched implementation
Agent0Policy = Agent0PolicyPatched

def _call_backend_agent0_with_overrides(prompt: str, backend: str, model: str, api_key: str | None = None) -> str:
    backend = (backend or "").lower()
    if backend == "openai":
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package is not installed. pip install openai") from e
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (and no API key override provided).")
        client = OpenAI(api_key=key)
        mdl = model or os.getenv("LLM_AGENT0_MODEL", "gpt-4o-mini")
        out = client.responses.create(model=mdl, input=prompt)
        return out.output_text
    # default to ollama
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").rstrip("/")
    mdl = model or os.getenv("LLM_AGENT0_MODEL", "llama3.1:8b-instruct-q4_K_M")
    temperature = float(os.getenv("LLM_AGENT0_TEMPERATURE", "0.2"))
    num_ctx = int(os.getenv("LLM_AGENT0_NUM_CTX", "4096"))
    import requests
    r = requests.post(url, json={
        "model": mdl,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")
