import os, re
import json
import time
import pathlib
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

# t

# --- CONFIG ---

load_dotenv()
PPLX_API_KEY = os.getenv("PPLX_API_KEY")

url = "https://api.perplexity.ai/chat/completions"
headers = {
    "Authorization": f"Bearer {PPLX_API_KEY}",
    "Content-Type": "application/json",
}
print(PPLX_API_KEY);exit()
# --- CONFIG ---
messages = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": (
        "Find and summarize recent research papers (2022–2025) on "
        "Mineral prospect/potential mapping/prediction/geological using machine learning/artificial intelligence. "
        "List the methods, key findings, and provide source links. "
        "Provide structured information with fields: Year, Month, Journal, Impact factor, Commodity, Methodology, "
        "Title, Author(s), Abstract, Availability of original data, URL."
    )},
]

# ----------------- FORCE JSON FROM THE MODEL (recommended) -----------------
# Tip: strengthen your prompt before the API call:
messages.append({
  "role": "user",
  "content": (
    "Return ONLY valid JSON (no prose, no markdown). The value must be a JSON array of objects "
    "with EXACT keys: Year, Month, Journal, Journal Impact factor, Commodity (multiple), "
    "Methodology, Title, Author, Abstract, Availability of original data, URL."
  )
})

data = {
    "model": "sonar-pro",
    "messages": messages,
    "temperature": 0.2,
    "top_p": 0.9,
    "return_images": False,
    "search_recency_filter": "month",
    "web_search": True,
}

# ----------------- PARSING HELPERS -----------------
COLS = [
    "Year", "Month", "Journal", "Journal Impact factor", "Commodity (multiple)",
    "Methodology (from given file with classification)", "Title", "Author",
    "abtract", "availability of original data", "url"
]

# ----------------- CONFIG FOR OUTPUT -----------------
OUT_CSV = Path("papers_summary.csv")
OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)

OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)
ts = time.strftime("%Y%m%d-%H%M%S")
raw_http_path = OUT_DIR / f"perplexity_http_{ts}.txt"

def _norm(s: str) -> str:
    s = ("" if s is None else str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return s

def _make_key(row) -> str:
    return " | ".join([_norm(row.get("Journal","")),
                       _norm(row.get("Author","")),
                       _norm(row.get("Title",""))])

def _extract_json_block(text: str):
    """Try to load JSON from the whole text or from a ```json ... ``` fenced block."""
    # 1) direct JSON
    try:
        obj = json.loads(text)
        return obj
    except Exception:
        pass
    # 2) fenced code block
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        block = m.group(1)
        return json.loads(block)
    return None

def _coerce_record(d: dict) -> dict:
    """Coerce model keys → our expected keys."""
    out = {}
    # Accept common variants and map them
    out["Year"]  = d.get("Year", d.get("year",""))
    out["Month"] = d.get("Month", d.get("month",""))
    out["Journal"] = d.get("Journal", d.get("journal",""))
    out["Journal Impact factor"] = d.get("Journal Impact factor", d.get("Impact factor", d.get("impact_factor","")))
    out["Commodity (multiple)"] = d.get("Commodity (multiple)", d.get("Commodity", d.get("commodities","")))
    # Methodology: accept either name; we’ll map later
    out["Methodology (from given file with classification)"] = d.get("Methodology (from given file with classification)", d.get("Methodology", d.get("method","")))
    out["Title"] = d.get("Title", d.get("title",""))
    out["Author"] = d.get("Author", d.get("Authors", d.get("authors","")))
    # NOTE: the spec uses 'abtract' (typo) → we map from 'Abstract'
    out["abtract"] = d.get("abtract", d.get("Abstract", d.get("abstract","")))
    out["availability of original data"] = d.get("availability of original data", d.get("Availability of original data", d.get("data_availability","")))
    out["url"] = d.get("url", d.get("URL", d.get("link","")))
    return out

def _find_upwards(filename: str, start: Path = None) -> Path | None:
    p = (start or Path(__file__).resolve().parent)
    while True:
        cand = p / filename
        if cand.exists():
            return cand
        if p.parent == p:  # reached filesystem root
            return None
        p = p.parent

def _map_methodologies(rows: list[dict]):
    path = _find_upwards("method_classification.json")
    if not path:
        print("⚠️ method_classification.json not found. Skipping methodology mapping.")
        return
    try:
        mapping = json.loads(path.read_text())
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}. Skipping methodology mapping.")
        return
    for r in rows:
        raw = r.get("Methodology (from given file with classification)", "")
        hits = set()
        low = str(raw).lower()
        for k, v in mapping.items():
            if k.lower() in low:
                hits.add(v)
        r["Methodology (from given file with classification)"] = "; ".join(sorted(hits)) if hits else raw


# ----------------- SAVE RAW + GET CONTENT -----------------
try:
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    # Always save raw HTTP body for inspection
    raw_http_path.write_text(resp.text)
    print("HTTP:", resp.status_code, resp.reason, "| elapsed:", resp.elapsed)
    print("💾 Saved raw HTTP body →", raw_http_path)

    # Will raise for 4xx/5xx
    resp.raise_for_status()

    # Try to parse JSON
    payload = resp.json()
    print("Top-level keys:", list(payload.keys()))

    # Safely extract model content (OpenAI-style schema)
    content = (
        payload.get("choices", [{}])[0]
               .get("message", {})
               .get("content", "")
    )
    if not content:
        # Some responses might use other fields — dump sample for debugging
        raise RuntimeError(
            "Empty 'content' in response. Inspect the raw HTTP file above. "
            f"Choices: {payload.get('choices')}"
        )

    # Save the LLM-rendered content separately (what your downstream parser expects)
    raw_path = OUT_DIR / f"perplexity_raw_{ts}.md"
    raw_path.write_text(content)
    print(f"💾 Saved model content → {raw_path}")

except requests.exceptions.RequestException as e:
    # Network / HTTP errors (timeouts, 401/403/429/5xx, etc.)
    print("HTTP error:", e)
    print("Raw body:", resp.text if 'resp' in locals() else "<no response>")
    raise
except ValueError as e:
    # JSON decode error
    print("JSON parse error:", e)
    print("Raw body saved at:", raw_http_path)
    raise

ts = time.strftime("%Y%m%d-%H%M%S")
raw_path = OUT_DIR / f"perplexity_raw_{ts}.md"
raw_path.write_text(content)
print(f"💾 Saved raw response → {raw_path}")

# ----------------- TRY TO PARSE -----------------

papers_json = _extract_json_block(content)
if papers_json is None:
    # Nothing parseable → stop early with a helpful message
    raise RuntimeError(
        f"❌ Could not parse JSON from model output. Inspect {raw_path}.\n"
        "Hint: Add a follow-up message asking the model to return STRICT JSON (no code fences)."
    )

if isinstance(papers_json, dict):
    # Sometimes model returns an object with a top-level key
    # Try common keys like 'results' or 'papers'
    for k in ("results", "papers", "items", "data"):
        if k in papers_json and isinstance(papers_json[k], list):
            papers_json = papers_json[k]
            break

if not isinstance(papers_json, list):
    raise RuntimeError("❌ Parsed JSON is not a list. Adjust the prompt to return an array of objects.")

# Coerce to our schema
rows = [_coerce_record(d) for d in papers_json]

# Optional: Methodology mapping
_map_methodologies(rows)

# Build DataFrame with expected columns
df_new = pd.DataFrame(rows)
for c in COLS:
    if c not in df_new.columns:
        df_new[c] = ""

# Drop clearly empty rows (no title & no URL)
df_new = df_new[~(df_new["Title"].astype(str).str.strip().eq("")) | ~(df_new["url"].astype(str).str.strip().eq(""))]

# Build dedup key on NEW data
df_new["_key"] = df_new.apply(_make_key, axis=1)

# ----------------- DEDUP + SAVE -----------------
if OUT_CSV.exists():
    df_old = pd.read_csv(OUT_CSV)
    for c in COLS:
        if c not in df_old.columns:
            df_old[c] = ""
    if "_key" not in df_old.columns:
        df_old["_key"] = df_old.apply(_make_key, axis=1)

    old_keys = set(df_old["_key"].tolist())
    df_to_add = df_new[~df_new["_key"].isin(old_keys)].copy()

    combined = pd.concat([df_old, df_to_add], ignore_index=True)
    combined = combined.drop_duplicates(subset=["_key"], keep="first")
    combined = combined[COLS + ["_key"]]
    combined.to_csv(OUT_CSV, index=False)

    print(f"✅ Appended {len(df_to_add)} new rows. Total now: {len(combined)} → {OUT_CSV}")
else:
    df_new = df_new.drop_duplicates(subset=["_key"], keep="first")
    df_new = df_new[COLS + ["_key"]]
    df_new.to_csv(OUT_CSV, index=False)
    print(f"✅ Created {OUT_CSV} with {len(df_new)} rows.")