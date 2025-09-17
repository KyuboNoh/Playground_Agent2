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

if not PPLX_API_KEY:
    raise RuntimeError(
        "PPLX_API_KEY is not configured. Set it in your environment or .env file before running."
    )

url = "https://api.perplexity.ai/chat/completions"
headers = {
    "Authorization": f"Bearer {PPLX_API_KEY}",
    "Content-Type": "application/json",
}
# --- CONFIG ---
user_prompt = (
    "Find and summarize recent research papers (2022–2025) on "
    "Mineral prospect/potential mapping/prediction/geological using machine learning/artificial intelligence. "
    "List the methods, key findings, and provide source links. "
    "Provide structured information with fields: Year, Month, Journal, Impact factor, Commodity, Methodology, "
    "Title, Author(s), Abstract, Availability of original data, URL.\n\n"
    "Return ONLY valid JSON (no prose, no markdown). The value must be a JSON array of objects "
    "with EXACT keys: Year, Month, Journal, Journal Impact factor, Commodity (multiple), "
    "Methodology, Title, Author, Abstract, Availability of original data, URL."
)

messages = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": user_prompt},
]

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


def _ensure_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    sval = str(val).strip()
    return [sval] if sval else []


def _normalize_text_for_match(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9+&\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} " if text else ""


def _normalize_commodity_rule(entry: dict, extra_terms: List[str] | None = None):
    if not isinstance(entry, dict):
        return None

    def _pick(*names):
        for name in names:
            if name in entry and entry[name] not in (None, ""):
                return entry[name]
        return ""

    major = str(_pick("Major_category", "major_category", "MajorCategory", "major", "Major") or "").strip()
    minor = str(_pick("Minor_category", "minor_category", "MinorCategory", "minor", "Minor") or "").strip()

    elements = _ensure_list(
        _pick(
            "Commodities_elements",
            "commodities_elements",
            "Commodity_elements",
            "commodity_elements",
            "elements",
            "Element",
        )
    )
    combined = _ensure_list(
        _pick(
            "Commodities_combined",
            "commodities_combined",
            "Commodity_combined",
            "commodity_combined",
            "combined",
            "Combination",
            "name",
        )
    )
    if not combined and elements:
        combined = elements[:]
    if not elements and combined:
        elements = combined[:]

    term_fields: List[str] = []
    for key in ("aliases", "alias", "keywords", "keyword", "terms", "term", "match_terms", "matches", "search", "tokens"):
        term_fields.extend(_ensure_list(entry.get(key)))
    if extra_terms:
        term_fields.extend(extra_terms)
    if not term_fields:
        term_fields.extend(elements)
        term_fields.extend(combined)
        if minor:
            term_fields.append(minor)

    normalized_terms = []
    for term in term_fields:
        norm = _normalize_text_for_match(term).strip()
        if norm and norm not in normalized_terms:
            normalized_terms.append(norm)

    regexes = []
    for key in ("regex", "Regex", "patterns", "match_regex"):
        for pattern in _ensure_list(entry.get(key)):
            try:
                regexes.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                continue

    if not normalized_terms and not regexes:
        return None

    return {
        "major": major,
        "minor": minor,
        "elements": elements,
        "combined": combined,
        "terms": normalized_terms,
        "regex": regexes,
    }


def _load_commodity_rules() -> List[dict]:
    path = _find_upwards("Rule1_OreDeposit.json")
    if not path:
        print("⚠️ Rule1_OreDeposit.json not found. Skipping commodity mapping.")
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}. Skipping commodity mapping.")
        return []

    rules: List[dict] = []

    def _add_rule(obj, extra_terms: List[str] | None = None):
        normalized = _normalize_commodity_rule(obj, extra_terms)
        if normalized:
            rules.append(normalized)

    if isinstance(raw, list):
        for obj in raw:
            _add_rule(obj)
    elif isinstance(raw, dict):
        found_list = False
        for key in ("rules", "commodities", "entries", "data", "items"):
            if isinstance(raw.get(key), list):
                found_list = True
                for obj in raw[key]:
                    _add_rule(obj)
        if not found_list:
            for key, val in raw.items():
                if isinstance(val, dict):
                    _add_rule(val, extra_terms=[key])
    else:
        print(f"⚠️ Unexpected structure in {path}. Skipping commodity mapping.")

    return rules


def _map_commodities(rows: list[dict]):
    rules = _load_commodity_rules()
    if not rules:
        return

    for row in rows:
        text_parts = []
        for key in (
            "Commodity (multiple)",
            "Commodity",
            "commodities",
            "Title",
            "title",
            "Abstract",
            "abstract",
            "abtract",
        ):
            val = row.get(key)
            if val:
                text_parts.append(str(val))
        raw_text = " ".join(text_parts)
        norm_text = _normalize_text_for_match(raw_text)
        if not norm_text:
            continue

        matches: List[dict] = []
        seen = set()
        for rule in rules:
            matched = False
            for regex in rule.get("regex", []):
                if regex.search(raw_text):
                    matched = True
                    break
            if not matched:
                for term in rule.get("terms", []):
                    if term and f" {term} " in norm_text:
                        matched = True
                        break
            if matched:
                key = (
                    rule.get("major", ""),
                    rule.get("minor", ""),
                    tuple(rule.get("elements", [])),
                    tuple(rule.get("combined", [])),
                )
                if key not in seen:
                    seen.add(key)
                    matches.append(rule)

        if matches:
            formatted = []
            for rule in matches:
                major = rule.get("major") or "Unknown"
                minor = rule.get("minor") or "Unknown"
                elements = rule.get("elements") or []
                combined = rule.get("combined") or []
                elem_str = " + ".join(elements) if elements else "Unknown"
                comb_str = " + ".join(combined) if combined else elem_str
                formatted.append(f"{major}/{minor}/{elem_str}/{comb_str}")
            row["Commodity (multiple)"] = "; ".join(formatted)


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
# Optional: Commodity mapping using Rule1_OreDeposit.json
_map_commodities(rows)

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