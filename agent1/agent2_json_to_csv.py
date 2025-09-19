from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
import requests
from requests import RequestException

try:  # pragma: no cover - import guard for optional dependency
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "The 'openai' package is required to run agent1_openai. Install it via 'pip install openai'."
    ) from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
temperature = 0.1
INPUT_PAPER_FOLDER = Path(os.getenv("INPUT_PAPER_FOLDER", "input_papers"))
OUT_DIR = Path(os.getenv("AGENT1_OUTPUT_DIR", "out/agent1_openai"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = Path(os.getenv("AGENT1_SUMMARY_CSV", "papers_summary.csv"))
MODEL_NAME = os.getenv("AGENT1_OPENAI_MODEL", "gpt-4o-mini")
MAX_POINTS = int(os.getenv("AGENT1_MAX_KEY_POINTS", "5"))

# Multi-line instructions sent to the model alongside each PDF.

def _load_rule_json(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {path.name} not found. Proceeding without embedded rules.")
        return "[]"
    except json.JSONDecodeError as exc:
        print(f"⚠️ Could not parse {path.name}: {exc}. Proceeding without embedded rules.")
        return "[]"
    return json.dumps(data, ensure_ascii=False)


_repo_root = Path(__file__).resolve().parent.parent
rules1_str = _load_rule_json(_repo_root / "Rule1_OreDeposit.json")
rules2_str = _load_rule_json(_repo_root / "Rule2_method_classification.json")

# ---------------------------------------------------------------------------
# Prompt payloads used as inputs for the LLM
# (sourced from agent2_json_to_csv.py to keep behaviour aligned)
# ---------------------------------------------------------------------------
# system_msg = {
#     "role": "system",
#     "content": (
#         "You are an expert geoscientist and computational scientist.\n"
#         "TASK SCOPE:\n"
#         "• Read the attached PDF and extract structured metadata and content.\n"
#         "• Classify COMMODITY strictly using the provided Rule1 JSON (authoritative). If you cannot map the paper to a rule, leave the Commodity out.\n"
#         "• Classify METHODOLOGY strictly using the provided Rule2 JSON (authoritative). If you cannot map, leave empty.\n\n"
#         "COMMODITY RULES (Rule1_OreDeposit.json):\n"
#         "Return 'Commodity' as a JSON OBJECT copied from the rule with EXACT keys:\n"
#         "  major_category, minor_category, commodities_elements, commodities_combined\n"
#         "Also return 'Deposit type' equal to the rule's 'minor_category'.\n"
#         f"{rules1_str}\n\n"
#         "METHODOLOGY RULES (Rule2_method_classification.json):\n"
#         "Return 'Methodology' as a semicolon-separated list of canonical class labels from this mapping only (no free text):\n"
#         f"{rules2_str}\n\n"
#         "OUTPUT RULES:\n"
#         "• Return ONLY JSON that matches the provided schema (no prose/markdown/code fences). If nothing qualifies, return an empty object or empty arrays where appropriate.\n"
#         "• Never fabricate: if a field is unknown, set it to 'Unknown'.\n"
#         "• Prefer values explicitly present in the PDF. If the PDF lacks a field, leave it 'Unknown'.\n"
#     ),
# }
#
# user_msg = {
#     "role": "user",
#     "content": (
#         "You are a research assistant helping a mineral exploration team. Read the attached PDF.\n"
#         "Extract:\n"
#         "• Bibliographic metadata (Year, Month, Journal, Journal Impact factor, Title, Author, url)\n"
#         "• Abstract (concise)\n"
#         "• Methodology (use Rule2 classes only)\n"
#         "• Commodity (as a Rule1 object) and Deposit type (= Rule1 minor_category)\n"
#         "• Dataset fields (public availability, types (List all kinds af data and their units (e.g., magnetic field; nT), train/app relationship & scope)\n"
#         "• Key points (bullets)\n\n"
#         "For 'Training vs Application dataset relationship': return 'Same' if the same dataset is used for training and application (even if split into subsets), else 'Different'.\n"
#         "If 'Different', describe the transfer as 'Region A to Region B' in 'Training to application scope'.\n"
#         "Return ONLY JSON that matches the given schema."
#     ),
# }

# A Commodity OBJECT exactly like Rule1 entries
COMMODITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "major_category": {"type": "string"},
        "minor_category": {"type": "string"},
        "commodities_elements": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "commodities_combined": {"type": "array", "items": {"type": "string"}, "minItems": 1}
    },
    "required": ["major_category", "minor_category", "commodities_elements", "commodities_combined"]
}

# ---------------------------------------------------------------------------
# Summary CSV configuration (tabular output written after model parsing)
# ---------------------------------------------------------------------------
# -----------------------------
# Columns we want in the CSV
# -----------------------------
SUMMARY_COLUMNS = [
    "Year","Month","Journal","Journal Impact factor","Title", "Commodity","Deposit type","Methodology",
    "Title","Author","Public availability original dataset","Training/Application dataset type",
    "Training vs Application dataset relationship","Training to application scope","url",
]


# -----------------------------
# Small helpers
# -----------------------------
def _make_key(row: Dict[str, Any]) -> str:
    def _norm(s: Any) -> str:
        s = ("" if s is None else str(s)).strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s-]", "", s)
        return s
    # Stable de-duplication key
    return " | ".join([_norm(row.get("Journal","")), _norm(row.get("Author","")), _norm(row.get("Title",""))])


def get_first(d: Dict[str, Any] | None, *names: str, default=None):
    if not isinstance(d, dict):
        return default
    for name in names:
        if name in d and d[name] not in (None, ""):
            return d[name]
    return default


def commodity_obj_to_path(obj: dict | None) -> str:
    if not isinstance(obj, dict):
        return "Unknown"
    major = obj.get("major_category") or "Unknown"
    minor = obj.get("minor_category") or "Unknown"
    elems = obj.get("commodities_elements") or []
    combs = obj.get("commodities_combined") or []
    elem_str = " + ".join(map(str, elems)) if elems else "Unknown"
    comb_str = " + ".join(map(str, combs)) if combs else elem_str
    return f"{major}/{minor}/{elem_str}/{comb_str}"


def row_from_model_json(payload: Any) -> dict:
    """
    Be tolerant to the two shapes you have in your files:
      • {bibliographic_metadata: {...}, methodology: "...", commodity: {...}, dataset_fields: {...}}
      • {"Bibliographic metadata": {...}, "Methodology": "...", "Commodity": {...}, "Dataset fields": {...}}
    """
    row = {c: "Unknown" for c in SUMMARY_COLUMNS}

    # --- Bibliography block (both spellings) ---
    b = get_first(payload, "bibliographic_metadata", "Bibliographic metadata", default={}) or {}
    if isinstance(b, dict):
        if (y := get_first(b, "Year", "year")) is not None: row["Year"] = str(y)
        if (m := get_first(b, "Month", "month")): row["Month"] = str(m)
        if (j := get_first(b, "Journal", "journal", "Journal Name")): row["Journal"] = str(j)
        imp = get_first(b, "Journal Impact factor", "Journal Impact Factor", "impact factor")
        if imp is not None: row["Journal Impact factor"] = str(imp)
        if (t := get_first(b, "Title", "title", "Paper Title")): row["Title"] = str(t)
        a = get_first(b, "Author", "Authors", "author", "authors")
        if a is not None:
            row["Author"] = "; ".join(map(str, a)) if isinstance(a, list) else str(a)
        if (u := get_first(b, "url", "URL", "Link", "DOI", "doi", "URL of the paper")):
            row["url"] = str(u)

    # --- Methodology (either key) ---
    meth = get_first(payload, "methodology", "Methodology")
    if meth: row["Methodology"] = str(meth)

    # --- Commodity + deposit type (either keys) ---
    com = get_first(payload, "commodity", "Commodity")
    if isinstance(com, dict):
        dep = get_first(payload, "Deposit type", "deposit type")
        row["Deposit type"] = dep if isinstance(dep, str) and dep.strip() else (com.get("minor_category") or "Unknown")

        com_elm = com.get("commodities_elements")
        row["Commodity"] = com_elm if isinstance(com_elm, str) and com_elm.strip() else "Unknown"


    # --- Dataset fields (either key) ---
    ds = get_first(payload, "dataset_fields", "Dataset fields", default={}) or {}
    if isinstance(ds, dict):
        # public availability (two spellings)
        av = get_first(ds, "public_availability", "public availability")
        if av is not None:
            row["Public availability original dataset"] = "True" if av in (True, "Yes", "yes", "TRUE") else str(av)

        # types can be list[str] or list[dict]
        types = get_first(ds, "types", default=[])
        parts: List[str] = []
        if isinstance(types, list):
            for t in types:
                if isinstance(t, dict):
                    dt = (get_first(t, "data_type", "type", "data") or "").strip()
                    unit = (get_first(t, "unit", "units") or "").strip()
                    parts.append(f"{dt} ({unit})" if dt and unit else dt or unit)
                else:
                    parts.append(str(t))
        if parts:
            row["Training/Application dataset type"] = "; ".join([p for p in parts if p])

        rel = get_first(ds, "train_app_relationship", "train/app relationship")
        if rel: row["Training vs Application dataset relationship"] = str(rel)

        sc = get_first(ds, "scope")
        if sc: row["Training to application scope"] = str(sc)

    return row


def iter_json_payloads(json_file: Path) -> Iterable[Dict[str, Any]]:
    """Yield one or more dict payloads from the file (handles list-or-dict)."""
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def main():
    ap = argparse.ArgumentParser(description="Convert *_response.json files to papers_summary.csv")
    ap.add_argument("--json-dir", default="out/agent1_openai/raw", help="Directory containing *_response.json files")
    ap.add_argument("--summary-csv", default="papers_summary.csv", help="Output CSV path")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite CSV instead of appending/deduping")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise SystemExit(f"JSON directory not found: {json_dir}")

    json_files = sorted(json_dir.glob("*_response.json"))
    if not json_files:
        raise SystemExit(f"No *_response.json files found in {json_dir}")

    rows: List[Dict[str, Any]] = []
    for jf in json_files:
        for payload in iter_json_payloads(jf):
            row = row_from_model_json(payload)
            row["_key"] = _make_key(row)
            rows.append(row)

    if not rows:
        raise SystemExit("No usable JSON payloads found.")

    df_new = pd.DataFrame(rows, columns=SUMMARY_COLUMNS + ["_key"]).drop_duplicates(subset=["_key"], keep="first")

    out_csv = Path(args.summary_csv)
    if out_csv.exists() and not args.overwrite:
        old = pd.read_csv(out_csv)
        if "_key" not in old.columns:
            # Build keys for an older CSV without a key column
            def _mk(r): return _make_key({"Journal": r.get("Journal",""),
                                          "Author": r.get("Author",""),
                                          "Title": r.get("Title","")})
            old["_key"] = old.apply(_mk, axis=1)
        combined = pd.concat([old, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["_key"], keep="last")
        combined = combined[SUMMARY_COLUMNS + ["_key"]]
        combined.to_csv(out_csv, index=False)
        print(f"✅ Updated {out_csv} (rows: {len(combined)})")
    else:
        df_new.to_csv(out_csv, index=False)
        print(f"✅ Wrote {out_csv} (rows: {len(df_new)})")


if __name__ == "__main__":
    main()