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
system_msg = {
    "role": "system",
    "content": (
        "You are an expert geoscientist and computational scientist.\n"
        "TASK SCOPE:\n"
        "• Read the attached PDF and extract structured metadata and content.\n"
        "• Classify COMMODITY strictly using the provided Rule1 JSON (authoritative). If you cannot map the paper to a rule, leave the Commodity out.\n"
        "• Classify METHODOLOGY strictly using the provided Rule2 JSON (authoritative). If you cannot map, leave empty.\n\n"
        "COMMODITY RULES (Rule1_OreDeposit.json):\n"
        "Return 'Commodity' as a JSON OBJECT copied from the rule with EXACT keys:\n"
        "  major_category, minor_category, commodities_elements, commodities_combined\n"
        "Also return 'Deposit type' equal to the rule's 'minor_category'.\n"
        f"{rules1_str}\n\n"
        "METHODOLOGY RULES (Rule2_method_classification.json):\n"
        "Return 'Methodology' as a semicolon-separated list of canonical class labels from this mapping only (no free text):\n"
        f"{rules2_str}\n\n"
        "OUTPUT RULES:\n"
        "• Return ONLY JSON that matches the provided schema (no prose/markdown/code fences). If nothing qualifies, return an empty object or empty arrays where appropriate.\n"
        "• Never fabricate: if a field is unknown, set it to 'Unknown'.\n"
        "• Prefer values explicitly present in the PDF. If the PDF lacks a field, leave it 'Unknown'.\n"
    ),
}

user_msg = {
    "role": "user",
    "content": (
        "You are a research assistant helping a mineral exploration team. Read the attached PDF.\n"
        "Extract:\n"
        "• Bibliographic metadata (Year, Month, Journal, Journal Impact factor, Title, Author, url)\n"
        "• Abstract (concise)\n"
        "• Methodology (use Rule2 classes only)\n"
        "• Commodity (as a Rule1 object) and Deposit type (= Rule1 minor_category)\n"
        "• Dataset fields (public availability, types (List all kinds af data and their units (e.g., magnetic field; nT), train/app relationship & scope)\n"
        "• Key points (bullets)\n\n"
        "For 'Training vs Application dataset relationship': return 'Same' if the same dataset is used for training and application (even if split into subsets), else 'Different'.\n"
        "If 'Different', describe the transfer as 'Region A to Region B' in 'Training to application scope'.\n"
        "Return ONLY JSON that matches the given schema."
    ),
}

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
SUMMARY_COLUMNS = [
    "Year","Month","Journal","Journal Impact factor","Commodity","Deposit type","Methodology",
    "Title","Author","Public availability original dataset","Training/Application dataset type",
    "Training vs Application dataset relationship","Training to application scope","url",
]

SUMMARY_KEYS_CANON = {
    "Year","Month","Journal","Journal Impact factor","Title","Author","abstract","url",
    "Commodity","Deposit type","Methodology",
    "Public availability original dataset","Training/Application dataset type",
    "Training vs Application dataset relationship","Training to application scope"
}
_SUMMARY_KEYS_LOWER = {s.lower() for s in SUMMARY_KEYS_CANON}

def _iter_dicts(obj: Any):
    """Yield every dict inside obj (depth-first)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_dicts(it)

def _is_summary_like(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    keys = {k.lower() for k in d.keys() if isinstance(k, str)}
    # consider a dict "summary-like" if it has at least 2 canonical summary keys
    return len(keys & _SUMMARY_KEYS_LOWER) >= 2

def _merge_left(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ∪ b with b overwriting a (keep strings/arrays/objects as-is)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = v
    return out

def _extract_candidates(payload: Any) -> List[Dict[str, Any]]:
    """Return a list of candidate items with unified 'summary' dicts and meta arrays."""
    # If payload is already a list of items, process each entry independently
    roots = payload if isinstance(payload, list) else [payload]
    items: List[Dict[str, Any]] = []

    for root in roots:
        if not isinstance(root, (dict, list)):
            continue

        # 1) Prefer an explicit 'summary' dict if present at the root
        if isinstance(root, dict) and isinstance(root.get("summary"), dict):
            base = {
                "summary": root["summary"],
                "key_points": root.get("key_points") or [],
                "methodology_terms": root.get("methodology_terms") or [],
                "commodity_terms": root.get("commodity_terms") or [],
            }
            items.append(base)

        # 2) Otherwise, search deeply for the *most complete* summary-like dict(s)
        #    We pick the dict(s) that have the most canonical keys.
        best_dicts: List[Dict[str, Any]] = []
        best_score = 0
        for d in _iter_dicts(root):
            if not isinstance(d, dict):
                continue
            if "summary" in d and isinstance(d["summary"], dict):
                cand = d["summary"]
                score = len({k.lower() for k in cand.keys()} & _SUMMARY_KEYS_LOWER)
                if score > best_score:
                    best_score, best_dicts = score, [cand]
                elif score == best_score and score > 0:
                    best_dicts.append(cand)
            elif _is_summary_like(d):
                score = len({k.lower() for k in d.keys()} & _SUMMARY_KEYS_LOWER)
                if score > best_score:
                    best_score, best_dicts = score, [d]
                elif score == best_score and score > 0:
                    best_dicts.append(d)

        if best_dicts:
            # Merge all best candidates (last one wins), so scattered fields combine
            merged = {}
            for d in best_dicts:
                merged = _merge_left(merged, d)
            items.append({
                "summary": merged,
                "key_points": [],
                "methodology_terms": [],
                "commodity_terms": [],
            })

    # De-duplicate identical summaries by a stable fingerprint
    uniq, seen = [], set()
    for it in items:
        s = it.get("summary") or {}
        fp = tuple(sorted((k.lower(), str(v)) for k, v in s.items() if isinstance(k, str)))
        if fp not in seen and s:
            seen.add(fp)
            uniq.append(it)
    return uniq

def _normalise_model_payload(payload: Any) -> List[Dict[str, Any]]:
    """Wrapper that uses the deep extractor and guarantees the contract."""
    candidates = _extract_candidates(payload)
    # Always return at least one item with empty summary if nothing matched
    return candidates or [{"summary": {}, "key_points": [], "methodology_terms": [], "commodity_terms": []}]

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

def row_from_model_json(payload: dict) -> dict[str, str]:
    row = {c: "Unknown" for c in SUMMARY_COLUMNS}

    # 1) Bibliographic block
    b = payload.get("bibliographic_metadata") or {}
    if isinstance(b, dict):
        if b.get("Year") is not None: row["Year"] = str(b.get("Year"))
        if b.get("Month"): row["Month"] = str(b.get("Month"))
        if b.get("Journal"): row["Journal"] = str(b.get("Journal"))
        if b.get("Journal Impact factor") is not None:
            row["Journal Impact factor"] = str(b.get("Journal Impact factor"))
        if b.get("Title"): row["Title"] = str(b.get("Title"))
        a = b.get("Author")
        if a is not None:
            row["Author"] = "; ".join(map(str, a)) if isinstance(a, list) else str(a)
        if b.get("url"): row["url"] = str(b.get("url"))

    # 2) Methodology (flat string)
    if payload.get("methodology"):
        row["Methodology"] = str(payload["methodology"])

    # 3) Commodity object -> path; Deposit type from explicit field or minor_category
    com = payload.get("commodity")
    if isinstance(com, dict):
        row["Commodity"] = commodity_obj_to_path(com)
        dep = payload.get("Deposit type")
        row["Deposit type"] = dep.strip() if isinstance(dep, str) and dep.strip() else (com.get("minor_category") or "Unknown")

    # 4) Dataset fields
    ds = payload.get("dataset_fields") or {}
    if isinstance(ds, dict):
        if ds.get("public_availability") is not None:
            val = ds["public_availability"]
            row["Public availability original dataset"] = "True" if val in (True, "Yes", "yes", "TRUE") else str(val)
        types = ds.get("types") or []
        parts = []
        if isinstance(types, list):
            for t in types:
                if isinstance(t, dict):
                    dt = (t.get("data_type") or "").strip()
                    unit = (t.get("unit") or "").strip()
                    parts.append(f"{dt} ({unit})" if dt and unit else dt or unit)
                else:
                    parts.append(str(t))
        if parts:
            row["Training/Application dataset type"] = "; ".join(filter(None, parts))
        if ds.get("train_app_relationship"):
            row["Training vs Application dataset relationship"] = str(ds["train_app_relationship"])
        if ds.get("scope"):
            row["Training to application scope"] = str(ds["scope"])

    return row

# Alternate field names occasionally produced by the LLM. These aliases
# improve our chances of capturing key metadata even when the response
# deviates from the requested casing or wording. The requirement to
# "strongly" look for the journal impact factor is handled by mapping
# multiple possible spellings of that field.
SUMMARY_FIELD_ALIASES: Dict[str, List[str]] = {
    "Journal Impact factor": [
        "Journal Impact Factor",
        "journal impact factor",
        "Journal impact factor",
        "Impact Factor",
        "impact factor",
        "impact_factor",
        "journal impact",
        "journal_impact_factor",
    ],
    "Public availability original dataset": [
        "availability of original data",
        "Availability of original data",
        "Public availability of original dataset",
        "public availability original data",
        "availability_of_original_data",
    ],
    "Training/Application dataset type": [
        "Training dataset type",
        "Application dataset type",
        "Dataset type",
        "Training application dataset type",
    ],
    "Training vs Application dataset relationship": [
        "Training vs application dataset",
        "Training vs Application dataset",
        "Training application dataset match",
        "Training vs application relationship",
        "Dataset relationship",
    ],
    "Training to application scope": [
        "Training to application region",
        "Training to application geography",
        "Dataset transfer scope",
        "Training application scope",
    ],
}

SUMMARY_FIELD_ALIASES.update({
    "Year": ["Publication Year", "Published Year"],
    "Month": ["Publication Month"],
    "Journal": ["Journal Name"],
    "Title": ["Paper Title"],
    "Author": ["Authors","Author(s)"],
    "url": ["URL","Url","Link","DOI","doi","URL of the paper","URL of the papers"],
})

SUMMARY_FIELD_ALIAS_SOURCES_ENV = "SUMMARY_FIELD_ALIAS_URLS"
SUMMARY_FIELD_ALIAS_CACHE = OUT_DIR / "summary_field_aliases_cache.json"


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"⚠️ Invalid integer for {name}: {raw!r}. Using default {default}.")
        return default


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ Invalid float for {name}: {raw!r}. Using default {default}.")
        return default


SUMMARY_FIELD_ALIAS_CACHE_TTL = _get_int_env("SUMMARY_FIELD_ALIAS_CACHE_TTL", 24 * 60 * 60)
SUMMARY_FIELD_ALIAS_TIMEOUT = _get_float_env("SUMMARY_FIELD_ALIAS_TIMEOUT", 10.0)


def _parse_alias_source_urls(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[\s,]+", raw)
    return [token for token in (part.strip() for part in parts) if token]


def _load_alias_cache(path: Path, sources: List[str], max_age: int) -> Dict[str, List[str]] | None:
    if not path.exists():
        return None
    if max_age > 0:
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            age = max_age + 1
        if age > max_age:
            return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cached_sources = payload.get("sources")
    aliases = payload.get("aliases")
    if cached_sources != sources or not isinstance(aliases, dict):
        return None

    cleaned: Dict[str, List[str]] = {}
    for canonical, values in aliases.items():
        if not isinstance(canonical, str):
            continue
        if isinstance(values, str):
            candidates = [values]
        elif isinstance(values, list):
            candidates = [alias for alias in values if isinstance(alias, str)]
        else:
            continue
        unique: List[str] = []
        seen: set[str] = set()
        for alias in candidates:
            alias = alias.strip()
            if not alias:
                continue
            lower = alias.lower()
            if lower in seen:
                continue
            seen.add(lower)
            unique.append(alias)
        if unique:
            cleaned[canonical] = unique
    return cleaned


def _write_alias_cache(path: Path, sources: List[str], data: Dict[str, List[str]]) -> None:
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"sources": sources, "aliases": data}, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _normalise_alias_payload(payload: Any) -> Dict[str, List[str]]:
    if not isinstance(payload, dict):
        return {}
    normalised: Dict[str, List[str]] = {}
    for canonical, values in payload.items():
        if not isinstance(canonical, str):
            continue
        if isinstance(values, str):
            candidates = [values]
        elif isinstance(values, (list, tuple, set)):
            candidates = [alias for alias in values if isinstance(alias, str)]
        else:
            continue
        unique: List[str] = []
        seen: set[str] = set()
        for alias in candidates:
            alias = alias.strip()
            if not alias:
                continue
            lower = alias.lower()
            if lower in seen:
                continue
            seen.add(lower)
            unique.append(alias)
        if unique:
            normalised[canonical] = unique
    return normalised


def _read_alias_source(source: str) -> Dict[str, List[str]]:
    parsed = urlparse(source)
    try:
        if parsed.scheme in {"http", "https"}:
            response = requests.get(source, timeout=SUMMARY_FIELD_ALIAS_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        elif parsed.scheme == "file":
            payload_path = Path(parsed.path)
            with payload_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        elif parsed.scheme:
            print(f"⚠️ Unsupported alias source scheme '{parsed.scheme}' in {source}")
            return {}
        else:
            payload_path = Path(source)
            with payload_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
    except RequestException as exc:
        print(f"⚠️ Could not fetch alias mapping from {source}: {exc}")
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠️ Could not read alias mapping from {source}: {exc}")
        return {}
    return _normalise_alias_payload(payload)


def _merge_alias_maps(target: Dict[str, List[str]], additions: Dict[str, List[str]]) -> None:
    for canonical, aliases in additions.items():
        if not isinstance(canonical, str):
            continue
        canonical_key = canonical.strip()
        if not canonical_key:
            continue
        existing = target.setdefault(canonical_key, [])
        seen = {alias.lower(): alias for alias in existing if isinstance(alias, str)}
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            alias_clean = alias.strip()
            if not alias_clean:
                continue
            lower = alias_clean.lower()
            if lower in seen:
                continue
            existing.append(alias_clean)
            seen[lower] = alias_clean


def _load_remote_aliases() -> Dict[str, List[str]]:
    sources = _parse_alias_source_urls(os.getenv(SUMMARY_FIELD_ALIAS_SOURCES_ENV, ""))
    if not sources:
        return {}

    cached = _load_alias_cache(SUMMARY_FIELD_ALIAS_CACHE, sources, SUMMARY_FIELD_ALIAS_CACHE_TTL)
    if cached is not None:
        return cached

    merged: Dict[str, List[str]] = {}
    for source in sources:
        remote_aliases = _read_alias_source(source)
        if not remote_aliases:
            continue
        _merge_alias_maps(merged, remote_aliases)

    if merged:
        _write_alias_cache(SUMMARY_FIELD_ALIAS_CACHE, sources, merged)

    return merged


SUMMARY_FIELD_ALIASES_REMOTE = _load_remote_aliases()
if SUMMARY_FIELD_ALIASES_REMOTE:
    _merge_alias_maps(SUMMARY_FIELD_ALIASES, SUMMARY_FIELD_ALIASES_REMOTE)
    total_remote_aliases = sum(len(values) for values in SUMMARY_FIELD_ALIASES_REMOTE.values())
    print(
        "ℹ️ Loaded"
        f" {total_remote_aliases} remote summary field aliases from"
        f" {len(SUMMARY_FIELD_ALIASES_REMOTE)} internet source entries."
    )


def _iter_summary_aliases(column: str) -> Iterable[str]:
    aliases = SUMMARY_FIELD_ALIASES.get(column, [])
    seen: set[str] = set()
    for alias in aliases:
        if not isinstance(alias, str):
            continue
        alias_clean = alias.strip()
        if not alias_clean:
            continue
        lower = alias_clean.lower()
        if lower in seen:
            continue
        seen.add(lower)
        yield alias_clean

# ---------------------------------------------------------------------------
# Key findings CSV configuration (detail rows derived from model output)
# ---------------------------------------------------------------------------
DETAIL_COLUMNS = [
    "Title",
    "filename",
    "order",
    "key_point",
    "page_reference",
    "evidence",
    "confidence",
    "_parent_key",
]

# ---------------------------------------------------------------------------
# JSON schema describing the structured response we expect from the LLM
# (intermediate payload saved to raw JSON files before CSV generation)
# ---------------------------------------------------------------------------
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "paper_review",
        "strict": True,  # << enforce schema
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "Year": {"type": "string"},
                        "Month": {"type": "string"},
                        "Journal": {"type": "string"},
                        "Journal Impact factor": {"type": "string"},
                        "Title": {"type": "string"},
                        "Author": {"type": "string"},
                        "abstract": {"type": "string"},
                        "url": {"type": "string"},
                        "Commodity": COMMODITY_SCHEMA,
                        "Deposit type": {"type": "string"},  # << was object; must be string = minor_category
                        "Methodology": {"type": "string"},    # semicolon-separated canonical labels
                        "Public availability original dataset": {"type": ["string","boolean"]},  # allow 'Unknown'
                        "Training/Application dataset type": {"type": "string"},
                        "Training vs Application dataset relationship": {"type": "string"},
                        "Training to application scope": {"type": "string"}
                    },
                    # Require the stuff you actually need in the CSV.
                    # The model will use 'Unknown' if the PDF lacks it.
                    "required": [
                        "Title", "Author", "Year", "Journal", "Journal Impact factor",
                        "Methodology", "Commodity", "Deposit type", "url"
                    ]
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "summary": {"type": "string"},
                            "page_reference": {"type": "string"},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "string"}
                        },
                        "required": ["summary"]
                    }
                },
                "methodology_terms": {"type": "array", "items": {"type": "string"}},
                "commodity_terms": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["summary"]
        }
    }
}

# ---------------------------------------------------------------------------
# Helper utilities shared with the Perplexity workflow (duplicated here to
# avoid importing agent0 which requires credentials at import time).
# ---------------------------------------------------------------------------

def _ensure_list(val: Any) -> List[str]:
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


def _find_upwards(filename: str, start: Path | None = None) -> Path | None:
    p = (start or Path(__file__).resolve().parent)
    while True:
        cand = p / filename
        if cand.exists():
            return cand
        if p.parent == p:
            return None
        p = p.parent


def _normalize_commodity_rule(entry: Dict[str, Any], extra_terms: List[str] | None = None) -> Dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    major = entry.get("Major_category") or entry.get("major")
    minor = entry.get("Minor_category") or entry.get("minor")
    elements = _ensure_list(entry.get("Commodities_elements") or entry.get("elements"))
    combined = _ensure_list(entry.get("Commodities_combined") or entry.get("combined"))

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
    except Exception as exc:
        print(f"⚠️ Could not read {path}: {exc}. Skipping commodity mapping.")
        return []

    rules: List[dict] = []

    def _add_rule(obj: Dict[str, Any], extra_terms: List[str] | None = None):
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
    return rules


def _map_commodities(rows: List[Dict[str, Any]]):
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
        ):
            val = row.get(key)
            if val:
                text_parts.append(str(val))
        if row.get("_commodity_terms"):
            text_parts.extend(row["_commodity_terms"])
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


def _load_method_mapping() -> Dict[str, str]:
    path = _find_upwards("Rule2_method_classification.json")
    if not path:
        print("⚠️ Rule2_method_classification.json not found. Skipping methodology mapping.")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"⚠️ Could not read {path}: {exc}. Skipping methodology mapping.")
        return {}


def _map_methodologies(rows: List[Dict[str, Any]]):
    mapping = _load_method_mapping()
    if not mapping:
        return
    for row in rows:
        raw = row.get("Methodology", "")
        add_terms = row.get("_methodology_terms") or []
        combined_text = " ".join([raw, *add_terms])
        low = combined_text.lower()
        hits = set()
        for k, v in mapping.items():
            if k.lower() in low:
                hits.add(v)
        if hits:
            row["Methodology"] = "; ".join(sorted(hits))


def _make_key(row: Dict[str, Any]) -> str:
    def _norm(s: Any) -> str:
        s = ("" if s is None else str(s)).strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s-]", "", s)
        return s

    return " | ".join([
        _norm(row.get("Journal", "")),
        _norm(row.get("Author", "")),
        _norm(row.get("Title", "")),
    ])


# ---------------------------------------------------------------------------
# Core OpenAI interaction helpers
# ---------------------------------------------------------------------------

def _init_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide it in the environment or .env file before running agent1_openai."
        )
    return OpenAI(api_key=key)

def _prepare_prompt(pdf_name: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    system_content = [{"type": "input_text", "text": system_msg["content"]}]
    user_text = (
        f"{user_msg['content'].strip()}\n\n"
        f"The attached file is '{pdf_name}'. Follow all system instructions and return a JSON object that complies with the provided schema."
    )
    user_content = [{"type": "input_text", "text": user_text}]
    return system_content, user_content


def _clean_response_text(text: str) -> str:
    """Remove Markdown fences or extraneous text before JSON parsing."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_response_json(text: str, pdf_name: str) -> Dict[str, Any]:
    cleaned = _clean_response_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if match:
            snippet = match.group(1)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    raise RuntimeError(f"OpenAI response for {pdf_name} was not valid JSON.")

def _call_openai_with_pdf(client: OpenAI, pdf_path: Path, model: str, schema: dict | None) -> Dict[str, Any]:
    with pdf_path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="assistants")
    file_id = uploaded.id
    try:
        system_content, user_content = _prepare_prompt(pdf_path.name)
        user_content.append({"type": "input_file", "file_id": file_id})

        request_payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ],
            "temperature": temperature,
            "max_output_tokens": 2000,
        }

        # IMPORTANT: No response_format here (older SDKs don't accept it)
        response = client.responses.create(**request_payload)
        text = response.output_text
        return _parse_response_json(text, pdf_path.name)
    finally:
        try:
            client.files.delete(file_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Row construction + CSV writing
# ---------------------------------------------------------------------------


def _coerce_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    row = {col: "Unknown" for col in SUMMARY_COLUMNS}
    if not isinstance(data, dict):
        return row

    lower_map: Dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(key, str):
            lower_map[key.strip().lower()] = val

    for col in SUMMARY_COLUMNS:
        value: Any | None = None

        # Special handling for Commodity -> path + Deposit type
        if col == "Commodity":
            val = data.get("Commodity") or lower_map.get("commodity")
            if val:
                row["Commodity"] = commodity_obj_to_path(val)
                if not row.get("Deposit type") and isinstance(val, dict) and val.get("minor_category"):
                    row["Deposit type"] = str(val["minor_category"]).strip()
            continue

        if col == "Deposit type":
            val = data.get("Deposit type") or lower_map.get("deposit type")
            if isinstance(val, dict) and val.get("minor_category"):
                row[col] = str(val["minor_category"]).strip()
            elif val is not None:
                row[col] = str(val).strip()
            continue

        # Generic mapping + aliases
        if col in data and data[col] is not None:
            value = data[col]
        else:
            lowered = col.lower()
            if lowered in lower_map and lower_map[lowered] is not None:
                value = lower_map[lowered]
            else:
                for alias in _iter_summary_aliases(col):
                    if alias in data and data[alias] is not None:
                        value = data[alias]
                        break
                    alias_lower = alias.lower()
                    if alias_lower in lower_map and lower_map[alias_lower] is not None:
                        value = lower_map[alias_lower]
                        break
                else:
                    if col == "Journal Impact factor":
                        for key_lower, candidate in lower_map.items():
                            if "impact" in key_lower and "factor" in key_lower and candidate is not None:
                                value = candidate
                                break

        # Normalize booleans → strings for CSV
        if isinstance(value, bool):
            value = "True" if value else "False"

        if value is not None:
            row[col] = str(value).strip()

    return row

def _last_chance_fill(summary_row: Dict[str, Any], raw_summary: Dict[str, Any]):
    # Common containers the model sometimes uses
    for container_key in ("bibliography","biblio","metadata","paper","info"):
        sub = raw_summary.get(container_key)
        if not isinstance(sub, dict):
            continue
        # simple case-insensitive lift
        low = {k.lower(): v for k, v in sub.items() if isinstance(k,str)}
        if summary_row["Year"] == "Unknown" and "year" in low: summary_row["Year"] = str(low["year"]).strip()
        if summary_row["Month"] == "Unknown" and "month" in low: summary_row["Month"] = str(low["month"]).strip()
        if summary_row["Journal"] == "Unknown" and "journal" in low: summary_row["Journal"] = str(low["journal"]).strip()
        if summary_row["Journal Impact factor"] == "Unknown":
            for k in list(low.keys()):
                if "impact" in k and "factor" in k and sub[k] is not None:
                    summary_row["Journal Impact factor"] = str(sub[k]).strip()
                    break
        if summary_row["Title"] == "Unknown" and "title" in low: summary_row["Title"] = str(low["title"]).strip()
        if summary_row["Author"] == "Unknown":
            if "author" in low: summary_row["Author"] = "; ".join(sub["author"]) if isinstance(sub["author"], list) else str(sub["author"]).strip()
            if "authors" in low: summary_row["Author"] = "; ".join(sub["authors"]) if isinstance(sub["authors"], list) else str(sub["authors"]).strip()
        if summary_row["url"] == "Unknown":
            for k in ("url","link","doi"):
                if k in low and sub[k]:
                    summary_row["url"] = str(sub[k]).strip()
                    break

def _normalise_model_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalise varied OpenAI responses into a predictable structure.
       If the model returns both a top-level object and a nested `summary`,
       merge them (nested keys win), so we don't lose Year/Journal/etc.
    """

    def _clean_terms(values: Any) -> List[Any]:
        if not isinstance(values, list):
            return []
        return [item for item in values if item is not None]

    META_KEYS = {"summary", "key_points", "methodology_terms", "commodity_terms"}

    def _convert(item: Any) -> Dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        # 1) Start with top-level keys that look like summary fields
        merged_summary: Dict[str, Any] = {}
        for k, v in item.items():
            if not isinstance(k, str):
                continue
            if k in META_KEYS:
                continue
            # accept canonical keys (case-insensitive)
            if k in SUMMARY_KEYS_CANON or k.strip().lower() in {s.lower() for s in SUMMARY_KEYS_CANON}:
                merged_summary[k] = v

        # 2) If nested `summary` exists, it overrides top-level values
        if isinstance(item.get("summary"), dict):
            for k, v in item["summary"].items():
                merged_summary[k] = v

        return {
            "summary": merged_summary,
            "key_points": _clean_terms(item.get("key_points")),
            "methodology_terms": _clean_terms(item.get("methodology_terms")),
            "commodity_terms": _clean_terms(item.get("commodity_terms")),
        }

    if isinstance(payload, list):
        return [c for c in (_convert(x) for x in payload) if c]
    single = _convert(payload)
    return [single] if single else []



def _update_summary_csv(rows: List[Dict[str, Any]], csv_path: Path):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = ""
    if "_key" not in df_new.columns:
        df_new["_key"] = df_new.apply(_make_key, axis=1)

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        for col in SUMMARY_COLUMNS:
            if col not in df_old.columns:
                df_old[col] = ""
        if "_key" not in df_old.columns:
            df_old["_key"] = df_old.apply(_make_key, axis=1)
        old_keys = set(df_old["_key"].tolist())
        df_to_add = df_new[~df_new["_key"].isin(old_keys)].copy()
        combined = pd.concat([df_old, df_to_add], ignore_index=True)
        combined = combined.drop_duplicates(subset=["_key"], keep="first")
        combined = combined[SUMMARY_COLUMNS + ["_key"]]
        combined.to_csv(csv_path, index=False)
        print(f"✅ Appended {len(df_to_add)} new rows. Total now: {len(combined)} → {csv_path}")
    else:
        df_new = df_new.drop_duplicates(subset=["_key"], keep="first")
        df_new = df_new[SUMMARY_COLUMNS + ["_key"]]
        df_new.to_csv(csv_path, index=False)
        print(f"✅ Created {csv_path} with {len(df_new)} rows.")


def _update_details_csv(rows: List[Dict[str, Any]], csv_path: Path):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    for col in DETAIL_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = ""

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        for col in DETAIL_COLUMNS:
            if col not in df_old.columns:
                df_old[col] = ""
        combined = pd.concat([df_old, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["_parent_key", "order", "key_point"], keep="first")
        combined = combined[DETAIL_COLUMNS]
        combined.to_csv(csv_path, index=False)
    else:
        df_new = df_new.drop_duplicates(subset=["_parent_key", "order", "key_point"], keep="first")
        df_new = df_new[DETAIL_COLUMNS]
        df_new.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def process_pdfs(
    client: OpenAI,
    pdf_paths: Iterable[Path],
    model: str = MODEL_NAME,
    schema: dict = RESPONSE_FORMAT,
    summary_csv_path: Path | None = None,
    detail_csv_path: Path | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_csv_path = Path(summary_csv_path) if summary_csv_path else None
    detail_csv_path = Path(detail_csv_path) if detail_csv_path else None

    if summary_csv_path:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_csv_path:
        detail_csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_incrementally = summary_csv_path is not None or detail_csv_path is not None

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    failures: List[tuple[Path, Exception]] = []

    for pdf_path in pdf_paths:
        print(f"📄 Processing {pdf_path.name} ...")
        try:
            payload = _call_openai_with_pdf(client, pdf_path, model, schema)
        except Exception as exc:
            print(f"❌ Failed to analyse {pdf_path.name}: {exc}")
            failures.append((pdf_path, exc))
            continue

        raw_path = RAW_DIR / f"{pdf_path.stem}_response.json"
        raw_path.write_text(json.dumps(payload, indent=2))
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        row = row_from_model_json(payload)

    #     df = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
    #     # append-or-create CSV
    #     if SUMMARY_CSV.exists():
    #         old = pd.read_csv(SUMMARY_CSV)
    #         out = pd.concat([old, df], ignore_index=True)
    #     else:
    #         out = df
    #     out.to_csv(SUMMARY_CSV, index=False)
    #
    #     entries = _normalise_model_payload(payload)
    #     if not entries:
    #         error = RuntimeError("No usable summary data returned by the model")
    #         print(f"❌ No structured data extracted for {pdf_path.name}.")
    #         failures.append((pdf_path, error))
    #         continue
    #
    #     pdf_summary_rows: List[Dict[str, Any]] = []
    #     pdf_detail_rows: List[Dict[str, Any]] = []
    #
    #     for entry_index, entry in enumerate(entries, start=1):
    #         summary_payload = entry.get("summary", {})
    #         summary_row = _coerce_summary(summary_payload)
    #         _last_chance_fill(summary_row, summary_payload)
    #
    #         summary_row["_source_file"] = pdf_path.name
    #         entry_id = f"{pdf_path.name}::{entry_index}"
    #         summary_row["_entry_id"] = entry_id
    #         missing_keys = [k for k in ("Year", "Journal", "Title", "Author", "url") if
    #                         summary_row.get(k) in ("", "Unknown")]
    #         if missing_keys:
    #             print(f"🔎 Missing {missing_keys} for {pdf_path.name}. Model summary keys:",
    #                   list(summary_payload.keys()))
    #
    #         methodology_terms = [
    #             str(x).strip() for x in entry.get("methodology_terms", []) if str(x).strip()
    #         ]
    #         commodity_terms = [
    #             str(x).strip() for x in entry.get("commodity_terms", []) if str(x).strip()
    #         ]
    #         if (
    #             methodology_terms
    #             and not summary_row["Methodology"]
    #         ):
    #             summary_row["Methodology"] = "; ".join(
    #                 methodology_terms
    #             )
    #         if commodity_terms and not summary_row["Commodity (multiple)"]:
    #             summary_row["Commodity (multiple)"] = "; ".join(commodity_terms)
    #         summary_row["_methodology_terms"] = methodology_terms
    #         summary_row["_commodity_terms"] = commodity_terms
    #
    #         summary_rows.append(summary_row)
    #         pdf_summary_rows.append(summary_row)
    #
    #         key_points = entry.get("key_points") or []
    #         for idx, point in enumerate(key_points[:MAX_POINTS], start=1):
    #             detail_entry = {
    #                 "Title": summary_row.get("Title", ""),
    #                 "filename": pdf_path.name,
    #                 "order": idx,
    #                 # "key_point": str(point.get("summary", "")).strip(),
    #                 # "page_reference": str(point.get("page_reference", "")).strip(),
    #                 # "evidence": str(point.get("evidence", "")).strip(),
    #                 # "confidence": str(point.get("confidence", "")).strip(),
    #                 "_parent_key": "",  # to be populated after keys computed
    #                 "_entry_id": entry_id,
    #             }
    #             detail_rows.append(detail_entry)
    #             pdf_detail_rows.append(detail_entry)
    #
    #     if write_incrementally and pdf_summary_rows:
    #         # Fallbacks ONLY if missing
    #         need_method_map = any(not r.get("Methodology") for r in pdf_summary_rows)
    #         need_commodity_map = any(not r.get("Commodity") for r in
    #                                  pdf_summary_rows)  # Commodity is an object in JSON, but CSV holds a path string; check before CSV conversion.
    #
    #         if need_method_map:
    #             _map_methodologies(pdf_summary_rows)
    #
    #         if need_commodity_map:
    #             _map_commodities(pdf_summary_rows)
    #
    #         for row in pdf_summary_rows:
    #             row.setdefault("_methodology_terms", [])
    #             row.setdefault("_commodity_terms", [])
    #             row["_key"] = _make_key(row)
    #
    #         for detail in pdf_detail_rows:
    #             entry_id = detail.get("_entry_id")
    #             parent = None
    #             if entry_id:
    #                 parent = next(
    #                     (row for row in pdf_summary_rows if row.get("_entry_id") == entry_id),
    #                     None,
    #                 )
    #             if parent is None:
    #                 title = detail.get("Title", "")
    #                 parent = next(
    #                     (
    #                         row
    #                         for row in pdf_summary_rows
    #                         if row.get("Title") == title and row.get("_source_file") == pdf_path.name
    #                     ),
    #                     None,
    #                 )
    #             if parent:
    #                 detail["_parent_key"] = parent.get("_key", "")
    #             detail.pop("_entry_id", None)
    #
    #         if summary_csv_path:
    #             prepared_rows: List[Dict[str, Any]] = []
    #             for row in pdf_summary_rows:
    #                 prepared = {col: row.get(col, "") for col in SUMMARY_COLUMNS}
    #                 prepared["_key"] = row.get("_key", _make_key(row))
    #                 prepared_rows.append(prepared)
    #             if prepared_rows:
    #                 _update_summary_csv(prepared_rows, summary_csv_path)
    #
    #         # if detail_csv_path and pdf_detail_rows:
    #         #     prepared_details: List[Dict[str, Any]] = []
    #         #     for detail in pdf_detail_rows:
    #         #         prepared_detail = {col: detail.get(col, "") for col in DETAIL_COLUMNS}
    #         #         prepared_details.append(prepared_detail)
    #         #     if prepared_details:
    #         #         _update_details_csv(prepared_details, detail_csv_path)
    #
    # if not write_incrementally:
    #     # Apply mappings and compute keys after processing all PDFs
    #     _map_methodologies(summary_rows)
    #     _map_commodities(summary_rows)
    #
    #     for row in summary_rows:
    #         row.setdefault("_methodology_terms", [])
    #         row.setdefault("_commodity_terms", [])
    #         row["_key"] = _make_key(row)
    #
    #     for detail in detail_rows:
    #         entry_id = detail.get("_entry_id")
    #         parent = None
    #         if entry_id:
    #             parent = next((row for row in summary_rows if row.get("_entry_id") == entry_id), None)
    #         if parent is None:
    #             title = detail.get("Title", "")
    #             filename = detail.get("filename", "")
    #             parent = next(
    #                 (
    #                     row
    #                     for row in summary_rows
    #                     if row.get("Title") == title and row.get("_source_file") == filename
    #                 ),
    #                 None,
    #             )
    #         if parent:
    #             detail["_parent_key"] = parent.get("_key", "")
    #         detail.pop("_entry_id", None)
    # else:
    #     # Ensure helper IDs are removed when writing incrementally
    #     for detail in detail_rows:
    #         detail.pop("_entry_id", None)
    #
    # # Clean helper columns before returning
    # for row in summary_rows:
    #     row.pop("_methodology_terms", None)
    #     row.pop("_commodity_terms", None)
    #     row.pop("_source_file", None)
    #     row.pop("_entry_id", None)
    #
    # if failures:
    #     print(f"⚠️ {len(failures)} PDF(s) could not be processed.")

    return summary_rows, detail_rows


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse PDFs with OpenAI and export CSV summaries.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_PAPER_FOLDER),
        help="Folder containing PDF papers (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="OpenAI model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of PDFs to process (0 = all)",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=str(SUMMARY_CSV),
        help="Path to the summary CSV file",
    )
    # parser.add_argument(
    #     "--details-csv",
    #     type=str,
    #     default=str(DETAILS_CSV),
    #     help="Path to the key-findings CSV file",
    # )
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    pdf_paths = sorted(p for p in input_dir.glob("*.pdf") if p.is_file())
    if args.limit:
        pdf_paths = pdf_paths[: args.limit]
    if not pdf_paths:
        print(f"No PDF files found in {input_dir}.")
        return

    client = _init_client()

    summary_csv = Path(args.summary_csv)
    # detail_csv = Path(args.details_csv)

    start = time.time()
    summary_rows, detail_rows = process_pdfs(
        client,
        pdf_paths,
        schema=RESPONSE_FORMAT,
        model=args.model,
        summary_csv_path=summary_csv,
        # detail_csv_path=detail_csv,
    )
    elapsed = time.time() - start

    print(f"Processed {len(summary_rows)} PDFs in {elapsed:.1f}s")

    if summary_rows:
        print(f"✅ Summary saved to {summary_csv}")
    # if detail_rows:
    #     print(f"✅ Key findings saved to {detail_csv}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
