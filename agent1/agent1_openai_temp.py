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
# (sourced from agent1_openai.py to keep behaviour aligned)
# ---------------------------------------------------------------------------
system_msg = {
    "role": "system",
    "content": (
        "You are an expert geoscientist and computational scientist.\n"
        "TASK SCOPE (hard constraints):\n"
        "• Read each paper in pdf and summarize with given RULE that will be shown below.\n"
        "COMMODITY RULES (authoritative):\n"
        "you MUST return a CSV files with EXACT keys "
        "major_category, minor_category, commodities_elements, commodities_combined (no extra keys, no strings). "
        "If you cannot map a paper to a commodity in the rules, OMIT that paper entirely.\n"
        f"{rules1_str}\n\n"
        "METHODOLOGY RULES (authoritative):\n"
        "you MUST return a CSV files with EXACT keys "
        f"{rules2_str}\n\n"
        "OUTPUT RULES:\n"
        "• Return in CSV format (no prose/markdown/code fences). If nothing qualifies, return [].\n"
        "• Do NOT guess: if a field is unknown, set it to 'Unknown'.\n"
    ),
}

user_msg = {
    "role": "user",
    "content": (
        "You are a research assistant helping a mineral exploration team. "
        "Read the attached PDF (a journal article) carefully. "
        "Extract structured bibliographic metadata, a concise abstract, "
        "and the methodological/commodity information required for our tracking sheet. "
        "Focus on the actual content of the PDF—do not fabricate values. "
        "If a field is not explicitly stated, return an empty string for that field. "
        "Additionally, summarise the paper's key findings into short bullet points.\n\n"
        "Use EXACT keys per item: Index, Year, Month, Journal, Journal Impact factor, Methodology, Title, Author, Study Country, Study Province/Region"
        "URL of the papers, Public availability original dataset, Training/Application dataset type, "
        "Training vs Application dataset relationship, Training to application scope, URL of the data server, Commodity, "
        "(Commodity 2, Commodity 3, ...; Optional when multiple commodities).\n"
        "For 'Training vs Application dataset relationship', return 'Same' when the same dataset (even if split into subsets) is used for both training and application; otherwise return 'Different'."
        " When the relationship is 'Different', describe the transfer as 'Region A to Region B' (or similar) in 'Training to application scope' using the study areas discussed in the paper.\n"
        "Unknown fields → 'Unknown'. Return ONLY valid JSON (array of objects)."
    ),
}

# ---------------------------------------------------------------------------
# Summary CSV configuration (tabular output written after model parsing)
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "Year",
    "Month",
    "Journal",
    "Journal Impact factor",
    "Commodity (multiple)",
    "Methodology (from given file with classification)",
    "Title",
    "Author",
    "abstract",
    "Public availability original dataset",
    "Training/Application dataset type",
    "Training vs Application dataset relationship",
    "Training to application scope",
    "url",
]

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
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "object",
                    "properties": {
                        "Year": {"type": "string"},
                        "Month": {"type": "string"},
                        "Journal": {"type": "string"},
                        "Journal Impact factor": {"type": "string"},
                        "Commodity (multiple)": {"type": "string"},
                        "Methodology (from given file with classification)": {"type": "string"},
                        "Title": {"type": "string"},
                        "Author": {"type": "string"},
                        "abstract": {"type": "string"},
                        "Public availability original dataset": {"type": "boolean"},
                        "Training/Application dataset type": {"type": "string"},
                        "Training vs Application dataset relationship": {"type": "string"},
                        "Training to application scope": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["Title"],
                    "additionalProperties": False,
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "page_reference": {"type": "string"},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "string"},
                        },
                        "required": ["summary"],
                        "additionalProperties": False,
                    },
                },
                "methodology_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "commodity_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
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
            "Abstract",
            "abstract",
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
        raw = row.get("Methodology (from given file with classification)", "")
        add_terms = row.get("_methodology_terms") or []
        combined_text = " ".join([raw, *add_terms])
        low = combined_text.lower()
        hits = set()
        for k, v in mapping.items():
            if k.lower() in low:
                hits.add(v)
        if hits:
            row["Methodology (from given file with classification)"] = "; ".join(sorted(hits))


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


def _call_openai_with_pdf(client: OpenAI, pdf_path: Path, model: str, schema: dict) -> Dict[str, Any]:
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
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
        }
        if schema:
            request_payload["response_format"] = schema

        try:
            response = client.responses.create(**request_payload)
        except TypeError as exc:
            # Older versions of the OpenAI Python client (<= 1.14) do not yet
            # support the ``response_format`` argument on ``responses.create``.
            # Fall back to a plain JSON response by retrying without the schema
            # when we encounter this specific incompatibility.
            if "response_format" in str(exc) and "unexpected keyword" in str(exc):
                request_payload.pop("response_format", None)
                response = client.responses.create(**request_payload)
            else:
                raise
        text = response.output_text
        print(text)
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
    row = {col: "" for col in SUMMARY_COLUMNS}
    if not isinstance(data, dict):
        return row

    lower_map: Dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(key, str):
            lower_map[key.strip().lower()] = val

    for col in SUMMARY_COLUMNS:
        value: Any | None = None
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
        if value is not None:
            row[col] = str(value).strip()
    return row


def _normalise_model_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalise varied OpenAI responses into a predictable structure."""

    def _clean_terms(values: Any) -> List[Any]:
        if not isinstance(values, list):
            return []
        cleaned: List[Any] = []
        for item in values:
            if item is None:
                continue
            cleaned.append(item)
        return cleaned

    def _convert(item: Any) -> Dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        summary_data: Dict[str, Any]
        if isinstance(item.get("summary"), dict):
            summary_data = item["summary"]
        else:
            meta_keys = {"summary", "key_points", "methodology_terms", "commodity_terms"}
            summary_data = {
                key: value
                for key, value in item.items()
                if isinstance(key, str) and key not in meta_keys
            }

        return {
            "summary": summary_data,
            "key_points": _clean_terms(item.get("key_points")),
            "methodology_terms": _clean_terms(item.get("methodology_terms")),
            "commodity_terms": _clean_terms(item.get("commodity_terms")),
        }

    if isinstance(payload, list):
        normalised: List[Dict[str, Any]] = []
        for entry in payload:
            converted = _convert(entry)
            if converted:
                normalised.append(converted)
        return normalised

    converted = _convert(payload)
    return [converted] if converted else []


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

        entries = _normalise_model_payload(payload)
        if not entries:
            error = RuntimeError("No usable summary data returned by the model")
            print(f"❌ No structured data extracted for {pdf_path.name}.")
            failures.append((pdf_path, error))
            continue

        pdf_summary_rows: List[Dict[str, Any]] = []
        pdf_detail_rows: List[Dict[str, Any]] = []

        for entry_index, entry in enumerate(entries, start=1):
            summary_payload = entry.get("summary", {})
            summary_row = _coerce_summary(summary_payload)
            summary_row["_source_file"] = pdf_path.name
            entry_id = f"{pdf_path.name}::{entry_index}"
            summary_row["_entry_id"] = entry_id

            methodology_terms = [
                str(x).strip() for x in entry.get("methodology_terms", []) if str(x).strip()
            ]
            commodity_terms = [
                str(x).strip() for x in entry.get("commodity_terms", []) if str(x).strip()
            ]
            if (
                methodology_terms
                and not summary_row["Methodology (from given file with classification)"]
            ):
                summary_row["Methodology (from given file with classification)"] = "; ".join(
                    methodology_terms
                )
            if commodity_terms and not summary_row["Commodity (multiple)"]:
                summary_row["Commodity (multiple)"] = "; ".join(commodity_terms)
            summary_row["_methodology_terms"] = methodology_terms
            summary_row["_commodity_terms"] = commodity_terms

            summary_rows.append(summary_row)
            pdf_summary_rows.append(summary_row)

            key_points = entry.get("key_points") or []
            for idx, point in enumerate(key_points[:MAX_POINTS], start=1):
                detail_entry = {
                    "Title": summary_row.get("Title", ""),
                    "filename": pdf_path.name,
                    "order": idx,
                    "key_point": str(point.get("summary", "")).strip(),
                    "page_reference": str(point.get("page_reference", "")).strip(),
                    "evidence": str(point.get("evidence", "")).strip(),
                    "confidence": str(point.get("confidence", "")).strip(),
                    "_parent_key": "",  # to be populated after keys computed
                    "_entry_id": entry_id,
                }
                detail_rows.append(detail_entry)
                pdf_detail_rows.append(detail_entry)

        if write_incrementally and pdf_summary_rows:
            _map_methodologies(pdf_summary_rows)
            _map_commodities(pdf_summary_rows)

            for row in pdf_summary_rows:
                row.setdefault("_methodology_terms", [])
                row.setdefault("_commodity_terms", [])
                row["_key"] = _make_key(row)

            for detail in pdf_detail_rows:
                entry_id = detail.get("_entry_id")
                parent = None
                if entry_id:
                    parent = next(
                        (row for row in pdf_summary_rows if row.get("_entry_id") == entry_id),
                        None,
                    )
                if parent is None:
                    title = detail.get("Title", "")
                    parent = next(
                        (
                            row
                            for row in pdf_summary_rows
                            if row.get("Title") == title and row.get("_source_file") == pdf_path.name
                        ),
                        None,
                    )
                if parent:
                    detail["_parent_key"] = parent.get("_key", "")
                detail.pop("_entry_id", None)

            if summary_csv_path:
                prepared_rows: List[Dict[str, Any]] = []
                for row in pdf_summary_rows:
                    prepared = {col: row.get(col, "") for col in SUMMARY_COLUMNS}
                    prepared["_key"] = row.get("_key", _make_key(row))
                    prepared_rows.append(prepared)
                if prepared_rows:
                    _update_summary_csv(prepared_rows, summary_csv_path)

            # if detail_csv_path and pdf_detail_rows:
            #     prepared_details: List[Dict[str, Any]] = []
            #     for detail in pdf_detail_rows:
            #         prepared_detail = {col: detail.get(col, "") for col in DETAIL_COLUMNS}
            #         prepared_details.append(prepared_detail)
            #     if prepared_details:
            #         _update_details_csv(prepared_details, detail_csv_path)

    if not write_incrementally:
        # Apply mappings and compute keys after processing all PDFs
        _map_methodologies(summary_rows)
        _map_commodities(summary_rows)

        for row in summary_rows:
            row.setdefault("_methodology_terms", [])
            row.setdefault("_commodity_terms", [])
            row["_key"] = _make_key(row)

        for detail in detail_rows:
            entry_id = detail.get("_entry_id")
            parent = None
            if entry_id:
                parent = next((row for row in summary_rows if row.get("_entry_id") == entry_id), None)
            if parent is None:
                title = detail.get("Title", "")
                filename = detail.get("filename", "")
                parent = next(
                    (
                        row
                        for row in summary_rows
                        if row.get("Title") == title and row.get("_source_file") == filename
                    ),
                    None,
                )
            if parent:
                detail["_parent_key"] = parent.get("_key", "")
            detail.pop("_entry_id", None)
    else:
        # Ensure helper IDs are removed when writing incrementally
        for detail in detail_rows:
            detail.pop("_entry_id", None)

    # Clean helper columns before returning
    for row in summary_rows:
        row.pop("_methodology_terms", None)
        row.pop("_commodity_terms", None)
        row.pop("_source_file", None)
        row.pop("_entry_id", None)

    if failures:
        print(f"⚠️ {len(failures)} PDF(s) could not be processed.")

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
