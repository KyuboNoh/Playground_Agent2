from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from dotenv import load_dotenv

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

INPUT_PAPER_FOLDER = Path(os.getenv("INPUT_PAPER_FOLDER", "input_papers"))
OUT_DIR = Path(os.getenv("AGENT1_OUTPUT_DIR", "out/agent1_openai"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = Path(os.getenv("AGENT1_SUMMARY_CSV", "papers_summary.csv"))
DETAILS_CSV = Path(os.getenv("AGENT1_DETAILS_CSV", "papers_key_findings.csv"))
MODEL_NAME = os.getenv("AGENT1_OPENAI_MODEL", "gpt-4o-mini")
MAX_POINTS = int(os.getenv("AGENT1_MAX_KEY_POINTS", "5"))

# Multi-line instructions sent to the model alongside each PDF.
rules_str = (
    "You are a research assistant helping a mineral exploration team. "
    "Read the attached PDF (a journal article) carefully. "
    "Extract structured bibliographic metadata, a concise abstract, "
    "and the methodological/commodity information required for our tracking sheet. "
    "Focus on the actual content of the PDF—do not fabricate values. "
    "If a field is not explicitly stated, return an empty string for that field. "
    "Additionally, summarise the paper's key findings into short bullet points."
)

SUMMARY_COLUMNS = [
    "Year",
    "Month",
    "Journal",
    "Journal Impact factor",
    "Commodity (multiple)",
    "Methodology (from given file with classification)",
    "Title",
    "Author",
    "abtract",
    "availability of original data",
    "url",
]

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
                        "abtract": {"type": "string"},
                        "availability of original data": {"type": "string"},
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
            "abtract",
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
    path = _find_upwards("method_classification.json")
    if not path:
        print("⚠️ method_classification.json not found. Skipping methodology mapping.")
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


def _prepare_prompt(pdf_name: str, extra_rules: str | None = None) -> str:
    extra = (extra_rules or rules_str).strip()
    return (
        f"The attached file is '{pdf_name}'.\n"  # context for the model
        f"Follow these rules when analysing the document:\n{extra}\n\n"
        "Return a JSON object that complies with the provided schema."
    )


def _call_openai_with_pdf(client: OpenAI, pdf_path: Path, model: str, schema: dict) -> Dict[str, Any]:
    with pdf_path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="assistants")

    file_id = uploaded.id
    try:
        prompt = _prepare_prompt(pdf_path.name)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_file", "file_id": file_id},
                    ],
                }
            ],
            response_format=schema,
            temperature=0.2,
        )
        text = response.output_text
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI response for {pdf_path.name} was not valid JSON."
            ) from exc
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
    for col in SUMMARY_COLUMNS:
        if col in data and data[col] is not None:
            row[col] = str(data[col]).strip()
    return row


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
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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

        summary_payload = payload.get("summary", {})
        summary_row = _coerce_summary(summary_payload)
        summary_row["_source_file"] = pdf_path.name

        methodology_terms = [
            str(x).strip() for x in payload.get("methodology_terms", []) if str(x).strip()
        ]
        commodity_terms = [
            str(x).strip() for x in payload.get("commodity_terms", []) if str(x).strip()
        ]
        if methodology_terms and not summary_row["Methodology (from given file with classification)"]:
            summary_row["Methodology (from given file with classification)"] = "; ".join(methodology_terms)
        if commodity_terms and not summary_row["Commodity (multiple)"]:
            summary_row["Commodity (multiple)"] = "; ".join(commodity_terms)
        summary_row["_methodology_terms"] = methodology_terms
        summary_row["_commodity_terms"] = commodity_terms

        summary_rows.append(summary_row)

        key_points = payload.get("key_points") or []
        for idx, point in enumerate(key_points[:MAX_POINTS], start=1):
            detail_rows.append(
                {
                    "Title": summary_row.get("Title", ""),
                    "filename": pdf_path.name,
                    "order": idx,
                    "key_point": str(point.get("summary", "")).strip(),
                    "page_reference": str(point.get("page_reference", "")).strip(),
                    "evidence": str(point.get("evidence", "")).strip(),
                    "confidence": str(point.get("confidence", "")).strip(),
                    "_parent_key": "",  # to be populated after keys computed
                }
            )

    # Apply mappings and compute keys
    _map_methodologies(summary_rows)
    _map_commodities(summary_rows)

    for row in summary_rows:
        row.setdefault("_methodology_terms", [])
        row.setdefault("_commodity_terms", [])
        row["_key"] = _make_key(row)

    for detail in detail_rows:
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

    # Clean helper columns before returning
    for row in summary_rows:
        row.pop("_methodology_terms", None)
        row.pop("_commodity_terms", None)
        row.pop("_source_file", None)

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
    parser.add_argument(
        "--details-csv",
        type=str,
        default=str(DETAILS_CSV),
        help="Path to the key-findings CSV file",
    )
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

    start = time.time()
    summary_rows, detail_rows = process_pdfs(client, pdf_paths, model=args.model)
    elapsed = time.time() - start

    print(f"Processed {len(summary_rows)} PDFs in {elapsed:.1f}s")

    summary_csv = Path(args.summary_csv)
    detail_csv = Path(args.details_csv)

    _update_summary_csv(summary_rows, summary_csv)
    _update_details_csv(detail_rows, detail_csv)
    if detail_rows:
        print(f"✅ Key findings saved to {detail_csv}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()