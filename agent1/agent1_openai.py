from __future__ import annotations
"""Code that sends papers to openai and get the structured data"""
"""Agent3 – Prospectivity report generation."""

import base64
import io
import json
import mimetypes
import os
import re
import time
import pathlib
import textwrap
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, dotenv_values

# ----------------- CONFIG FOR Input -----------------
dotenv_values()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

INPUT_PAPER_FOLDER = os.getenv("INPUT_PAPER_FOLDER")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured. Set it in your environment or .env file before running."
    )

with open("../Rule1_OreDeposit.json", "r", encoding="utf-8") as f:
    rules = json.load(f)
rules1_str = json.dumps(rules, ensure_ascii=False)

with open("../Rule2_method_classification.json", "r", encoding="utf-8") as f:
    rules = json.load(f)
rules2_str = json.dumps(rules, ensure_ascii=False)

# --- CONFIG ---
# Build a strict system message that hard-scopes the subject and forbids off-topic items.
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
    )
}

# User query with explicit key contract and ranking preference
user_msg = {
    "role": "user",
    "content": (
        "Find and summarize recent research papers or research reports (2022–2025) on MINERAL deposit prospectivity/"
        "potential mapping/prediction using machine learning/artificial intelligence. Focus on data-driven geological targeting.\n"
        "STRICTLY include only items that match the scope above and whose Commodity can be labeled using the provided Rules.\n"
        "Prefer higher impact journals and top geological surveys/institutes.\n"
        "Avoid duplicates and excessive single-publisher dominance.\n\n"        
        "Use EXACT keys per item: Index, Year, Month, Journal, Journal Impact factor, Methodology, Title, Author, Study Country, Study Province/Region"
        "URL of the papers, Availability of original data, URL of the data server, Commodity, "
        "(Commodity 2, Commodity 3, ...; Optional when multiple commodities).\n"
        "Unknown fields → 'Unknown'. Return ONLY valid JSON (array of objects)."
    )
}

messages = [system_msg, user_msg]

print(messages)
# -----------------
@dataclass
class ContextBundle:
    """All artefacts that Agent3 consumes."""

    json_summaries: List[str]
    document_name: Optional[str]
    document_text: Optional[str]
    document_path: Optional[str]
    prediction_blocks: List[str]

    def compose_context(self) -> str:
        sections: List[str] = []
        if self.document_name or self.document_text:
            header = "Project Document"
            body_lines = []
            if self.document_name:
                body_lines.append(f"Filename: {self.document_name}")
            if self.document_path:
                body_lines.append(f"Located at: {self.document_path}")
            if self.document_text:
                excerpt = self.document_text.strip()
                if len(excerpt) > 4000:
                    excerpt = excerpt[:4000].rstrip() + "…"
                body_lines.append("Excerpt:\n" + excerpt)
            sections.append(_format_section(header, "\n".join(body_lines)))

        if self.json_summaries:
            sections.append(_format_section("Context JSON files", "\n\n".join(self.json_summaries)))

        if self.prediction_blocks:
            sections.append(_format_section("Prediction outputs", "\n\n".join(self.prediction_blocks)))

        if not sections:
            return "No context files were found."
        return "\n\n".join(sections)


def _format_section(title: str, body: str) -> str:
    body = body.strip()
    return f"## {title}\n{body}" if body else f"## {title}\n(Empty)"

def _safe_json_load(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Could not parse {path.name}: {exc}"}

def _summarize_step1(payload: Dict[str, Any]) -> str:
    doc = payload.get("doc") or "(unknown)"
    response = payload.get("response", {})
    if isinstance(response, dict):
        keys = [f"{k}: {v}" for k, v in response.items()]
        content = "\n".join(keys)
    else:
        content = str(response)
    model = payload.get("model") or ""
    timestamp = payload.get("timestamp") or ""
    lines = [f"Document analysed: {doc}"]
    if model:
        lines.append(f"Model: {model}")
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")
    if content:
        lines.append("Extracted metadata:\n" + content)
    if payload.get("doc_excerpt"):
        lines.append("Stored excerpt available for downstream agents.")
    return "\n".join(lines)


def _summarize_step2(payload: Dict[str, Any]) -> str:
    pre = payload.get("preprocess") or {}
    src = pre.get("source_path")
    features = pre.get("features") or []
    x_col = pre.get("x_col") or "?"
    y_col = pre.get("y_col") or "?"
    label_col = pre.get("label_col") or "?"
    lbl_mode = "separate" if pre.get("labels_uploaded_separately") else "embedded"
    lines = [
        f"Source path: {src or '(unknown)'}",
        f"Coordinates: x={x_col} | y={y_col}",
        f"Label column: {label_col} ({lbl_mode})",
        f"Selected features ({len(features)}): {', '.join(map(str, features))}" if features else "No features recorded.",
    ]

    numeric = pre.get("numeric") or {}
    if numeric:
        sample_numeric = []
        for name, cfg in list(numeric.items())[:5]:
            rng = cfg.get("num_ranges_state") or cfg.get("range") or []
            original = cfg.get("original") or {}
            sample_numeric.append(
                f"{name}: range={tuple(round(v, 3) for v in rng)} original_min={original.get('min')} original_max={original.get('max')}"
            )
        lines.append("Numeric feature ranges (up to 5 shown):\n" + "\n".join(sample_numeric))

    categorical = pre.get("categorical") or {}
    if categorical:
        lines.append(f"Categorical features recorded: {', '.join(list(categorical.keys())[:5])}…" if len(categorical) > 5 else f"Categorical features recorded: {', '.join(categorical.keys())}")

    return "\n".join(lines)


def _summarize_generic_json(name: str, payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2)[:1200]
    if len(text) == 1200:
        text = text.rstrip() + "…"
    return f"{name}:\n{text}"


def _list_context_json(context_dir: Path) -> List[str]:
    summaries: List[str] = []
    for path in sorted(context_dir.glob("*.json")):
        payload = _safe_json_load(path)
        name = path.name
        if name.startswith("step1"):
            summaries.append(_summarize_step1(payload))
        elif name.startswith("step2"):
            summaries.append(_summarize_step2(payload))
        else:
            summaries.append(_summarize_generic_json(name, payload))
    return summaries


def _guess_document(context_dir: Path, summaries: Iterable[Tuple[Path, Dict[str, Any]]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    doc_name: Optional[str] = None
    doc_text: Optional[str] = None
    doc_path: Optional[str] = None

    for path, payload in summaries:
        if path.name.startswith("step1"):
            doc_name = payload.get("doc") or doc_name
            doc_text = payload.get("doc_excerpt") or payload.get("doc_text") or doc_text
            stored_path = payload.get("doc_path")
            if stored_path and Path(stored_path).exists():
                doc_path = stored_path
            if not doc_path and doc_name:
                candidates = [
                    context_dir / doc_name,
                    Path("outputs") / doc_name,
                    Path(doc_name),
                ]
                for cand in candidates:
                    if cand.exists():
                        doc_path = str(cand)
                        break
            break

    if doc_path and doc_path.lower().endswith(".pdf") and doc_text is None:
        try:
            import fitz  # type: ignore

            with fitz.open(doc_path) as doc:
                pages = [p.get_text() or "" for p in doc]
            doc_text = "\n".join(pages)
        except Exception:
            doc_text = None

    return doc_name, doc_text, doc_path


def _iter_context_payloads(context_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    payloads: List[Tuple[Path, Dict[str, Any]]] = []
    for path in sorted(context_dir.glob("*.json")):
        payloads.append((path, _safe_json_load(path)))
    return payloads


def _summarize_predictions(predictions_dir: Path) -> List[str]:
    blocks: List[str] = []
    if not predictions_dir.exists():
        return blocks

    for method_dir in sorted(p for p in predictions_dir.glob("*") if p.is_dir()):
        prefix = method_dir.name
        run_dir = method_dir / prefix
        fig_path = run_dir / f"{prefix}_predictions.png"
        csv_path = run_dir / f"{prefix}_predictions.csv"
        metrics_path = run_dir / f"{prefix}_metrics.json"
        shap_path = method_dir / f"{prefix}_predictions_shap_sum.png"

        lines = [f"Method: {prefix}"]
        if fig_path.exists():
            lines.append(f"Prediction figure: {fig_path}")
        if shap_path.exists():
            lines.append(f"SHAP summary: {shap_path}")

        if metrics_path.exists():
            metrics = _safe_json_load(metrics_path)
            summary = metrics.get("summary") or metrics
            metric_lines = [f"{k}: {round(v, 4) if isinstance(v, (int, float)) else v}" for k, v in summary.items()]
            lines.append("Metrics:\n" + "\n".join(metric_lines))

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                lines.extend(_describe_prediction_dataframe(df))
            except Exception as exc:
                lines.append(f"Could not read predictions CSV: {exc}")
        else:
            lines.append("Prediction CSV not found.")

        blocks.append("\n".join(lines))

    return blocks


def _describe_prediction_dataframe(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    cols = df.columns
    prospect_col = next((c for c in cols if c.lower().startswith("prospect")), None)
    if prospect_col is None:
        return lines

    series = pd.to_numeric(df[prospect_col], errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return lines

    lines.append(
        "Prospectivity stats: "
        + f"count={len(valid)}, mean={valid.mean():.4f}, std={valid.std():.4f}, min={valid.min():.4f}, max={valid.max():.4f}"
    )
    quantiles = valid.quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    quant_line = ", ".join(f"q{int(q*100)}={val:.4f}" for q, val in quantiles.items())
    lines.append(f"Quantiles: {quant_line}")

    x_col = next((c for c in cols if c.lower() in {"x", "lon", "longitude", "longitude_epsg3978"}), None)
    y_col = next((c for c in cols if c.lower() in {"y", "lat", "latitude", "latitude_epsg3978"}), None)

    try:
        top_df = df.loc[valid.nlargest(10).index]
    except Exception:
        top_df = df.head(10)

    def format_row(row: pd.Series) -> str:
        val = row.get(prospect_col, float("nan"))
        x = row.get(x_col) if x_col else "?"
        y = row.get(y_col) if y_col else "?"
        return f"score={val:.4f} @ ({x}, {y})"

    top_lines = [format_row(row) for _, row in top_df.iterrows()]
    lines.append("Top-ranked locations (up to 10):\n" + "\n".join(top_lines))

    if "uncertainty" in df.columns:
        unc = pd.to_numeric(df["uncertainty"], errors="coerce").dropna()
        if not unc.empty:
            lines.append(
                "Uncertainty stats: "
                + f"mean={unc.mean():.4f}, std={unc.std():.4f}, min={unc.min():.4f}, max={unc.max():.4f}"
            )

    return lines


def _build_context_bundle(context_dir: Path, predictions_dir: Path) -> ContextBundle:
    context_payloads = _iter_context_payloads(context_dir)
    summaries = _list_context_json(context_dir)
    doc_name, doc_text, doc_path = _guess_document(context_dir, context_payloads)
    prediction_blocks = _summarize_predictions(predictions_dir)
    return ContextBundle(summaries, doc_name, doc_text, doc_path, prediction_blocks)


def _clean_path_string(value: str) -> str:
    candidate = value.strip()
    candidate = candidate.strip("\"'")
    candidate = candidate.strip()
    if candidate.startswith("(") and candidate.endswith(")") and len(candidate) > 2:
        candidate = candidate[1:-1].strip()
    if candidate.startswith("[") and candidate.endswith("]") and len(candidate) > 2:
        candidate = candidate[1:-1].strip()
    while candidate and candidate[-1] in ",.;:)]}":
        candidate = candidate[:-1]
    return candidate


def _extract_image_payloads(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    assets: Dict[str, Dict[str, str]] = {}
    chosen_order: List[str] = []
    current_method: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("method:"):
            method = line.split(":", 1)[1].strip()
            if method:
                current_method = method
                assets.setdefault(method, {})
        elif "chosen result" in lowered:
            chosen_val = line.split(":", 1)[-1].strip()
            if chosen_val:
                chosen_order.append(chosen_val)
        elif lowered.startswith("prediction figure:"):
            path = line.split(":", 1)[1].strip()
            if path and current_method:
                assets.setdefault(current_method, {})["prediction"] = path
        elif lowered.startswith("shap summary:"):
            path = line.split(":", 1)[1].strip()
            if path and current_method:
                assets.setdefault(current_method, {})["shap"] = path

    unique_chosen: List[str] = []
    seen_methods: set[str] = set()
    for name in chosen_order:
        if name and name not in seen_methods:
            seen_methods.add(name)
            unique_chosen.append(name)

    method_order: List[str] = [m for m in assets if m]
    target_methods: List[str] = [m for m in unique_chosen if m in assets] or method_order

    seen_paths: set[str] = set()
    items: List[Dict[str, Any]] = []
    records: List[Dict[str, str]] = []

    def _add_path(raw_path: str, *, method: Optional[str], kind: str) -> None:
        normalized = _clean_path_string(raw_path)
        if not normalized or normalized in seen_paths:
            return
        path_obj = Path(normalized).expanduser()
        if not path_obj.exists():
            return
        try:
            data = path_obj.read_bytes()
        except Exception:
            return
        mime = mimetypes.guess_type(str(path_obj))[0] or "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        seen_paths.add(normalized)
        try:
            display_path = os.path.relpath(path_obj.resolve(), Path.cwd())
        except Exception:
            display_path = str(path_obj)
        records.append(
            {
                "path": display_path,
                "kind": kind,
                "method": method or "",
                "transport": "inline-data",
            }
        )
        data_url = f"data:{mime};base64,{b64}"
        items.append(
            {
                "type": "input_image",
                "image_url": data_url,
                "detail": "low",
            }
        )

    for method in target_methods:
        entry = assets.get(method, {})
        if entry.get("prediction"):
            _add_path(entry["prediction"], method=method, kind="prediction")
        if entry.get("shap"):
            _add_path(entry["shap"], method=method, kind="shap")

    if not items:
        pattern = re.compile(
            r"((?:[A-Za-z]:)?[^\s]*?outputs[\\/]+predictions[^\s]*?_(?:predictions|predictions_shap_sum)\.png)",
            re.IGNORECASE,
        )
        for match in pattern.findall(text):
            _add_path(match, method=None, kind="image")

    return items, records


def _describe_attachment(record: Dict[str, str]) -> str:
    path = record.get("path") or "(unknown path)"
    annotations: List[str] = []
    kind = (record.get("kind") or "").strip()
    if kind:
        annotations.append(kind)
    method = (record.get("method") or "").strip()
    if method:
        annotations.append(f"method={method}")
    transport = (record.get("transport") or "").strip()
    if transport:
        annotations.append(f"transport={transport}")
    if annotations:
        return f"{path} ({', '.join(annotations)})"
    return path


def _log_attachment_event(model_name: str, records: List[Dict[str, str]], succeeded: bool) -> None:
    if records:
        action = "Sent" if succeeded else "Attempted to send"
        plural = "s" if len(records) != 1 else ""
        print(f"[Agent3] {action} {len(records)} file{plural} to model '{model_name}':")
        for record in records:
            print(f"  - {_describe_attachment(record)}")
    else:
        message = (
            f"[Agent3] No files were sent to model '{model_name}'."
            if succeeded
            else f"[Agent3] No image attachments were available for model '{model_name}'."
        )
        print(message)


def _fallback_report(bundle: ContextBundle) -> str:
    lines = ["\n**Note:** OpenAI response unavailable; provided heuristic summary instead."]
    return "\n".join(lines)



######################################################################
def run_agent1_report(
    model: Optional[str] = None,
    context_dir: str | Path = "outputs/context",
    predictions_dir: str | Path = "outputs/predictions",
    report_dir: str | Path = "outputs/reports",
    max_words: int = 1000,
) -> Dict[str, Any]:
    context_dir = Path(context_dir)
    predictions_dir = Path(predictions_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle = _build_context_bundle(context_dir, predictions_dir)
    context_text = bundle.compose_context()

    model_name = OPENAI_MODEL or "gpt-5o-mini"
    used_fallback = False
    error: Optional[str] = None
    attachments_prepared: List[Dict[str, str]] = []
    attachments_sent: List[Dict[str, str]] = []
    attachment_message = ""

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": context_text}]

    if OPENAI_API_KEY:
        # image_payloads, attachments_prepared = _extract_image_payloads(context_text)
        image_payloads, attachments_prepared = None, None
        try:
            report = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": messages},
                    {"role": "user", "content": user_content},
                ],
                max_output_tokens=5000,
            ).output_text.strip()

        except Exception as exc:
            error = str(exc)
            report = _fallback_report(bundle)
            used_fallback = True
            _log_attachment_event(model_name, attachments_prepared, succeeded=False)
            if attachments_prepared:
                plural = "s" if len(attachments_prepared) != 1 else ""
                lines = [
                    f"Attempted to send {len(attachments_prepared)} file{plural} to model `{model_name}`, but the request failed.",
                ]
                lines.extend(f"- {_describe_attachment(record)}" for record in attachments_prepared)
                if error:
                    lines.append(f"⚠️ {error}")
                attachment_message = "\n".join(lines)
            else:
                failure_note = "The request failed before any image attachments could be processed."
                if error:
                    failure_note = f"{failure_note} ({error})"
                attachment_message = f"No files were sent to model `{model_name}`. {failure_note}"
    else:
        error = "OPENAI_API_KEY not set"
        report = _fallback_report(bundle)
        used_fallback = True
        log_message = f"No files were sent to model '{model_name}' because the OpenAI API key is not set."
        print(f"[Agent3] {log_message}")
        attachment_message = log_message.replace("'", "`")

    # TODO: Write new CSV format
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = report_dir / f"agent1_result_{INPUT_PAPER_FOLDER}_{timestamp}.pdf"

    return {
        "report": report,
        "csv_path": str(csv_path),
        "model": model_name,
        "used_fallback": used_fallback,
        "error": error,
        "context_summary": context_text,
        "attachments": attachments_sent,
        "attachment_message": attachment_message,
    }

if __name__ == "__main__":  # pragma: no cover - CLI entry
    run_agent1_report()