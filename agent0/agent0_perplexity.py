"""Code that calls perplexity to get the research papers"""

import os, re
import json
import time
import pathlib
import pandas as pd
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests


# ----------------- CONFIG FOR Input -----------------
dotenv_values()
load_dotenv()
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_MODEL = os.getenv("PPLX_MODEL")
PPLX_SEARCH_FILTER = os.getenv("PPLX_SEARCH_FILTER")

if not PPLX_API_KEY:
    raise RuntimeError(
        "PPLX_API_KEY is not configured. Set it in your environment or .env file before running."
    )

with open("../Rule1_OreDeposit.json", "r", encoding="utf-8") as f:
    rules = json.load(f)
rules_str = json.dumps(rules, ensure_ascii=False)

# ----------------- CONFIG FOR OUTPUT -----------------
OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)
ts = time.strftime("%Y%m%d-%H%M%S")
raw_http_path = OUT_DIR / f"perplexity_http_{ts}.txt"

# Build a schema snippet for a Commodity OBJECT
commodity_obj_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "major_category": {"type": "string"},
        "minor_category": {"type": "string"},
        "commodities_elements": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        },
        "commodities_combined": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        }
    },
    "required": [
        "major_category",
        "minor_category",
        "commodities_elements",
        "commodities_combined"
    ]
}

# Build a schema snippet
papers_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Index": {"type": "integer"},
            "Year": {"type": "string"},
            "Month": {"type": "string"},
            "Journal": {"type": "string"},
            "Journal Impact factor": {"type": "string"},
            "Methodology": {"type": "string"},
            "Title": {"type": "string"},
            "Author": {"type": "string"},
            "Study Country": {"type": "string"},
            "Study Province/Region": {"type": "string"},
            "URL of the papers": {"type": "string"},
            "Availability of original data": {"type": "string"},
            "URL of the data server": {"type": "string"},
            "Commodity": commodity_obj_schema,
            # optional Commodity 2..n – allow as additional string props
        },
        "required": [
            "Index","Year","Month","Journal","Journal Impact factor",
            "Methodology","Title","Author","Study Country","Study Province/Region","URL of the papers",
            "Availability of original data","URL of the data server","Commodity"
        ],
        # allow Commodity 2..n, each must be the same object shape
        "patternProperties": {
            r"^Commodity( [2-9]| [1-9][0-9])$": commodity_obj_schema
        }
    }
}

# --- CONFIG ---
# Build a strict system message that hard-scopes the subject and forbids off-topic items.
system_msg = {
    "role": "system",
    "content": (
        "You are an expert geoscientist and computational scientist.\n"
        "TASK SCOPE (hard constraints):\n"
        "• Topic MUST be mineral deposit prospectivity/potential MAPPING or PREDICTION or MODELING using machine learning / artificial intelligence, "
        "  including data-driven geological targeting or mineral systems modeling.\n"
        "• EXCLUDE unrelated ML topics (e.g., landslides, hydrology, climate, hazards, remote sensing for non-mineral targets, seismic explosion monitoring, etc.).\n"
        "• Publication window: 2022–2025 inclusive.\n"
        "• Prioritize: (a) peer-reviewed journals with higher impact factors, and (b) reports from top geological surveys or institutes (e.g., USGS, NRCan/Geological Survey of Canada, Geoscience Australia, BGS).\n"
        "• If an item does not clearly match the scope above, DO NOT include it.\n\n"
        "COMMODITY RULES (authoritative):\n"
        "you MUST return a JSON OBJECT copied from the rules with EXACT keys "
        "major_category, minor_category, commodities_elements, commodities_combined (no extra keys, no strings). "
        "If you cannot map a paper to a commodity in the rules, OMIT that paper entirely.\n"
        f"{rules_str}\n\n"
        "OUTPUT RULES:\n"
        "• Return ONLY a JSON array (no prose/markdown/code fences). If nothing qualifies, return [].\n"
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

# --- Request: use non-reasoning model + structured outputs + academic search + date filters ---
url = "https://api.perplexity.ai/chat/completions"
headers = {
    "Authorization": f"Bearer {PPLX_API_KEY}",
    "Content-Type": "application/json",
    }
data = {
    "model": PPLX_MODEL,
    "messages": messages,
    "temperature": 0.05,
    "top_p": 0.9,
    "search_mode": "academic",     # documented param
    "search_after_date_filter": "01/01/2022",
    "search_before_date_filter": "09/17/2025",
    "response_format": {
        "type": "json_schema",
        "json_schema": { "schema": papers_schema },
    }
}
# # Optional: bias toward reputable domains
if PPLX_SEARCH_FILTER:
    new={"search_domain_filter": [PPLX_SEARCH_FILTER]}
    data.update(new)

# --- REAL CALL
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
    # TODO.
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
    raw_path = OUT_DIR / f"perplexity_raw_{ts}.json"
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
