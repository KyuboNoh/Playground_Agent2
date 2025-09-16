import os, re
import json
import time
import pathlib
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional


# --- CONFIG ---
# SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")          # set this
# OPENAI_API_KEY             = os.environ.get("OPENAI_API_KEY")                   # set this

# Current file path
here = Path(__file__)
env_api_path = here.parent / ".env.api"

if not (os.getenv("SEMANTIC_SCHOLAR_API_KEY")):
    load_dotenv(env_api_path, override=True)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Ask for the fields we’ll use:
S2_FIELDS = ",".join([
    "title","abstract","year","venue","url",
    "openAccessPdf","isOpenAccess","authors.name","externalIds"
])

SAVE_DIR = pathlib.Path("papers_out")
PDF_DIR = SAVE_DIR / "pdfs"
SAVE_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Make a filesystem-safe filename from a paper title."""
    safe = re.sub(r"[^\w\s\-\(\)\[\]\.\,]+", "", name).strip()
    safe = re.sub(r"\s+", " ", safe)
    return safe[:max_len] or "paper"

def s2_search(query: str, limit: int = 20, retry: int = 3, pause: float = 0.25) -> List[Dict[str, Any]]:
    params = {"query": query, "limit": limit, "fields": S2_FIELDS}
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    for attempt in range(1, retry+1):
        try:
            r = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            if attempt == retry:
                raise
            time.sleep(pause * attempt)
    return []

def try_download_pdf(url: str, outpath: pathlib.Path, timeout: int = 60) -> Optional[str]:
    """Download PDF; return local path on success, else None."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            # Basic content-type check (best-effort)
            ctype = r.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                # Some OA links redirect to HTML landing pages
                return None
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(1024 * 64):
                    if chunk:
                        f.write(chunk)
        return str(outpath)
    except Exception:
        return None

def harvest(query: str, n_results: int = 20) -> List[Dict[str, Any]]:
    items = s2_search(query, limit=n_results)
    rows = []
    for i, p in enumerate(items, 1):
        title   = p.get("title") or ""
        year    = p.get("year")
        venue   = p.get("venue") or ""
        url     = p.get("url") or ""  # publisher/landing page
        abstract = (p.get("abstract") or "").strip()

        oa_pdf_url = None
        if p.get("openAccessPdf") and p["openAccessPdf"].get("url"):
            oa_pdf_url = p["openAccessPdf"]["url"]

        local_pdf = None
        if oa_pdf_url:
            fname = sanitize_filename(title) + ".pdf"
            local_pdf = try_download_pdf(oa_pdf_url, PDF_DIR / fname)

        rows.append({
            "title": title,
            "year": year,
            "venue": venue,
            "url": url,
            "abstract": abstract if not local_pdf else abstract,  # always keep abstract if present
            "is_open_access": bool(p.get("isOpenAccess")),
            "oa_pdf_url": oa_pdf_url or "",
            "local_pdf": local_pdf or "",  # empty string if not downloaded
            "authors": ", ".join(a.get("name","") for a in (p.get("authors") or [])[:12]),
            "externalIds": p.get("externalIds") or {},  # doi/arxivId/etc.
        })
        time.sleep(0.2)  # polite pacing

    # Save JSON + CSV
    (SAVE_DIR / "metadata.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(SAVE_DIR / "metadata.csv", index=False)
    print(f"Saved {len(rows)} records to {SAVE_DIR/'metadata.csv'}")
    print(f"PDFs (if downloaded) saved under: {PDF_DIR}")
    return rows

if __name__ == "__main__":
    # Example: replace with your [given prompt]
    query = "Mineral prospectivity modeling using machine learning, since 2022, prioritize open-access papers."
    harvest(query, n_results=25)
