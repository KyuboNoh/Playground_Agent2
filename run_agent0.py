#!/usr/bin/env python3
"""Generate the Agent3 prospectivity report from stored outputs."""

# python run_agent3.py --openai-model gpt-5-nano --openai-api-key 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
from __future__ import annotations

import argparse
import os
from pathlib import Path

from Agents.agent3 import run_agent3_report

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent3 report generation")
    parser.add_argument("--openai-api-key", dest="openai_api_key", default=None, help="Override OPENAI_API_KEY for this run")
    parser.add_argument(
        "--openai-model",
        dest="openai_model",
        default=None,
        help="Model name to use (defaults to OPENAI_AGENT3_MODEL or OPENAI_MODEL)",
    )
    parser.add_argument(
        "--context-dir",
        dest="context_dir",
        default="outputs/context",
        help="Directory containing context JSON files (default: outputs/context)",
    )
    parser.add_argument(
        "--predictions-dir",
        dest="predictions_dir",
        default="outputs/predictions",
        help="Directory containing prediction outputs (default: outputs/predictions)",
    )
    parser.add_argument(
        "--report-dir",
        dest="report_dir",
        default="outputs/reports",
        help="Directory to store generated PDF reports (default: outputs/reports)",
    )
    parser.add_argument(
        "--max-words",
        dest="max_words",
        type=int,
        default=1000,
        help="Maximum number of words requested from the agent (<=1000)",
    )
    parser.add_argument(
        "--print-context",
        dest="print_context",
        action="store_true",
        help="Print the textual context that will be sent to the agent",
    )
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["AGENT3_OPENAI_API_KEY"] = args.openai_api_key

    if args.openai_model:
        os.environ["AGENT3_OPENAI_MODEL"] = args.openai_model

    limit = max(1, min(args.max_words, 5000))

    result = run_agent3_report(
        model=args.openai_model,
        context_dir=Path(args.context_dir),
        predictions_dir=Path(args.predictions_dir),
        report_dir=Path(args.report_dir),
        max_words=limit,
    )

    pdf_path = Path(result.get("pdf_path", ""))
    print(f"Report saved to: {pdf_path}")
    if result.get("used_fallback"):
        print("⚠️ Fallback summary was used.")
    if result.get("error"):
        print(f"Info: {result['error']}")

    if args.print_context:
        context = result.get("context_summary")
        if context:
            print("\n--- Agent3 context ---\n")
            print(context)

    report_text = result.get("report")
    if report_text:
        print("\n--- Report text ---\n")
        print(report_text)


if __name__ == "__main__":
    main()
