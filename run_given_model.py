#!/usr/bin/env python3
"""Run a given prediction method using stored Step 2 outputs.

This is a lightweight helper for developers to quickly test any prediction
method located under the ``Methods`` directory.  It loads the
``outputs/context/step2.json`` configuration (and associated CSV files) and
invokes the method's ``run`` function.  The results are written to
``outputs/predictions/<method_name>``.

Example
-------
Run the 2D baseline:

    python run_given_model.py --method 2D/2D_prediction_baseline
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import inspect
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from Agents.agent2.agent2_main import call_agent2


def _load_method(method: str):
    """Dynamically import a prediction method from the Methods folder."""
    # --- Resolve base directory of the script and Methods dir ---
    this_file = Path(__file__).resolve()
    project_root = this_file.parent
    methods_dir = project_root / "Methods"

    # --- Normalize the incoming string (whitespace, unicode, slashes) ---
    m = unicodedata.normalize("NFKC", str(method)).strip()
    m = m.replace("\\", "/").strip("/")  # tolerate backslashes and leading/trailing slashes

    # --- Ensure .py extension for relative names ---
    if not m.lower().endswith(".py"):
        m_with_ext = m + ".py"
    else:
        m_with_ext = m

    candidates: list[Path] = []

    # 1) Absolute path?
    p_abs = Path(m_with_ext)
    if p_abs.is_absolute():
        candidates.append(p_abs)

    # 2) Direct relative path under Methods/
    p_rel = methods_dir / m_with_ext
    candidates.append(p_rel)

    # --- Pick the first existing candidate, or fall back to a case-insensitive search ---
    full_path: Path | None = next((p for p in candidates if p.exists()), None)

    if full_path is None:
        # Case-insensitive recursive search under Methods/
        target_rel_lower = m_with_ext.lower()
        target_name_lower = Path(m_with_ext).name.lower()
        target_stem_lower = Path(m_with_ext).stem.lower()

        matches: list[Path] = []
        if methods_dir.exists():
            for p in methods_dir.rglob("*.py"):
                rel_str_lower = str(p.relative_to(methods_dir)).replace("\\", "/").lower()
                if (
                    rel_str_lower == target_rel_lower
                    or p.name.lower() == target_name_lower
                    or p.stem.lower() == target_stem_lower
                ):
                    matches.append(p)

        if len(matches) == 1:
            full_path = matches[0]
        elif len(matches) > 1:
            # Ambiguous — ask the user to disambiguate by exact relative path
            opts = "\n  - " + "\n  - ".join(str(p.relative_to(methods_dir)) for p in matches[:10])
            raise FileNotFoundError(
                f"Ambiguous method name '{method}'. Multiple matches under {methods_dir}:\n{opts}\n"
                f"Please pass a more specific path (relative to Methods/) or an absolute path."
            )
        else:
            # Nothing found — build helpful suggestions from available files
            if methods_dir.exists():
                available = [str(p.relative_to(methods_dir)) for p in methods_dir.rglob("*.py")]
                suggestions = get_close_matches(m_with_ext, available, n=5, cutoff=0.4)
                msg_sug = ("\nDid you mean:\n  - " + "\n  - ".join(suggestions)) if suggestions else ""
            else:
                available, msg_sug = [], ""

            debug_info = (
                f"\n[Debug]\n__file__ = {this_file}\nproject_root = {project_root}\nmethods_dir = {methods_dir}\n"
                f"tried:\n  - {p_abs}\n  - {p_rel}\n"
            )
            raise FileNotFoundError(
                f"Method file not found for '{method}'. Looked under {methods_dir} and as absolute path."
                f"{msg_sug}{debug_info}"
            )

    # --- Import module from the resolved path ---
    # Create a mostly-unique module name to avoid collisions in repeated runs
    mod_name = "given_method_" + full_path.stem
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load method module: {full_path}")
    module = importlib.util.module_from_spec(spec)
    # Attach a helpful attribute for later debugging
    module.__dict__["__loaded_from__"] = str(full_path)
    spec.loader.exec_module(module)  # type: ignore[misc]

    if not hasattr(module, "run"):
        raise AttributeError(f"Method module lacks a 'run' function: {full_path}")

    return module


def run(method: str, cfg: str, data_csv: str | None, label_csv: str | None) -> None:
    """Run the specified prediction method."""

    msg, data_path, label_path, features = call_agent2(cfg, data_csv, label_csv, call_llm=False)
    print(msg)

    # here KN - read coordinate column names from the configuration
    with open(cfg, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    x_col = cfg_json["preprocess"]["x_col"]
    y_col = cfg_json["preprocess"]["y_col"]

    module = _load_method(method)

    run_kwargs = {}
    sig = inspect.signature(module.run)
    if "x_col" in sig.parameters:
        run_kwargs["x_col"] = x_col
    if "y_col" in sig.parameters:
        run_kwargs["y_col"] = y_col
    run_kwargs["shap_analysis"] = True

    result = module.run(data_path, label_path, features, **run_kwargs)
    # some of result are saved internally.

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a prediction method using stored outputs")
    parser.add_argument("--method", required=True, help="Path to method under Methods, e.g. Dim2/2D_prediction_baseline")
    parser.add_argument("--cfg", default="outputs/context/step2.json", help="Path to step2.json")
    parser.add_argument("--data_csv", default=None, help="Override data CSV path")
    parser.add_argument("--label_csv", default=None, help="Override label CSV path")
    args = parser.parse_args()

    run(args.method, args.cfg, args.data_csv, args.label_csv)

if __name__ == "__main__":
    main()
