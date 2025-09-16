from __future__ import annotations
import os, argparse, json
import pandas as pd

from utils import has_ssl, ensure_dir
from pipeline.orchestrator import Orchestrator


def run_headless(args):
    ensure_dir("outputs")

    fbytes = open(args.features, "rb").read() if args.features else None
    lbytes = open(args.labels, "rb").read() if args.labels else None

    feat_cols = [c.strip() for c in args.feature_cols.split(',') if c.strip()] if args.feature_cols else None

    preprocess = None
    if args.config and os.path.exists(args.config):
        preprocess = json.load(open(args.config))
        if "preprocess" in preprocess:
            preprocess = preprocess["preprocess"]

    orch = Orchestrator()
    out = orch.process(
        deposit=args.deposit,
        data_file=fbytes,
        labels_file=lbytes,
        algo=args.algo,
        k=args.k,
        x_col=args.x_col or None,
        y_col=args.y_col or None,
        feature_cols=feat_cols,
        label_source=args.label_source,
        label_col=args.label_col or None,
        label_mode=args.label_mode,
        preprocess=preprocess,
    )

    out["targets"].to_csv("outputs/top_targets.csv", index=False)
    pd.DataFrame(out["review"]["curve"]).to_csv("outputs/discovery_curve.csv", index=False)
    json.dump(out["metrics"], open("outputs/metrics.json", "w"), indent=2)

    if out.get("heatmap_png"):
        open("outputs/heatmap.png", "wb").write(out["heatmap_png"])
    open("outputs/agent_logs.txt", "w").write("\n".join(f"[{m['name']}] {m['content']}" for m in out["logs"]))
    print("Headless run complete. See ./outputs")


def run_tests():
    from pipeline.orchestrator import Orchestrator
    import pandas as pd

    o = Orchestrator()
    # basic synthetic
    res = o.process("Porphyry Cu", None, None, "rf", 10)
    assert res["targets"].shape[0] == 10

    # data labels posneg
    df = pd.DataFrame({"lon":[0,1,2,3,4], "lat":[0,0,0,0,0], "f1":[1,2,3,4,5], "lab":[1,0,1,0,1]})
    res2 = o.process("Porphyry Cu", df.to_csv(index=False).encode(), None, "rf", 3, x_col="lon", y_col="lat", feature_cols=["f1"], label_source="data", label_col="lab", label_mode="posneg")
    assert res2["targets"].shape[0] == 3

    # separate labels pos_only
    grid = pd.DataFrame([{ "X":i, "Y":j, "f":i+j } for i in range(10) for j in range(10)])
    labs = pd.DataFrame({"X":[1,2,3], "Y":[1,2,3]})
    res3 = o.process("Porphyry Cu", grid.to_csv(index=False).encode(), labs.to_csv(index=False).encode(), "rf", 5, x_col="X", y_col="Y", feature_cols=["f"], label_source="separate", label_mode="pos_only")
    assert res3["targets"].shape[0] == 5
    print("All tests passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser("AgenticMPM")
    p.add_argument("--ui", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--features", type=str, default="")
    p.add_argument("--labels", type=str, default="")
    p.add_argument("--x-col", type=str, default="")
    p.add_argument("--y-col", type=str, default="")
    p.add_argument("--feature-cols", type=str, default="")
    p.add_argument("--label-source", choices=["none","data","separate"], default="none")
    p.add_argument("--label-col", type=str, default="")
    p.add_argument("--label-mode", choices=["pos_only","posneg"], default="pos_only")
    p.add_argument("--deposit", type=str, default="Porphyry Cu")
    p.add_argument("--algo", choices=["xgb","rf"], default="xgb")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--config", type=str, default="")
    args = p.parse_args()

    if args.test:
        run_tests(); raise SystemExit(0)

    if args.ui and has_ssl():
        from ui_main import launch_ui
        launch_ui(Orchestrator())
    elif args.ui and not has_ssl():
        print("SSL not available; run without --ui or install OpenSSL.")
    else:
        run_headless(args)
