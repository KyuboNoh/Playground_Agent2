# AgenticMPM (modular)

Run UI (needs SSL and `pip install gradio`):
  python agentic_mpm/app.py --ui

Run headless:
  python agentic_mpm/app.py --features your.csv --x-col X --y-col Y --feature-cols a,b,c --label-source data --label-col label --label-mode pos_only

Save config in UI (outputs/config.json), then reuse in headless with:
  python agentic_mpm/app.py --config outputs/config.json

Development helper:
  python run_given_model.py --method Dim2/2D_prediction_baseline
