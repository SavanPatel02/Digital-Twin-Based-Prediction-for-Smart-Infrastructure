"""
Train all three models in sequence.

Each model runs in its own subprocess to avoid TensorFlow / scikit-learn
memory conflicts that cause segfaults when loaded in the same process.

Run from the project root:
    python train_all.py
"""

import sys
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

# Windows terminal UTF-8 fix
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_script(label: str, script: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, script)],
        env=env,
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    print("\n" + "=" * 60)
    print("  DIGITAL TWIN — Training Pipeline")
    print("=" * 60)

    run_script("[1/3] Isolation Forest — Anomaly Detection",
               os.path.join("models", "train_isolation_forest.py"))

    run_script("[2/3] LSTM — Failure Prediction",
               os.path.join("models", "train_lstm.py"))

    run_script("[3/3] RUL Regressor — Remaining Useful Life",
               os.path.join("models", "train_rul.py"))

    print("\n" + "=" * 60)
    print("  All models trained and saved to models/")
    print("  Run:  python -m streamlit run dashboard/app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
