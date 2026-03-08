#!/usr/bin/env bash
# ============================================================================
# .sh script developed by Anthropic's Claude AI.
#
# Reproduce all experiments from:
#   "Two-Timescale Conformal Prediction Under Distribution Shift"
#
# Usage:
#   chmod +x reproduce.sh && ./reproduce.sh
#
# Total runtime: ~5 minutes on a modern CPU (no GPU required)
# ============================================================================

set -e

echo "============================================================"
echo "  Two-Timescale Conformal Prediction — Full Reproduction"
echo "============================================================"
echo ""

# --- Environment setup ---
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] Virtual environment already exists."
fi

source venv/bin/activate
echo "[2/5] Installing dependencies..."
pip install -q -r requirements.txt

# --- Synthetic experiments (Experiments A–E) ---
echo ""
echo "[3/5] Running synthetic experiments (A–E)..."
python conformal_experiments_v3.py
echo "  -> Figures saved to figures/synthetic/"

# --- Regret bound verification (Theorem 1) ---
echo ""
echo "[4/5] Running theorem verification..."
python theory_regret_bound.py
echo "  -> Figure saved to figures/theory/"

# --- Real-data validation (Beijing PM2.5 + Jena Climate) ---
echo ""
echo "[5/5] Running real-data validation..."
echo "  (Jena Climate auto-downloads on first run)"
python run_real_data.py
echo "  -> Figures saved to figures/real_data/"

echo ""
echo "============================================================"
echo "  Done. All figures are in figures/."
echo "  Paper source: paper/paper_draft.tex"
echo "============================================================"
