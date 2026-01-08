# ============================================================
# README.txt
# ============================================================
# Project: Portfolio Part II (Streamlit) — Optimization Dashboard
#
# GOAL
# - Upload ONE Excel file (template-required)
# - Compute μ (annual expected return) + Σ (annualized covariance
# - Show current portfolio analytics (return/vol/sharpe + risk contribution)
# - Run optimization and export weights/trades to CSV
#
# ============================================================

# ============================================================
# STEP 0 — Reproducibility checklist
# ============================================================
# A) Files expected in your repo/folder:
#   1) app.py                      <-- your Streamlit app
#   2) FE.png (optional)            <-- logo (same folder as app.py)
#   3) requirements.txt (recommended for deploy)
#   4) runtime.txt  (recommended for deploy)
#   5) README.txt                   <-- this file
#
# B) Python version:
#   - Works best with Python 3.10+
#
# C) Required Python packages:
#   - streamlit, numpy, pandas, scipy, openpyxl, matplotlib
#   - (matplotlib is important on Streamlit Cloud because pandas .style gradients often rely on it)
# ============================================================

# ============================================================
# STEP 1 — Prepare your Excel file (TEMPLATE RULES — DO NOT SKIP)
# ============================================================
# You MUST upload Input.xlsx workbook that matches the template format.
# The app requires EXACT sheet names and certain columns/locations.
#
# REQUIRED SHEETS (exact names):
#   1) "Current Portfolio"
#   2) "Input"
#   3) "Close Price Data"
#
# ------------------------------------------------------------
# 1A) Sheet: "Current Portfolio"
# Required columns (exact column names):
#   - "Ticker"
#   - "Current Allocations"
#
# Example:
#   Ticker | Current Allocations
#   SGOV   | 0.05
#   HYG    | 0.10
#
# ------------------------------------------------------------
# 1B) Sheet: "Input"  (μ loader is strict because it matches Part I)
# The app loads μ (expected returns) using:
#   pd.read_excel(sheet_name="Input", usecols="B,K", skiprows=4)
#
# That means your template MUST have:
# - Column B: Ticker
# - Column K: Annualized Expected Return (in % units, e.g., 8.5 means 8.5%)
# - The first 4 rows are skipped (so data must start after that)
#
# Notes:
# - Keep the same row as the provided "Input" template.
#
# ------------------------------------------------------------
# 1C) Sheet: "Close Price Data"  (Σ loader is strict)
# Required structure:
# - One date column (preferred name: "Date" or "Datetime"; otherwise the first column is treated as date)
# - Remaining columns are tickers (prices)
# - All prices must be numeric (avoid "-" or text; blanks become NaN)
#
# Σ computation:
# - log returns: log(price_t / price_{t-1})
# - annualized covariance: cov(daily_log_returns) * 252
#
# Notes:
# - Missing / non-numeric values can create NaNs in Σ and break risk contribution.
# - If a ticker has too many missing prices, you should remove it or fix the data.
# ============================================================


# ============================================================
# STEP 2 — Run Streamlit locally
# ============================================================
# Open Visual Studio Code terminal and go to the folder containing app.py:
#
#   cd /path/to/your/project
#
# 2A) Create a virtual environment (recommended):
#   python -m venv .venv
#
# 2B) Activate it:
#   Mac/Linux:
#     source .venv/bin/activate
#   Windows PowerShell:
#     .venv\Scripts\activate
#
# 2C) Install packages:
#   pip install streamlit
#   pip install -r requirements.txt
#   pip install streamlit numpy pandas scipy openpyxl matplotlib
#
# 2D) Start the app:
#   streamlit run app.py
#
# 2E) Open the local URL shown in terminal:
#   Usually http://localhost:8501
# ============================================================


# ============================================================
# STEP 3 — Use the dashboard (what to click, in order)
# ============================================================
# 3A) Upload
# - Left sidebar → "1) Data Input" → Upload Excel (.xlsx)
# - If you do not upload, the app stops with an instruction message.
#
# 3B) Preview the workbook (sanity check)
# - Sidebar shows all sheet names
# - Use "Select a sheet to preview" to confirm:
#   - The required sheets exist
#   - Column names and formats are correct
#
# 3C) Confirm current portfolio snapshot
# - The page shows:
#   - Return (Ann.), Vol (Ann.), Sharpe
# - Two tabs:
#   - Weights (current)
#   - Risk Contribution (current)
#
# 3D) Choose Optimization Settings (sidebar)
# - Risk-free rate (annual): affects Sharpe display only
# - Risk aversion (λ): used in mean-variance objective utility
# - Objective:
#   - "Mean-Variance"
#   - "Min Variance (extra)"         (implemented by setting μ=0 in MV solver)
#   - "Target Return (extra)"        (min variance subject to μ >= target)
#
# 3E) Choose Optimization Approach (sidebar)
# - A) Theoretical Optimal:
#     Full optimization under policy constraints + max weight cap
# - B) Incremental Replacement:
#     Adds per-asset constraint: |w - w_current| <= max_active_weight
# - C) Risk Budget:
#     Uses penalty terms to discourage risk concentration:
#       - Max RC share cap (soft penalty)
#       - Top-3 RC share cap (soft penalty)
#
# 3F) Click Run
# - Sidebar button: "Run Optimization"
# - The app will compute optimized weights and show results.
#
# 3G) Download outputs
# - trades.csv: current vs optimized weights, Δ weight
# - weights.csv: optimized weights only
# ============================================================


# ============================================================
# STEP 4 — Streamlit Cloud deploy
# ============================================================
# If you do not deploy, other people usually cannot access it via a public URL.
# Local "streamlit run" is only on your machine.
#
# 4A) Put these files in your GitHub repo:
#   - app.py
#   - README.txt
#   - requirements.txt 
#   - runtime.txt 
#   - FE.png (optional)
#
# 4B) Create requirements.txt:
#   streamlit
#   numpy
#   pandas
#   scipy
#   openpyxl
#   matplotlib
#
# 4C) On Streamlit Cloud:
# - Create / deploy app from GitHub
# - Select repo + branch
# - Set main file path = app.py
#
# ============================================================


# ============================================================
# STEP 5 — Common errors (what it means + how to fix)
# ============================================================
# ERROR A) Missing required sheets
# Message like:
#   "Missing required sheets for Part I alignment: {...}"
# Fix:
# - Ensure sheet names match exactly:
#   "Current Portfolio", "Input", "Close Price Data"
#
# ERROR B) Current Portfolio missing required columns
# Message like:
#   "Current Portfolio sheet must contain columns: 'Ticker' and 'Current Allocations'"
# Fix:
# - Rename columns to match exactly
#
# ERROR C) μ missing for tickers in universe
# Message like:
#   "μ missing for tickers in universe: [...]"
# Why:
# - Your Close Price Data contains tickers not found in Input (μ loader)
# Fix:
# - Add those tickers to Input (template columns B + K, respecting skiprows=4), OR
# - Remove those ticker columns from Close Price Data
#
# ERROR D) Covariance matrix contains NaNs
# Message like:
#   "Covariance matrix contains NaNs..."
# Why:
# - Missing/non-numeric price values created NaNs in log returns covariance
# Fix:
# - Make sure Close Price Data prices are numeric
# - Remove columns/tickers with insufficient history
# - Avoid placeholders like "-" in price cells
#
# ERROR E) TypeError around scalar conversion (common on Cloud)
# Example:
#   TypeError at: float(w.T @ cov_m @ w)
# Why:
# - Some operations return (1,1) arrays instead of scalars on some environments
# Fix pattern:
# - Always convert 1x1 to scalar with .item() or to_scalar()
# - (Your code already applies this fix in key places)
# ============================================================


# ============================================================
# END OF README.txt
# ============================================================
