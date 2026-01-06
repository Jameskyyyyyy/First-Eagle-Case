import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from pathlib import Path
from typing import Any, Tuple

# =========================
# Page config
# =========================
st.set_page_config(page_title="Portfolio Part II (Streamlit)", layout="wide")

# =========================
# Header with logo (center title)
# =========================
logo_path = Path(__file__).parent / "FE.png"

col_left, col_center, col_right = st.columns([1, 4, 1], vertical_alignment="center")

with col_center:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>Portfolio Analysis</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #6b7280; margin-top: 4px;'>"
        "Interactive optimization & analytics dashboard"
        "</p>",
        unsafe_allow_html=True
    )

with col_right:
    if logo_path.exists():
        st.image(str(logo_path), width=300)

st.markdown("---")

# =========================
# Helpers: ticker normalize
# =========================
def norm_ticker_index(idx) -> pd.Index:
    return pd.Index(idx).astype(str).str.strip().str.upper()

def norm_series_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = norm_ticker_index(s.index)
    return s

def norm_cov(cov: pd.DataFrame) -> pd.DataFrame:
    cov = cov.copy()
    cov.index = norm_ticker_index(cov.index)
    cov.columns = norm_ticker_index(cov.columns)
    return cov

# =========================
# Formatting helpers
# =========================
def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.2f}%"

def fmt_float(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.6f}"

def style_weights_table(df: pd.DataFrame, highlight_top_n: int = 5):
    """
    df columns (all in % units):
    - Current Weight (%)
    - Optimized Weight (%)
    - Δ Weight (%)
    - |Δ| (%)
    """
    df2 = df.copy()

    # mark top movers by |Δ|
    if "|Δ| (%)" in df2.columns:
        top_idx = df2["|Δ| (%)"].nlargest(highlight_top_n).index
    else:
        top_idx = []

    def highlight_rows(row):
        if row.name in top_idx:
            return ["background-color: rgba(255, 215, 0, 0.22)"] * len(row)
        return [""] * len(row)

    # red/green for delta; stronger for bigger moves
    sty = (
        df2.style
        .apply(highlight_rows, axis=1)
        .format({
            "Current Weight (%)": fmt_pct,
            "Optimized Weight (%)": fmt_pct,
            "Δ Weight (%)": fmt_pct,
            "|Δ| (%)": fmt_pct
        })
    )

    # background gradient on |Δ|
    if "|Δ| (%)" in df2.columns:
        sty = sty.background_gradient(subset=["|Δ| (%)"], cmap="Blues")

    # subtle red/green on Δ
    if "Δ Weight (%)" in df2.columns:
        sty = sty.background_gradient(subset=["Δ Weight (%)"], cmap="RdYlGn")

    return sty

def make_weights_comparison_table(
    w_current: pd.Series,
    w_opt: pd.Series,
    top_n_highlight: int = 5
) -> Tuple[pd.DataFrame, Any]:

    """
    Return (df, styler) where weights are in % and rounded to 2 decimals.
    """
    w_current = norm_series_index(w_current)
    w_opt = norm_series_index(w_opt)

    idx = w_opt.index.union(w_current.index)
    w_cur = w_current.reindex(idx).fillna(0.0)
    w_o = w_opt.reindex(idx).fillna(0.0)

    df = pd.DataFrame({
        "Current Weight (%)": w_cur * 100.0,
        "Optimized Weight (%)": w_o * 100.0,
    })
    df["Δ Weight (%)"] = df["Optimized Weight (%)"] - df["Current Weight (%)"]
    df["|Δ| (%)"] = df["Δ Weight (%)"].abs()

    # investment-friendly sorting: biggest optimized weights first
    df = df.sort_values("Optimized Weight (%)", ascending=False)

    # round numeric storage
    df = df.round(2)

    sty = style_weights_table(df, highlight_top_n=top_n_highlight)
    return df, sty

def style_current_weights_table(w_current: pd.Series):
    df = (w_current * 100.0).to_frame("Current Weight (%)").round(2)
    return (
        df.style
        .format({"Current Weight (%)": fmt_pct})
        .background_gradient(subset=["Current Weight (%)"], cmap="Blues")
    )

def style_rc_table(rc_tbl: pd.DataFrame):
    """
    Make risk contribution view more 'investment' friendly:
    - weight and RC Share in %
    - other columns keep numeric
    """
    df = rc_tbl.copy()
    df["weight (%)"] = (df["weight"] * 100.0).round(2)
    df["RC Share (%)"] = (df["RC Share"] * 100.0).round(2)
    df = df.drop(columns=["weight", "RC Share"])

    # reorder
    cols = ["weight (%)", "RC Share (%)", "MRC (to var)", "RC (to var)"]
    df = df[[c for c in cols if c in df.columns]].round(6)

    sty = (
        df.style
        .format({
            "weight (%)": fmt_pct,
            "RC Share (%)": fmt_pct,
            "MRC (to var)": fmt_float,
            "RC (to var)": fmt_float,
        })
        .background_gradient(subset=["RC Share (%)"], cmap="Blues")
    )
    return df, sty

# =========================
# Part I-aligned portfolio_stats (DO NOT CHANGE)
# =========================
def portfolio_stats(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> dict:
    weights = weights.copy()
    weights.index = weights.index.astype(str).str.strip().str.upper()

    mu = mu.copy()
    mu.index = mu.index.astype(str).str.strip().str.upper()

    cov = cov.copy()
    cov.index = cov.index.astype(str).str.strip().str.upper()
    cov.columns = cov.columns.astype(str).str.strip().str.upper()

    tickers = cov.columns

    w = weights.reindex(tickers).fillna(0.0).values
    mu_v = mu.reindex(tickers).fillna(0.0).values
    cov_m = cov.values

    port_ret = float(w @ mu_v)
    port_var = float(w @ cov_m @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan

    return {"return": port_ret, "vol": port_vol, "variance": port_var, "sharpe": sharpe}

# =========================
# Risk contribution (aligned to cov.columns)
# =========================
def risk_contribution_table(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    weights = norm_series_index(weights)
    cov = norm_cov(cov)

    tickers = list(cov.columns)

    # 1) Force numeric covariance (DataFrame level)
    cov_num = cov.apply(pd.to_numeric, errors="coerce")

    # 2) Force numpy float64 arrays (this is the key)
    cov_m = np.asarray(cov_num.values, dtype=np.float64)
    w = np.asarray(
        weights.reindex(tickers).fillna(0.0).astype(float).values,
        dtype=np.float64
    ).reshape(-1, 1)

    # 3) Debug info (shows on the app page; helps even when Cloud redacts errors)
    st.caption(
        f"[DEBUG RC] w shape={w.shape}, w dtype={w.dtype} | "
        f"cov shape={cov_m.shape}, cov dtype={cov_m.dtype} | "
        f"cov NaNs={int(np.isnan(cov_m).sum())}"
    )

    # 4) If Σ has NaNs, raise a clear error instead of failing later
    if np.isnan(cov_m).any():
        raise ValueError(
            "Covariance matrix contains NaNs. This usually happens when Close Price Data has missing/non-numeric "
            "prices for one or more tickers. Please clean the input or remove tickers with insufficient history."
        )

    sigma_w = cov_m @ w
    port_var = float((w.T @ cov_m @ w).item())  # .item() avoids dtype surprises
    port_var = max(port_var, 1e-18)

    mrc = sigma_w.flatten()
    rc = (w.flatten() * mrc)
    rc_share = rc / port_var

    out = pd.DataFrame(
        {"weight": w.flatten(), "MRC (to var)": mrc, "RC (to var)": rc, "RC Share": rc_share},
        index=tickers
    ).sort_values("RC Share", ascending=False)

    return out


def compute_rebalance_metrics(w_current: pd.Series, w_opt: pd.Series, trade_threshold: float = 0.005) -> dict:
    """
    trade_threshold: in weight units (0.005 = 0.5%)
    """
    w_current = norm_series_index(w_current)
    w_opt = norm_series_index(w_opt)

    idx = sorted(set(w_current.index) | set(w_opt.index))
    wc = w_current.reindex(idx).fillna(0.0)
    wo = w_opt.reindex(idx).fillna(0.0)

    dw = wo - wc
    abs_dw = dw.abs()

    turnover = float(abs_dw.sum() / 2.0)  # standard definition
    n_trades = int((abs_dw > trade_threshold).sum())
    max_change = float(abs_dw.max()) if len(abs_dw) else 0.0

    adds = int((dw > trade_threshold).sum())
    trims = int((dw < -trade_threshold).sum())

    return {
        "turnover": turnover,
        "n_trades": n_trades,
        "max_change": max_change,
        "adds": adds,
        "trims": trims,
        "threshold": trade_threshold,
    }

def compute_before_after_risk_summary(
    w_current: pd.Series,
    w_opt: pd.Series,
    cov_annual: pd.DataFrame,
    illiquid_tickers: tuple[str, ...],
    equity_tickers: tuple[str, ...],
) -> pd.DataFrame:
    """
    Returns a compact table for PM-style comparison.
    """
    w_current = norm_series_index(w_current).reindex(cov_annual.columns).fillna(0.0)
    w_opt = norm_series_index(w_opt).reindex(cov_annual.columns).fillna(0.0)
    cov_annual = norm_cov(cov_annual)

    rc_cur = risk_contribution_table(w_current, cov_annual)
    rc_opt = risk_contribution_table(w_opt, cov_annual)

    def top3(rc_tbl): return float(rc_tbl["RC Share"].head(3).sum())
    def maxrc(rc_tbl): return float(rc_tbl["RC Share"].max())

    cur_top3, opt_top3 = top3(rc_cur), top3(rc_opt)
    cur_max, opt_max = maxrc(rc_cur), maxrc(rc_opt)

    cur_ill = float(w_current.reindex(list(illiquid_tickers)).fillna(0.0).sum())
    opt_ill = float(w_opt.reindex(list(illiquid_tickers)).fillna(0.0).sum())

    cur_eq = float(w_current.reindex(list(equity_tickers)).fillna(0.0).sum())
    opt_eq = float(w_opt.reindex(list(equity_tickers)).fillna(0.0).sum())

    cur_sgov = float(w_current.get("SGOV", 0.0))
    opt_sgov = float(w_opt.get("SGOV", 0.0))

    out = pd.DataFrame(
        {
            "Current": [cur_top3, cur_max, cur_sgov, cur_eq, cur_ill],
            "Optimized": [opt_top3, opt_max, opt_sgov, opt_eq, opt_ill],
        },
        index=[
            "Top-3 Risk Contribution Share",
            "Max Risk Contribution Share",
            "SGOV Weight",
            "Equity Bucket Weight",
            "Illiquid Bucket Weight",
        ],
    )
    out["Δ"] = out["Optimized"] - out["Current"]
    return out

def make_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


# =========================
# Part I μ loader (Input sheet: usecols="B,K", skiprows=4; return_annualized / 100)
# =========================
def load_mu_part1(uploaded) -> tuple[pd.Series, str]:
    xls = pd.ExcelFile(uploaded)
    if "Input" not in xls.sheet_names:
        raise ValueError("Input sheet not found.")

    df = pd.read_excel(uploaded, sheet_name="Input", usecols="B,K", skiprows=4)
    df.columns = ["Ticker", "Return_Annualized"]
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    mu = df.set_index("Ticker")["Return_Annualized"].astype(float) / 100.0  # IMPORTANT
    mu = norm_series_index(mu)

    return mu, "Input!B,K (skiprows=4) Return_Annualized / 100 (Part I-aligned)"

# =========================
# Part I Σ loader (Close Price Data: log returns cov * 252)
# =========================
def load_cov_part1(uploaded) -> tuple[pd.DataFrame, str, dict]:
    xls = pd.ExcelFile(uploaded)
    if "Close Price Data" not in xls.sheet_names:
        raise ValueError("Close Price Data sheet not found (Part I uses it to compute Σ).")

    prices = pd.read_excel(uploaded, sheet_name="Close Price Data")

    # assume first col is Date
    date_col = None
    for c in prices.columns:
        if str(c).strip().lower() in ["date", "datetime"]:
            date_col = c
            break
    if date_col is None:
        date_col = prices.columns[0]

    prices[date_col] = pd.to_datetime(prices[date_col])
    prices = prices.sort_values(date_col).set_index(date_col)

    # numeric + normalize tickers
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices.columns = norm_ticker_index(prices.columns)

    # meta info
    start_date = prices.index.min()
    end_date = prices.index.max()
    n_rows = int(prices.shape[0])
    n_cols = int(prices.shape[1])
    missing_ratio = float(prices.isna().sum().sum() / (prices.size + 1e-18))

    # each ticker valid count
    valid_counts = prices.notna().sum().sort_values(ascending=False)
    min_valid = int(valid_counts.min()) if len(valid_counts) else 0
    max_valid = int(valid_counts.max()) if len(valid_counts) else 0

    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")

    trading_days = 252
    cov_annual = log_returns.cov() * trading_days
    cov_annual = norm_cov(cov_annual)

    meta = {
        "date_col": str(date_col),
        "start_date": start_date,
        "end_date": end_date,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_ratio": missing_ratio,
        "valid_counts": valid_counts,   # pd.Series
        "min_valid": min_valid,
        "max_valid": max_valid,
    }

    return cov_annual, "Close Price Data → log_returns.cov()*252 (Part I-aligned)", meta


# =========================
# Part I policy constraints helper
# =========================
def add_policy_constraints(
    constraints: list,
    tickers: list[str],
    sgov_min: float, sgov_max: float,
    fehix_max: float, hyg_max: float, fegix_max: float,
    illiquid_max: float, equity_max: float,
    illiquid_tickers: tuple[str, ...],
    equity_tickers: tuple[str, ...],
):
    if "SGOV" in tickers:
        idx = tickers.index("SGOV")
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx: w[idx] - sgov_min})
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx: sgov_max - w[idx]})

    if "FEHIX" in tickers:
        idx = tickers.index("FEHIX")
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx: fehix_max - w[idx]})
    if "HYG" in tickers:
        idx = tickers.index("HYG")
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx: hyg_max - w[idx]})
    if "FEGIX" in tickers:
        idx = tickers.index("FEGIX")
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx: fegix_max - w[idx]})

    ill_idx = [tickers.index(t) for t in illiquid_tickers if t in tickers]
    if len(ill_idx) > 0:
        constraints.append({"type": "ineq", "fun": lambda w, ii=ill_idx: illiquid_max - np.sum(w[ii])})

    eq_idx = [tickers.index(t) for t in equity_tickers if t in tickers]
    if len(eq_idx) > 0:
        constraints.append({"type": "ineq", "fun": lambda w, ei=eq_idx: equity_max - np.sum(w[ei])})

# =========================
# Part I-aligned solvers
# =========================
def solve_part1_theoretical(
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    w_current: pd.Series,
    risk_aversion: float,
    max_asset_weight: float,
    sgov_min: float, sgov_max: float,
    fehix_max: float, hyg_max: float, fegix_max: float,
    illiquid_max: float, equity_max: float,
    illiquid_tickers: tuple[str, ...],
    equity_tickers: tuple[str, ...],
    verbose: bool = False,
) -> pd.Series:
    mu_annual = norm_series_index(mu_annual)
    cov_annual = norm_cov(cov_annual)
    w_current = norm_series_index(w_current)

    tickers = list(cov_annual.columns)
    cov = cov_annual.loc[tickers, tickers].astype(float)
    mu = mu_annual.reindex(tickers).astype(float)
    if mu.isna().any():
        missing = mu[mu.isna()].index.tolist()
        raise ValueError(f"Missing expected returns for tickers: {missing}")

    n = len(tickers)
    w0 = w_current.reindex(tickers).fillna(0.0).astype(float).values
    cov_m = cov.values
    mu_v = mu.values

    def neg_utility(w):
        ret = float(w @ mu_v)
        var = float(w.T @ cov_m @ w)
        util = ret - 0.5 * risk_aversion * var
        return -util

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    add_policy_constraints(
        constraints, tickers,
        sgov_min, sgov_max,
        fehix_max, hyg_max, fegix_max,
        illiquid_max, equity_max,
        illiquid_tickers, equity_tickers
    )

    bounds = [(0.0, max_asset_weight) for _ in range(n)]
    x0 = np.clip(w0, 0.0, max_asset_weight)
    if x0.sum() <= 1e-12:
        x0 = np.ones(n) / n
    else:
        x0 = x0 / x0.sum()

    res = minimize(
        fun=neg_utility,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12, "disp": verbose},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = pd.Series(res.x, index=tickers)
    return w / w.sum()

def solve_part1_incremental(
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    w_current: pd.Series,
    risk_aversion: float,
    max_active_weight: float,
    max_asset_weight: float,
    sgov_min: float, sgov_max: float,
    fehix_max: float, hyg_max: float, fegix_max: float,
    illiquid_max: float, equity_max: float,
    illiquid_tickers: tuple[str, ...],
    equity_tickers: tuple[str, ...],
    verbose: bool = False,
) -> pd.Series:
    mu_annual = norm_series_index(mu_annual)
    cov_annual = norm_cov(cov_annual)
    w_current = norm_series_index(w_current)

    tickers = list(cov_annual.columns)
    cov = cov_annual.loc[tickers, tickers].astype(float)
    mu = mu_annual.reindex(tickers).astype(float)
    if mu.isna().any():
        missing = mu[mu.isna()].index.tolist()
        raise ValueError(f"Missing expected returns for tickers: {missing}")

    n = len(tickers)
    w0 = w_current.reindex(tickers).fillna(0.0).astype(float).values
    cov_m = cov.values
    mu_v = mu.values

    def neg_utility(w):
        ret = float(w @ mu_v)
        var = float(w.T @ cov_m @ w)
        util = ret - 0.5 * risk_aversion * var
        return -util

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # |w_i - w0_i| <= max_active_weight
    for i in range(n):
        constraints.append({"type": "ineq", "fun": lambda w, i=i: max_active_weight - (w[i] - w0[i])})
        constraints.append({"type": "ineq", "fun": lambda w, i=i: max_active_weight + (w[i] - w0[i])})

    add_policy_constraints(
        constraints, tickers,
        sgov_min, sgov_max,
        fehix_max, hyg_max, fegix_max,
        illiquid_max, equity_max,
        illiquid_tickers, equity_tickers
    )

    bounds = [(0.0, max_asset_weight) for _ in range(n)]
    x0 = np.clip(w0, 0.0, max_asset_weight)
    if x0.sum() <= 1e-12:
        x0 = np.ones(n) / n
    else:
        x0 = x0 / x0.sum()

    res = minimize(
        fun=neg_utility,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12, "disp": verbose},
    )
    if not res.success:
        raise RuntimeError(f"Incremental optimization failed: {res.message}")

    w = pd.Series(res.x, index=tickers)
    return w / w.sum()

def solve_part1_risk_budget(
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    w_current: pd.Series,
    risk_aversion: float,
    max_asset_weight: float,
    max_rc_share: float,
    top3_rc_cap: float,
    sgov_min: float, sgov_max: float,
    fehix_max: float, hyg_max: float, fegix_max: float,
    illiquid_max: float, equity_max: float,
    illiquid_tickers: tuple[str, ...],
    equity_tickers: tuple[str, ...],
    verbose: bool = False,
) -> pd.Series:
    mu_annual = norm_series_index(mu_annual)
    cov_annual = norm_cov(cov_annual)
    w_current = norm_series_index(w_current)

    tickers = list(cov_annual.columns)
    cov = cov_annual.loc[tickers, tickers].astype(float)
    mu = mu_annual.reindex(tickers).astype(float)
    if mu.isna().any():
        missing = mu[mu.isna()].index.tolist()
        raise ValueError(f"Missing expected returns for tickers: {missing}")

    n = len(tickers)
    w0 = w_current.reindex(tickers).fillna(0.0).astype(float).values
    cov_m = cov.values
    mu_v = mu.values

    def rc_shares(w):
        w = np.array(w).reshape(-1, 1)
        sigma_w = cov_m @ w
        pv = float(w.T @ cov_m @ w)
        pv = max(pv, 1e-18)
        mrc = sigma_w.flatten()
        rc = (w.flatten() * mrc)
        return rc / pv

    def neg_utility_with_penalty(w):
        ret = float(w @ mu_v)
        var = float(w.T @ cov_m @ w)
        util = ret - 0.5 * risk_aversion * var

        shares = rc_shares(w)
        p1 = 2000.0 * max(0.0, float(shares.max() - max_rc_share)) ** 2
        top3 = float(np.sort(shares)[-3:].sum())
        p2 = 2000.0 * max(0.0, top3 - top3_rc_cap) ** 2

        return -(util) + p1 + p2

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    add_policy_constraints(
        constraints, tickers,
        sgov_min, sgov_max,
        fehix_max, hyg_max, fegix_max,
        illiquid_max, equity_max,
        illiquid_tickers, equity_tickers
    )

    bounds = [(0.0, max_asset_weight) for _ in range(n)]
    x0 = np.clip(w0, 0.0, max_asset_weight)
    if x0.sum() <= 1e-12:
        x0 = np.ones(n) / n
    else:
        x0 = x0 / x0.sum()

    res = minimize(
        fun=neg_utility_with_penalty,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 4000, "ftol": 1e-12, "disp": verbose},
    )
    if not res.success:
        raise RuntimeError(f"Risk-budget optimization failed: {res.message}")

    w = pd.Series(res.x, index=tickers)
    return w / w.sum()

# =========================
# Constraints report (extended with policy constraints)
# =========================
def constraints_report(
    w_opt: pd.Series,
    w_current: pd.Series | None,
    approach: str,
    max_asset_weight: float,
    sgov_min: float, sgov_max: float,
    fehix_max: float, hyg_max: float, fegix_max: float,
    illiquid_max: float, equity_max: float,
    illiquid_tickers: tuple[str, ...],
    equity_tickers: tuple[str, ...],
    cov: pd.DataFrame,
    max_active: float | None,
    max_rc_share: float | None,
    top3_rc_cap: float | None,
):
    w_opt = norm_series_index(w_opt)
    cov = norm_cov(cov)
    tickers = list(cov.columns)

    rep = []
    rep.append(("Sum weights = 1", abs(float(w_opt.sum()) - 1.0) <= 1e-6, float(w_opt.sum())))
    rep.append(("Long-only (min >= 0)", float(w_opt.min()) >= -1e-10, float(w_opt.min())))
    rep.append(("Max asset weight cap", float(w_opt.max()) <= max_asset_weight + 1e-10, float(w_opt.max())))

    if "SGOV" in tickers:
        w_sgov = float(w_opt.get("SGOV", 0.0))
        rep.append(("SGOV min", w_sgov >= sgov_min - 1e-10, w_sgov))
        rep.append(("SGOV max", w_sgov <= sgov_max + 1e-10, w_sgov))

    for t, cap in [("FEHIX", fehix_max), ("HYG", hyg_max), ("FEGIX", fegix_max)]:
        if t in tickers:
            wt = float(w_opt.get(t, 0.0))
            rep.append((f"{t} max", wt <= cap + 1e-10, wt))

    ill_sum = float(w_opt.reindex(list(illiquid_tickers)).fillna(0.0).sum())
    eq_sum = float(w_opt.reindex(list(equity_tickers)).fillna(0.0).sum())
    rep.append(("Illiquid bucket cap", ill_sum <= illiquid_max + 1e-10, ill_sum))
    rep.append(("Equity bucket cap", eq_sum <= equity_max + 1e-10, eq_sum))

    if approach.startswith("B)") and w_current is not None and max_active is not None:
        w_current = norm_series_index(w_current).reindex(tickers).fillna(0.0)
        active = (w_opt.reindex(tickers).fillna(0.0) - w_current).abs()
        rep.append(("Max active per asset", float(active.max()) <= max_active + 1e-10, float(active.max())))

    if approach.startswith("C)") and (max_rc_share is not None) and (top3_rc_cap is not None):
        rc_tbl = risk_contribution_table(w_opt, cov)
        max_rc = float(rc_tbl["RC Share"].max())
        top3 = float(rc_tbl["RC Share"].head(3).sum())
        rep.append(("Max RC share cap", max_rc <= max_rc_share + 1e-10, max_rc))
        rep.append(("Top-3 RC share cap", top3 <= top3_rc_cap + 1e-10, top3))

    df = pd.DataFrame(rep, columns=["Constraint", "Pass", "Value"])
    # make Value nicer for display
    df["Value"] = df["Value"].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else x)
    return df

# =========================
# Sidebar: upload + preview
# =========================
st.sidebar.header("1) Data Input")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], key="file_uploader")

if uploaded is None:
    st.info("Please upload your Excel file first (.xlsx).")
    st.stop()

xls = pd.ExcelFile(uploaded)
st.sidebar.write("Sheets:", xls.sheet_names)

sheet = st.sidebar.selectbox("Select a sheet to preview", xls.sheet_names, key="sheet_preview_select")
df_preview = pd.read_excel(uploaded, sheet_name=sheet)

st.subheader(f"Preview: {sheet}")
st.dataframe(df_preview, use_container_width=True)

# =========================
# Sidebar: settings
# =========================
st.sidebar.header("2) Optimization Settings")

rf = st.sidebar.number_input(
    "Risk-free rate (annual)",
    value=0.00, step=0.005, format="%.3f", key="rf_input"
)

risk_aversion = st.sidebar.slider(
    "Risk aversion (λ)", 0.5, 20.0, 3.0, 0.5, key="risk_aversion_slider"
)

objective = st.sidebar.selectbox(
    "Objective",
    ["Mean-Variance", "Min Variance (extra)", "Target Return (extra)"],
    key="objective_select"
)

target_return = None
if objective == "Target Return (extra)":
    target_return = st.sidebar.number_input(
        "Target return (annual)", value=0.08, step=0.01, format="%.3f", key="target_return_input"
    )

st.sidebar.header("3) Optimization Approach")
approach = st.sidebar.selectbox(
    "Choose an approach",
    [
        "A) Theoretical Optimal",
        "B) Incremental Replacement (stay close to current)",
        "C) Risk Budget (cap risk contribution)"
    ],
    key="approach_select"
)

st.sidebar.subheader("Constraints (Global)")
max_asset_weight = st.sidebar.slider(
    "Max weight per asset (single-name cap)", 0.05, 1.00, 0.40, 0.01, key="max_asset_weight_slider"
)

st.sidebar.markdown("**Policy constraints**")
sgov_min = st.sidebar.slider("SGOV min", 0.0, 0.50, 0.02, 0.01, key="sgov_min_slider")
sgov_max = st.sidebar.slider("SGOV max", 0.0, 0.50, 0.10, 0.01, key="sgov_max_slider")
if sgov_max < sgov_min:
    st.sidebar.error("SGOV max must be >= SGOV min")

fehix_max = st.sidebar.slider("FEHIX max", 0.0, 1.00, 0.20, 0.01, key="fehix_max_slider")
hyg_max   = st.sidebar.slider("HYG max",   0.0, 1.00, 0.20, 0.01, key="hyg_max_slider")
fegix_max = st.sidebar.slider("FEGIX max", 0.0, 1.00, 0.20, 0.01, key="fegix_max_slider")

illiquid_max = st.sidebar.slider("Illiquid bucket max", 0.0, 1.00, 0.20, 0.01, key="illiquid_max_slider")
equity_max   = st.sidebar.slider("Equity bucket max",   0.0, 1.00, 0.60, 0.01, key="equity_max_slider")

illiquid_tickers = ("FECRX", "FERLX")
equity_tickers   = ("FEGIX", "SGOIX", "FEAIX", "FESCX", "FESMX")

# Approach-specific constraints (keep your structure)
max_active = turnover_cap = None
max_rc_share = top3_rc_cap = None

if approach.startswith("B)"):
    st.sidebar.subheader("Constraints (Incremental)")
    max_active = st.sidebar.slider("Max |w - w_current| per asset", 0.00, 0.50, 0.10, 0.01, key="max_active_slider")
    turnover_cap = st.sidebar.slider("Max turnover (sum |Δw|) [UI only]", 0.00, 2.00, 0.50, 0.01, key="turnover_cap_slider")

if approach.startswith("C)"):
    st.sidebar.subheader("Constraints (Risk Budget)")
    max_rc_share = st.sidebar.slider("Max RC Share per asset", 0.05, 1.00, 0.30, 0.01, key="max_rc_share_slider")
    top3_rc_cap = st.sidebar.slider("Max Top-3 RC Share", 0.10, 1.00, 0.70, 0.01, key="top3_rc_cap_slider")

highlight_top_n = 5

run = st.sidebar.button("Run Optimization", type="primary", key="run_button")

# =========================
# Load sheets: Current Portfolio (w_current) + μ + Σ (Part I-aligned)
# =========================
need = {"Current Portfolio", "Input", "Close Price Data"}
missing = need - set(xls.sheet_names)
if missing:
    st.error(f"Missing required sheets for Part I alignment: {missing}")
    st.stop()

cur = pd.read_excel(uploaded, sheet_name="Current Portfolio")
if "Ticker" not in cur.columns or "Current Allocations" not in cur.columns:
    st.error("Current Portfolio sheet must contain columns: 'Ticker' and 'Current Allocations'")
    st.stop()

cur["Ticker"] = cur["Ticker"].astype(str).str.strip().str.upper()
cur = cur[cur["Ticker"].notna() & (cur["Ticker"] != "") & (cur["Ticker"] != "NAN") & (cur["Ticker"] != "NONE")]

w_current = cur.set_index("Ticker")["Current Allocations"].astype(float)
w_current = w_current / w_current.sum()
w_current = norm_series_index(w_current)

mu_all, mu_src = load_mu_part1(uploaded)
cov_annual, cov_src, cov_meta = load_cov_part1(uploaded)

universe = list(cov_annual.columns)

w_current_u = w_current.reindex(universe).fillna(0.0)
w_current_u = w_current_u / w_current_u.sum()

mu_u = mu_all.reindex(universe)
if mu_u.isna().any():
    missing_mu = mu_u[mu_u.isna()].index.tolist()
    st.error(f"μ missing for tickers in universe: {missing_mu}. Fix Input sheet coverage.")
    st.stop()

# =========================
# Current snapshot
# =========================
st.subheader("Current Portfolio Snapshot")
st.caption(
    f"Universe: {len(universe)} tickers | Current holdings: {int((w_current_u > 0).sum())} | "
)

cur_stats = portfolio_stats(w_current_u, mu_u, cov_annual, rf=rf)
c1, c2, c3 = st.columns(3)
c1.metric("Return (Ann.)", f"{cur_stats['return']*100:.2f}%")
c2.metric("Vol (Ann.)", f"{cur_stats['vol']*100:.2f}%")
c3.metric("Sharpe (Rf)", f"{cur_stats['sharpe']:.2f}" if np.isfinite(cur_stats["sharpe"]) else "NA")

exp_ret_manual = (w_current_u * mu_u).sum()
st.caption(f"Manual check — Expected Return = {exp_ret_manual*100:.2f}% (should match above Return)")

tab_cur1, tab_cur2 = st.tabs(["Weights", "Risk Contribution"])

with tab_cur1:
    st.write("Current Weights (Universe)")
    st.dataframe(style_current_weights_table(w_current_u), use_container_width=True)

with tab_cur2:
    st.write("Current Risk Contribution (Universe)")
    rc_cur = risk_contribution_table(w_current_u, cov_annual)
    rc_cur_df, rc_cur_sty = style_rc_table(rc_cur)
    st.dataframe(rc_cur_sty, use_container_width=True)

# =========================
# Recommended Portfolio output
# =========================
st.divider()
st.subheader("Recommended Portfolio Output")

if not run:
    st.info("Adjust constraints in the sidebar, then click **Run Optimization**.")
    st.stop()

try:
    if objective == "Mean-Variance":
        if approach.startswith("A)"):
            w_opt = solve_part1_theoretical(
                mu_annual=mu_u,
                cov_annual=cov_annual,
                w_current=w_current_u,
                risk_aversion=risk_aversion,
                max_asset_weight=max_asset_weight,
                sgov_min=sgov_min, sgov_max=sgov_max,
                fehix_max=fehix_max, hyg_max=hyg_max, fegix_max=fegix_max,
                illiquid_max=illiquid_max, equity_max=equity_max,
                illiquid_tickers=illiquid_tickers,
                equity_tickers=equity_tickers,
                verbose=False,
            )
        elif approach.startswith("B)"):
            w_opt = solve_part1_incremental(
                mu_annual=mu_u,
                cov_annual=cov_annual,
                w_current=w_current_u,
                risk_aversion=risk_aversion,
                max_active_weight=max_active if max_active is not None else 0.10,
                max_asset_weight=max_asset_weight,
                sgov_min=sgov_min, sgov_max=sgov_max,
                fehix_max=fehix_max, hyg_max=hyg_max, fegix_max=fegix_max,
                illiquid_max=illiquid_max, equity_max=equity_max,
                illiquid_tickers=illiquid_tickers,
                equity_tickers=equity_tickers,
                verbose=False,
            )
        else:
            w_opt = solve_part1_risk_budget(
                mu_annual=mu_u,
                cov_annual=cov_annual,
                w_current=w_current_u,
                risk_aversion=risk_aversion,
                max_asset_weight=max_asset_weight,
                max_rc_share=max_rc_share if max_rc_share is not None else 0.30,
                top3_rc_cap=top3_rc_cap if top3_rc_cap is not None else 0.70,
                sgov_min=sgov_min, sgov_max=sgov_max,
                fehix_max=fehix_max, hyg_max=hyg_max, fegix_max=fegix_max,
                illiquid_max=illiquid_max, equity_max=equity_max,
                illiquid_tickers=illiquid_tickers,
                equity_tickers=equity_tickers,
                verbose=False,
            )

    elif objective == "Min Variance (extra)":
        mu_zero = mu_u * 0.0
        w_opt = solve_part1_theoretical(
            mu_annual=mu_zero,
            cov_annual=cov_annual,
            w_current=w_current_u,
            risk_aversion=1.0,
            max_asset_weight=max_asset_weight,
            sgov_min=sgov_min, sgov_max=sgov_max,
            fehix_max=fehix_max, hyg_max=hyg_max, fegix_max=fegix_max,
            illiquid_max=illiquid_max, equity_max=equity_max,
            illiquid_tickers=illiquid_tickers,
            equity_tickers=equity_tickers,
            verbose=False,
        )

    else:
        if target_return is None:
            raise ValueError("Target return is required.")

        tickers = list(cov_annual.columns)
        cov_m = cov_annual.values
        mu_v = mu_u.values
        n = len(tickers)

        def port_var(w): return float(w.T @ cov_m @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: float(w @ mu_v) - float(target_return)},
        ]

        add_policy_constraints(
            constraints, tickers,
            sgov_min, sgov_max,
            fehix_max, hyg_max, fegix_max,
            illiquid_max, equity_max,
            illiquid_tickers, equity_tickers
        )

        bounds = [(0.0, max_asset_weight) for _ in range(n)]
        x0 = np.clip(w_current_u.reindex(tickers).fillna(0.0).values, 0.0, max_asset_weight)
        x0 = (np.ones(n) / n) if x0.sum() <= 1e-12 else (x0 / x0.sum())

        res = minimize(
            fun=lambda w: port_var(w),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 4000, "ftol": 1e-12, "disp": False},
        )
        if not res.success:
            raise RuntimeError(f"Target-return optimization failed: {res.message}")

        w_opt = pd.Series(res.x, index=tickers)
        w_opt = w_opt / w_opt.sum()

except Exception as e:
    st.error(str(e))
    st.stop()

w_opt = norm_series_index(w_opt).reindex(universe).fillna(0.0)
w_opt = w_opt / w_opt.sum()

opt_stats = portfolio_stats(w_opt, mu_u, cov_annual, rf=rf)

m1, m2, m3 = st.columns(3)
m1.metric("Return (Ann.)", f"{opt_stats['return']*100:.2f}%")
m2.metric("Vol (Ann.)", f"{opt_stats['vol']*100:.2f}%")
m3.metric("Sharpe (Rf)", f"{opt_stats['sharpe']:.2f}" if np.isfinite(opt_stats["sharpe"]) else "NA")

# ---------- Rebalance Summary (PM-style) ----------
st.subheader("Rebalance Summary")

trade_threshold = 0.0001

reb = compute_rebalance_metrics(w_current_u, w_opt, trade_threshold=trade_threshold)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Turnover (Σ|Δw|/2)", f"{reb['turnover']:.2%}")
s2.metric("# Trades", f"{reb['n_trades']}")
s3.metric("Max |Δw|", f"{reb['max_change']:.2%}")
s4.metric("Adds / Trims", f"{reb['adds']} / {reb['trims']}")

# ---------- Downloads ----------
st.caption("Exports are designed for execution workflow (trades + weights).")

# Build trades table
idx = list(cov_annual.columns)
wc = w_current_u.reindex(idx).fillna(0.0)
wo = w_opt.reindex(idx).fillna(0.0)
trades_df = pd.DataFrame(
    {
        "Current Weight (%)": (wc * 100.0),
        "Optimized Weight (%)": (wo * 100.0),
        "Δ Weight (pp)": ((wo - wc) * 100.0),
        "|Δ| (pp)": ((wo - wc).abs() * 100.0),
    },
    index=idx
)

# Optional: sort by biggest trades
trades_sorted = trades_df.sort_values("|Δ| (pp)", ascending=False)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Download trades (CSV)",
        data=make_download_bytes(trades_sorted.round(2)),
        file_name="trades.csv",
        mime="text/csv",
        key="download_trades_btn"
    )
with d2:
    weights_df = pd.DataFrame({"Weight (%)": (wo * 100.0)}, index=idx)
    st.download_button(
        "Download weights (CSV)",
        data=make_download_bytes(weights_df.round(2)),
        file_name="weights.csv",
        mime="text/csv",
        key="download_weights_btn"
    )

# =========================
# Professional layout: tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Weights", "Risk Contribution", "Constraints Check"])

with tab1:
    st.write("Optimized Weights")
    w_df, w_sty = make_weights_comparison_table(w_current_u, w_opt, top_n_highlight=highlight_top_n)
    st.caption("Highlighted rows = largest absolute rebalances (by |Δw|). All weights shown in %.")
    st.dataframe(w_sty, use_container_width=True, height=520)

    # Optional quick summary: top trades
    top_trades = w_df.sort_values("|Δ| (%)", ascending=False).head(highlight_top_n)
    st.write(f"Top {highlight_top_n} Rebalances")
    st.dataframe(
        top_trades.style.format({
            "Current Weight (%)": fmt_pct,
            "Optimized Weight (%)": fmt_pct,
            "Δ Weight (%)": fmt_pct,
            "|Δ| (%)": fmt_pct
        }),
        use_container_width=True
    )

    # Bar chart: optimized weights (keep raw weights for chart)
    st.write("Optimized Weights (bar)")
    st.bar_chart(w_opt)

with tab2:
    st.write("Risk Contribution Share (Optimized)")
    rc_opt = risk_contribution_table(w_opt, cov_annual)
    rc_opt_df, rc_opt_sty = style_rc_table(rc_opt)
    st.dataframe(rc_opt_sty, use_container_width=True, height=520)

    st.write("Risk Contribution Share (bar)")
    st.bar_chart((rc_opt["RC Share"] * 100.0).round(2))  # show % on chart

with tab3:
    st.subheader("Constraints Check")
    rep_df = constraints_report(
        w_opt=w_opt,
        w_current=w_current_u,
        approach=approach,
        max_asset_weight=max_asset_weight,
        sgov_min=sgov_min, sgov_max=sgov_max,
        fehix_max=fehix_max, hyg_max=hyg_max, fegix_max=fegix_max,
        illiquid_max=illiquid_max, equity_max=equity_max,
        illiquid_tickers=illiquid_tickers,
        equity_tickers=equity_tickers,
        cov=cov_annual,
        max_active=max_active,
        max_rc_share=max_rc_share,
        top3_rc_cap=top3_rc_cap
    )

    st.dataframe(rep_df, use_container_width=True)
