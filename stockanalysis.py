"""
stock_analysis.py
A practical, end-to-end stock analysis report generator for any ticker.

Generates:
- Price/returns metrics (CAGR, vol, max drawdown, Sharpe, beta vs benchmark)
- Technical indicators (SMA/EMA, RSI, MACD, Bollinger)
- Fundamentals snapshot (yfinance info + basic financial statement trends)
- Simple FCF-based DCF valuation (with sensible defaults + fallbacks)
- Markdown report + PNG charts

Usage:
  pip install yfinance pandas numpy matplotlib
  python stock_analysis.py AAPL --period 5y --benchmark SPY
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf


# ----------------------------
# Utilities
# ----------------------------

def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def annualize_vol(daily_std: float, trading_days: int = 252) -> float:
    return daily_std * math.sqrt(trading_days)


def annualize_return_from_daily(daily_mean: float, trading_days: int = 252) -> float:
    # Approx compounding
    return (1.0 + daily_mean) ** trading_days - 1.0


def max_drawdown(series: pd.Series) -> float:
    # series is price or equity curve
    roll_max = series.cummax()
    dd = (series / roll_max) - 1.0
    return float(dd.min())


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> float:
    # Convert annual rf to daily
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = daily_returns - rf_daily
    if excess.std(ddof=0) == 0:
        return np.nan
    return float((excess.mean() / excess.std(ddof=0)) * math.sqrt(trading_days))


def sortino_ratio(daily_returns: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> float:
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    denom = downside.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float((excess.mean() / denom) * math.sqrt(trading_days))


def beta_vs_benchmark(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < 50:
        return np.nan
    a = df.iloc[:, 0].values
    b = df.iloc[:, 1].values
    var_b = np.var(b)
    if var_b == 0:
        return np.nan
    cov = np.cov(a, b, ddof=0)[0, 1]
    return float(cov / var_b)


# ----------------------------
# Technical Indicators
# ----------------------------

def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ema = pd.Series(up, index=close.index).ewm(span=period, adjust=False).mean()
    down_ema = pd.Series(down, index=close.index).ewm(span=period, adjust=False).mean()
    rs = up_ema / (down_ema + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    std = close.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ----------------------------
# Fundamentals + Simple DCF
# ----------------------------

@dataclass
class DCFResult:
    intrinsic_equity_value: float
    intrinsic_value_per_share: float
    current_price: float
    upside_pct: float
    used_fcf: float
    assumptions: dict


def get_last_valid(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    return float(series.iloc[0])


def estimate_fcf_from_cashflow(cf: pd.DataFrame) -> float:
    """
    yfinance cashflow often has rows like:
    - Total Cash From Operating Activities
    - Capital Expenditures
    We'll compute FCF = CFO + CapEx (CapEx is negative usually).
    """
    if cf is None or cf.empty:
        return np.nan

    # Normalize row names
    idx = [str(i).strip().lower() for i in cf.index]
    cf2 = cf.copy()
    cf2.index = idx

    # yfinance usually returns columns as periods, newest first
    cfo = None
    capex = None

    for key in ["total cash from operating activities", "operating cash flow", "cashflow from operations"]:
        if key in cf2.index:
            cfo = cf2.loc[key]
            break

    for key in ["capital expenditures", "capital expenditure"]:
        if key in cf2.index:
            capex = cf2.loc[key]
            break

    if cfo is None or capex is None:
        return np.nan

    # Take the most recent column
    cfo_latest = get_last_valid(cfo)
    capex_latest = get_last_valid(capex)

    if np.isnan(cfo_latest) or np.isnan(capex_latest):
        return np.nan

    return float(cfo_latest + capex_latest)


def simple_dcf(
    current_price: float,
    shares_outstanding: float,
    fcf: float,
    growth_5y: float = 0.08,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    years: int = 5
) -> DCFResult:
    """
    Very simple equity DCF from FCF:
      PV = sum( FCF_t / (1+r)^t ) + TerminalValue / (1+r)^years
      TerminalValue = FCF_{y} * (1+g) / (r-g)

    Notes:
    - This treats FCF as to equity (not perfect). For a cleaner model you’d use FCFF and EV then adjust net debt.
    - Still useful for quick sanity checks / scenario ranges.
    """
    assumptions = {
        "growth_5y": growth_5y,
        "discount_rate": discount_rate,
        "terminal_growth": terminal_growth,
        "years": years
    }

    if any(np.isnan(x) for x in [current_price, shares_outstanding, fcf]) or shares_outstanding <= 0:
        return DCFResult(np.nan, np.nan, current_price, np.nan, fcf, assumptions)

    # Project FCF
    fcfs = []
    fcf_t = fcf
    for _ in range(years):
        fcf_t *= (1 + growth_5y)
        fcfs.append(fcf_t)

    # PV of projected FCFs
    pv_fcfs = sum(fcfs[t] / ((1 + discount_rate) ** (t + 1)) for t in range(years))

    # Terminal value
    if discount_rate <= terminal_growth:
        terminal_value = np.nan
    else:
        terminal_fcf = fcfs[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)

    pv_terminal = terminal_value / ((1 + discount_rate) ** years) if not np.isnan(terminal_value) else np.nan

    intrinsic_equity = pv_fcfs + pv_terminal
    intrinsic_per_share = intrinsic_equity / shares_outstanding if shares_outstanding else np.nan
    upside_pct = (intrinsic_per_share / current_price - 1) * 100 if current_price and not np.isnan(intrinsic_per_share) else np.nan

    return DCFResult(
        intrinsic_equity_value=float(intrinsic_equity) if not np.isnan(intrinsic_equity) else np.nan,
        intrinsic_value_per_share=float(intrinsic_per_share) if not np.isnan(intrinsic_per_share) else np.nan,
        current_price=float(current_price),
        upside_pct=float(upside_pct) if not np.isnan(upside_pct) else np.nan,
        used_fcf=float(fcf),
        assumptions=assumptions
    )


# ----------------------------
# Plotting
# ----------------------------

def plot_price_with_indicators(df: pd.DataFrame, outpath: str, ticker: str) -> None:
    close = df["Close"]
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    upper, mid, lower = bollinger(close, 20, 2.0)

    plt.figure(figsize=(12, 6))
    plt.plot(close.index, close.values, label="Close")
    plt.plot(sma50.index, sma50.values, label="SMA 50")
    plt.plot(sma200.index, sma200.values, label="SMA 200")
    plt.plot(upper.index, upper.values, label="BB Upper (20,2)")
    plt.plot(lower.index, lower.values, label="BB Lower (20,2)")
    plt.title(f"{ticker} Price + Trend Indicators")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_rsi(df: pd.DataFrame, outpath: str, ticker: str) -> None:
    r = rsi(df["Close"], 14)
    plt.figure(figsize=(12, 4))
    plt.plot(r.index, r.values, label="RSI (14)")
    plt.axhline(70, linestyle="--")
    plt.axhline(30, linestyle="--")
    plt.title(f"{ticker} RSI (14)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_macd(df: pd.DataFrame, outpath: str, ticker: str) -> None:
    macd_line, signal_line, hist = macd(df["Close"])
    plt.figure(figsize=(12, 4))
    plt.plot(macd_line.index, macd_line.values, label="MACD")
    plt.plot(signal_line.index, signal_line.values, label="Signal")
    plt.plot(hist.index, hist.values, label="Hist")
    plt.title(f"{ticker} MACD (12,26,9)")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_drawdown(df: pd.DataFrame, outpath: str, ticker: str) -> None:
    close = df["Close"].dropna()
    roll_max = close.cummax()
    dd = (close / roll_max) - 1.0
    plt.figure(figsize=(12, 4))
    plt.plot(dd.index, dd.values, label="Drawdown")
    plt.title(f"{ticker} Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ----------------------------
# Main analysis
# ----------------------------

def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=period, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No price data found for ticker: {ticker}")
    return df


def compute_perf_metrics(price_df: pd.DataFrame, rf: float, benchmark_df: Optional[pd.DataFrame] = None) -> dict:
    close = price_df["Close"].dropna()
    daily_ret = close.pct_change().dropna()

    metrics = {}
    metrics["start_price"] = float(close.iloc[0])
    metrics["end_price"] = float(close.iloc[-1])
    metrics["total_return_pct"] = float((close.iloc[-1] / close.iloc[0] - 1) * 100)

    # Time-based CAGR
    days = (close.index[-1] - close.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    if years and years > 0:
        metrics["cagr_pct"] = float(((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100)
    else:
        metrics["cagr_pct"] = np.nan

    metrics["ann_vol_pct"] = float(annualize_vol(daily_ret.std(ddof=0)) * 100)
    metrics["sharpe"] = sharpe_ratio(daily_ret, rf_annual=rf)
    metrics["sortino"] = sortino_ratio(daily_ret, rf_annual=rf)
    metrics["max_drawdown_pct"] = float(max_drawdown(close) * 100)

    if benchmark_df is not None and not benchmark_df.empty:
        b_close = benchmark_df["Close"].dropna()
        b_ret = b_close.pct_change().dropna()
        # Align
        beta = beta_vs_benchmark(daily_ret, b_ret)
        metrics["beta_vs_benchmark"] = beta
    else:
        metrics["beta_vs_benchmark"] = np.nan

    return metrics


def fundamentals_snapshot(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}

    snap = {
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "marketCap": safe_float(info.get("marketCap")),
        "currentPrice": safe_float(info.get("currentPrice")),
        "previousClose": safe_float(info.get("previousClose")),
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "priceToSalesTrailing12Months": safe_float(info.get("priceToSalesTrailing12Months")),
        "profitMargins": safe_float(info.get("profitMargins")),
        "operatingMargins": safe_float(info.get("operatingMargins")),
        "grossMargins": safe_float(info.get("grossMargins")),
        "returnOnEquity": safe_float(info.get("returnOnEquity")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "totalCash": safe_float(info.get("totalCash")),
        "totalDebt": safe_float(info.get("totalDebt")),
        "freeCashflow": safe_float(info.get("freeCashflow")),
        "sharesOutstanding": safe_float(info.get("sharesOutstanding")),
        "beta": safe_float(info.get("beta")),
        "dividendYield": safe_float(info.get("dividendYield")),
        "52w_high": safe_float(info.get("fiftyTwoWeekHigh")),
        "52w_low": safe_float(info.get("fiftyTwoWeekLow")),
    }

    # Financial statements (best-effort)
    try:
        fin = t.financials  # income statement (annual) columns newest first
        cf = t.cashflow
        bs = t.balance_sheet
    except Exception:
        fin, cf, bs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    snap["_income_stmt"] = fin
    snap["_cashflow"] = cf
    snap["_balance_sheet"] = bs

    return snap


def summarize_financial_trends(fin: pd.DataFrame) -> dict:
    """
    Extract simple trends from yfinance income statement.
    """
    out = {}
    if fin is None or fin.empty:
        return out

    # Normalize index
    idx = [str(i).strip().lower() for i in fin.index]
    fin2 = fin.copy()
    fin2.index = idx

    # Helper to fetch a row
    def row(*names: str) -> Optional[pd.Series]:
        for n in names:
            n2 = n.lower()
            if n2 in fin2.index:
                return fin2.loc[n2]
        return None

    revenue = row("Total Revenue", "total revenue", "revenue")
    op_income = row("Operating Income", "operating income")
    net_income = row("Net Income", "net income")

    # Columns are usually newest first
    def growth(series: pd.Series) -> Optional[float]:
        s = series.dropna()
        if len(s) < 2:
            return None
        newest = float(s.iloc[0])
        oldest = float(s.iloc[-1])
        if oldest == 0:
            return None
        # CAGR over number of years-1 intervals
        n_years = len(s) - 1
        return (newest / oldest) ** (1 / n_years) - 1

    if revenue is not None:
        g = growth(revenue)
        out["revenue_cagr_est"] = g
        out["revenue_latest"] = get_last_valid(revenue)

    if op_income is not None and revenue is not None:
        # latest margin
        opm = get_last_valid(op_income) / (get_last_valid(revenue) + 1e-12)
        out["op_margin_latest_est"] = opm

    if net_income is not None and revenue is not None:
        npm = get_last_valid(net_income) / (get_last_valid(revenue) + 1e-12)
        out["net_margin_latest_est"] = npm

    return out


def make_markdown_report(
    ticker: str,
    snap: dict,
    perf: dict,
    tech_last: dict,
    dcf: Optional[DCFResult],
    charts: dict,
    outpath: str
) -> None:
    def fmt_money(x):
        if x is None or np.isnan(x):
            return "N/A"
        # Compact formatting
        absx = abs(x)
        if absx >= 1e12:
            return f"{x/1e12:.2f}T"
        if absx >= 1e9:
            return f"{x/1e9:.2f}B"
        if absx >= 1e6:
            return f"{x/1e6:.2f}M"
        return f"{x:,.0f}"

    def fmt_pct(x):
        return "N/A" if x is None or np.isnan(x) else f"{x*100:.2f}%"

    def fmt_num(x, nd=2):
        return "N/A" if x is None or np.isnan(x) else f"{x:.{nd}f}"

    name = snap.get("longName") or ticker
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"# Stock Analysis Report: {name} ({ticker})")
    lines.append(f"*Generated: {now}*")
    lines.append("")
    lines.append("## 1) Company Snapshot")
    lines.append(f"- Sector / Industry: {snap.get('sector','N/A')} / {snap.get('industry','N/A')}")
    lines.append(f"- Country: {snap.get('country','N/A')}")
    lines.append(f"- Market Cap: {fmt_money(snap.get('marketCap'))} {snap.get('currency','')}")
    lines.append(f"- Price: {fmt_num(snap.get('currentPrice'))}  |  52W Range: {fmt_num(snap.get('52w_low'))} – {fmt_num(snap.get('52w_high'))}")
    lines.append("")

    lines.append("## 2) Valuation Multiples (Quick View)")
    lines.append(f"- Trailing P/E: {fmt_num(snap.get('trailingPE'))}")
    lines.append(f"- Forward P/E: {fmt_num(snap.get('forwardPE'))}")
    lines.append(f"- Price/Sales (TTM): {fmt_num(snap.get('priceToSalesTrailing12Months'))}")
    lines.append("")

    lines.append("## 3) Profitability & Balance Sheet (Quick View)")
    lines.append(f"- Gross Margin: {fmt_pct(snap.get('grossMargins'))}")
    lines.append(f"- Operating Margin: {fmt_pct(snap.get('operatingMargins'))}")
    lines.append(f"- Net Margin: {fmt_pct(snap.get('profitMargins'))}")
    lines.append(f"- ROE: {fmt_pct(snap.get('returnOnEquity'))}")
    lines.append(f"- Debt/Equity: {fmt_num(snap.get('debtToEquity'))}")
    lines.append(f"- Total Cash: {fmt_money(snap.get('totalCash'))} | Total Debt: {fmt_money(snap.get('totalDebt'))}")
    lines.append("")

    lines.append("## 4) Performance & Risk (from price history)")
    lines.append(f"- Start Price: {fmt_num(perf.get('start_price'))} → End Price: {fmt_num(perf.get('end_price'))}")
    lines.append(f"- Total Return: {fmt_num(perf.get('total_return_pct'))}%")
    lines.append(f"- CAGR (est): {fmt_num(perf.get('cagr_pct'))}%")
    lines.append(f"- Annualized Volatility: {fmt_num(perf.get('ann_vol_pct'))}%")
    lines.append(f"- Max Drawdown: {fmt_num(perf.get('max_drawdown_pct'))}%")
    lines.append(f"- Sharpe (rf adj): {fmt_num(perf.get('sharpe'))} | Sortino: {fmt_num(perf.get('sortino'))}")
    lines.append(f"- Beta vs Benchmark: {fmt_num(perf.get('beta_vs_benchmark'))}")
    lines.append("")

    lines.append("## 5) Technical Indicators (latest)")
    lines.append(f"- SMA 50: {fmt_num(tech_last.get('sma50'))} | SMA 200: {fmt_num(tech_last.get('sma200'))}")
    lines.append(f"- RSI (14): {fmt_num(tech_last.get('rsi14'))}  *(>70 overbought, <30 oversold as a rough rule)*")
    lines.append(f"- MACD: {fmt_num(tech_last.get('macd'))} | Signal: {fmt_num(tech_last.get('macd_signal'))}")
    lines.append("")

    lines.append("## 6) Simple DCF (FCF-based sanity check)")
    if dcf is None or np.isnan(dcf.intrinsic_value_per_share):
        lines.append("- DCF could not be computed (missing FCF or shares data).")
    else:
        lines.append(f"- Current Price: {dcf.current_price:.2f}")
        lines.append(f"- FCF used (most recent, est): {fmt_money(dcf.used_fcf)}")
        lines.append(f"- Intrinsic Value / Share (est): {dcf.intrinsic_value_per_share:.2f}")
        lines.append(f"- Upside / Downside (est): {dcf.upside_pct:.1f}%")
        lines.append(f"- Assumptions: growth={dcf.assumptions['growth_5y']*100:.1f}%, discount={dcf.assumptions['discount_rate']*100:.1f}%, terminal={dcf.assumptions['terminal_growth']*100:.1f}%")
    lines.append("")

    lines.append("## 7) Charts")
    for k, v in charts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is an automated, best-effort report using Yahoo Finance data via `yfinance`.")
    lines.append("- Not financial advice. Always validate numbers with official filings and consider scenario ranges.")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate a detailed stock analysis report for any ticker.")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., AAPL, MSFT, TSLA, SPY, BTC-USD)")
    parser.add_argument("--period", type=str, default="5y", help="Price history period (e.g., 1y, 2y, 5y, 10y, max)")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker for beta (default: SPY)")
    parser.add_argument("--rf", type=float, default=0.03, help="Annual risk-free rate for Sharpe/Sortino (default: 0.03)")
    parser.add_argument("--dcf_growth", type=float, default=None, help="Override 5y FCF growth (e.g., 0.08 for 8%)")
    parser.add_argument("--dcf_discount", type=float, default=0.10, help="DCF discount rate (default: 0.10)")
    parser.add_argument("--dcf_terminal", type=float, default=0.025, help="DCF terminal growth (default: 0.025)")
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()
    benchmark = args.benchmark.upper().strip()

    # Output folders
    today = dt.datetime.utcnow().strftime("%Y%m%d")
    report_dir = os.path.join("reports", ticker)
    chart_dir = os.path.join("charts", ticker)
    ensure_dir(report_dir)
    ensure_dir(chart_dir)

    # Fetch data
    price_df = fetch_history(ticker, period=args.period)
    bench_df = None
    try:
        bench_df = fetch_history(benchmark, period=args.period)
    except Exception:
        bench_df = None

    # Compute metrics
    perf = compute_perf_metrics(price_df, rf=args.rf, benchmark_df=bench_df)

    # Technicals (latest)
    close = price_df["Close"].dropna()
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)

    tech_last = {
        "sma50": safe_float(sma50.iloc[-1]) if len(sma50.dropna()) else np.nan,
        "sma200": safe_float(sma200.iloc[-1]) if len(sma200.dropna()) else np.nan,
        "rsi14": safe_float(rsi14.iloc[-1]) if len(rsi14.dropna()) else np.nan,
        "macd": safe_float(macd_line.iloc[-1]) if len(macd_line.dropna()) else np.nan,
        "macd_signal": safe_float(signal_line.iloc[-1]) if len(signal_line.dropna()) else np.nan,
    }

    # Fundamentals + DCF
    snap = fundamentals_snapshot(ticker)
    fin_trends = summarize_financial_trends(snap.get("_income_stmt"))
    # Decide growth assumption: user override > revenue CAGR estimate > default 8%
    growth_est = fin_trends.get("revenue_cagr_est", None)
    if args.dcf_growth is not None:
        dcf_growth = args.dcf_growth
    elif growth_est is not None and not np.isnan(growth_est):
        # clamp to a reasonable range for stability
        dcf_growth = float(np.clip(growth_est, -0.05, 0.20))
    else:
        dcf_growth = 0.08

    # FCF estimate
    fcf_info = safe_float(snap.get("freeCashflow"))
    fcf_cf = estimate_fcf_from_cashflow(snap.get("_cashflow"))
    fcf_used = fcf_info if not np.isnan(fcf_info) else fcf_cf

    current_price = safe_float(snap.get("currentPrice"))
    shares = safe_float(snap.get("sharesOutstanding"))
    dcf_res = simple_dcf(
        current_price=current_price,
        shares_outstanding=shares,
        fcf=fcf_used,
        growth_5y=dcf_growth,
        discount_rate=args.dcf_discount,
        terminal_growth=args.dcf_terminal,
        years=5
    )

    # Charts
    charts = {}
    p1 = os.path.join(chart_dir, f"{ticker}_{today}_price.png")
    p2 = os.path.join(chart_dir, f"{ticker}_{today}_rsi.png")
    p3 = os.path.join(chart_dir, f"{ticker}_{today}_macd.png")
    p4 = os.path.join(chart_dir, f"{ticker}_{today}_drawdown.png")

    plot_price_with_indicators(price_df, p1, ticker)
    plot_rsi(price_df, p2, ticker)
    plot_macd(price_df, p3, ticker)
    plot_drawdown(price_df, p4, ticker)

    charts["Price + SMA + Bollinger"] = p1
    charts["RSI"] = p2
    charts["MACD"] = p3
    charts["Drawdown"] = p4

    # Report
    report_path = os.path.join(report_dir, f"{ticker}_{today}_report.md")
    make_markdown_report(
        ticker=ticker,
        snap=snap,
        perf=perf,
        tech_last=tech_last,
        dcf=dcf_res,
        charts=charts,
        outpath=report_path
    )

    print(f"\n✅ Report saved: {report_path}")
    print(f"✅ Charts saved in: {chart_dir}\n")


if __name__ == "__main__":
    main()
