"""
Cash Secured Put (CSP) Options Scanner
Finds optimal entry points with ~0.15 delta and max 40 DTE.
Focus on maximizing ROI while managing risk for a limited ticker universe.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Optional

st.set_page_config(
    page_title="CSP Options Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def parse_tickers(ticker_text: str) -> List[str]:
    raw = [s.strip().upper() for s in ticker_text.replace(",", " ").split() if s.strip()]
    return list(dict.fromkeys(raw))[:50]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    # Black-Scholes d1 and put delta formula
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float('nan')
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _normal_cdf(d1) - 1.0


@st.cache_data(ttl=900, show_spinner=False)
def fetch_option_metadata(ticker: str) -> Optional[Dict]:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="70d")
        if history.empty:
            return None

        last_price = float(history["Close"].iloc[-1])
        prev_close = float(history["Close"].iloc[-2]) if len(history) > 1 else last_price
        day_change_pct = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0

        window = history.tail(52)
        low_52 = float(window["Low"].min())
        high_52 = float(window["High"].max())
        range_52d = f"{low_52:.2f} - {high_52:.2f}"

        expirations = stock.options or []
        return {
            "ticker": ticker,
            "price": last_price,
            "expirations": expirations,
            "day_change_pct": day_change_pct,
            "range_52d": range_52d,
        }
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_option_chain(ticker: str, expiration: str) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        return chain.puts
    except Exception:
        return None


def build_candidates(
    ticker: str,
    target_delta: float,
    max_dte: int,
    min_dte: int,
    max_candidates: int = 3,
) -> List[Dict]:
    metadata = fetch_option_metadata(ticker)
    if metadata is None or metadata["expirations"] is None:
        return []

    price = metadata["price"]
    if np.isnan(price) or price <= 0:
        return []

    target_delta = -abs(target_delta)
    now = datetime.now()
    candidates: List[Dict] = []

    for expiration in metadata["expirations"]:
        try:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d")
        except ValueError:
            continue

        dte = max((exp_date - now).days, 0)
        if dte < min_dte or dte > max_dte:
            continue

        puts = fetch_option_chain(ticker, expiration)
        if puts is None or puts.empty:
            continue

        puts = puts.copy()
        # Ensure bid/ask numeric
        puts["bid"] = pd.to_numeric(puts.get("bid", 0), errors="coerce").fillna(0.0)
        puts["ask"] = pd.to_numeric(puts.get("ask", 0), errors="coerce").fillna(0.0)
        puts["mid"] = (puts["bid"] + puts["ask"]) / 2.0

        # Delta may not be present in the option chain returned by yfinance.
        # If missing, try to compute delta using any available implied vol column.
        if "delta" not in puts.columns:
            iv_cols = [c for c in puts.columns if "implied" in c.lower()]
            if iv_cols:
                iv_col = iv_cols[0]
                puts["impliedVol"] = pd.to_numeric(puts[iv_col], errors="coerce")
                T = max(dte / 365.0, 1.0 / 365.0)
                r = 0.02

                def _row_delta(row):
                    iv = row.get("impliedVol")
                    if pd.isna(iv) or iv <= 0:
                        return np.nan
                    return _bs_put_delta(price, float(row.get("strike", 0)), T, r, float(iv))

                puts["delta"] = puts.apply(_row_delta, axis=1)
            else:
                puts["delta"] = np.nan

        puts["delta"] = pd.to_numeric(puts["delta"], errors="coerce")
        puts = puts[puts["delta"].notna()]
        if puts.empty:
            continue

        puts["delta_diff"] = (puts["delta"] - target_delta).abs()
        puts["strike_below_price"] = puts["strike"] <= price
        puts = puts.sort_values(by=["delta_diff", "strike_below_price"], ascending=[True, False])
        best = puts.iloc[0]

        if best["bid"] <= 0 or best["strike"] <= 0:
            continue

        strike = float(best["strike"])
        bid = float(best["bid"])
        mid = float(best["mid"] if not np.isnan(best["mid"]) else bid)
        roi = bid / strike * 100
        prob_otm = round(1.0 - abs(best["delta"]), 4)
        option_volume_raw = best.get("volume", 0)
        if pd.isna(option_volume_raw):
            option_volume = 0
        else:
            try:
                option_volume = int(option_volume_raw)
            except (TypeError, ValueError):
                option_volume = 0

        candidates.append(
            {
                "ticker": ticker,
                "price": price,
                "stock_price": price,
                "expiration": expiration,
                "dte": dte,
                "strike": strike,
                "bid": bid,
                "ask": float(best["ask"]),
                "mid": mid,
                "volume": option_volume,
                "delta": float(best["delta"]),
                "roi_pct": roi,
                "prob_otm": prob_otm,
                "delta_gap": float(best["delta_diff"]),
                "day_change_pct": metadata.get("day_change_pct", 0.0),
                "range_52d": metadata.get("range_52d", "N/A"),
            }
        )

        if len(candidates) >= max_candidates:
            break

    return sorted(candidates, key=lambda row: row["roi_pct"], reverse=True)


def scan_tickers(
    tickers: List[str],
    target_delta: float,
    max_dte: int,
    min_dte: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for ticker in tickers:
        candidates = build_candidates(ticker, target_delta, max_dte, min_dte)
        if not candidates:
            rows.append({"ticker": ticker, "status": "no options found or data unavailable"})
            continue
        rows.extend(candidates)

    df = pd.DataFrame(rows)
    if not df.empty and "roi_pct" in df.columns:
        df = df.sort_values(by="roi_pct", ascending=False).reset_index(drop=True)
    return df


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    st.title("Cash Secured Put Options Scanner")
    st.write(
        "Use a limited ticker universe, target ~-0.15 delta, and max 40 days to expiration to find high ROI cash secured put candidates."
    )

    with st.sidebar:
        st.header("Scanner Settings")
        ticker_text = st.text_area(
            "Tickers (comma or space separated)",
            value="SOXL, DRAM, MSOS, IBIT, URA, UNG, ARKK, UVIX, IGV, UVXY, SLV, GDX, KWEB, SVIX, EFA, JETS",
            height=140,
        )
        target_delta = st.slider("Target absolute put delta", min_value=0.05, max_value=0.35, value=0.15, step=0.01)
        max_dte = st.slider("Max days to expiration", min_value=10, max_value=40, value=40, step=1)
        min_dte = st.slider("Min days to expiration", min_value=0, max_value=30, value=0, step=1)
        show_all = st.checkbox("Show all tickers even when no candidate found", value=True)
        run_scan = st.button("Scan Options")

    tickers = parse_tickers(ticker_text)

    if not tickers:
        st.warning("Enter at least one ticker symbol to begin scanning.")
        return

    if run_scan:
        with st.spinner(f"Scanning {len(tickers)} tickers... this may take a few moments."):
            result_df = scan_tickers(tickers, target_delta, max_dte, min_dte)

        candidate_df = result_df.copy()
        missing = None
        if "status" in candidate_df.columns:
            missing = candidate_df[candidate_df["status"].notna()][["ticker", "status"]]
            candidate_df = candidate_df[candidate_df["status"].isna()].drop(columns=["status"])

        if candidate_df.empty or "price" not in candidate_df.columns:
            st.error("No valid cash secured put candidates found. Try widening the DTE range or checking tickers.")
            if show_all and missing is not None and not missing.empty:
                st.warning("Some tickers did not return a valid candidate:")
                st.table(missing)
            return

        display_df = candidate_df.copy()
        display_df["price"] = display_df["price"].apply(format_currency)
        display_df["stock_price"] = display_df["stock_price"].apply(format_currency)
        display_df["strike"] = display_df["strike"].apply(format_currency)
        display_df["bid"] = display_df["bid"].apply(format_currency)
        display_df["ask"] = display_df["ask"].apply(format_currency)
        display_df["mid"] = display_df["mid"].apply(format_currency)
        display_df["roi_pct"] = display_df["roi_pct"].map(lambda v: f"{v:.2f}%")
        display_df["prob_otm"] = display_df["prob_otm"].map(lambda v: f"{v:.1%}")
        display_df["delta"] = display_df["delta"].map(lambda v: f"{v:.2f}")
        display_df["day_change_pct"] = display_df["day_change_pct"].map(lambda v: f"{v:.2f}%")

        st.subheader("Top Cash Secured Put Candidates")
        st.dataframe(
            display_df[
                [
                    "ticker",
                    "expiration",
                    "dte",
                    "stock_price",
                    "strike",
                    "bid",
                    "ask",
                    "mid",
                    "volume",
                    "delta",
                    "roi_pct",
                    "prob_otm",
                    "day_change_pct",
                    "range_52d",
                ]
            ].head(20),
            use_container_width=True,
        )
        if show_all and missing is not None and not missing.empty:
            st.warning("Some tickers did not return a valid candidate:")
            st.table(missing)

if __name__ == "__main__":
    main()