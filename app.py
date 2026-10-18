# -*- coding: utf-8 -*-
"""
HỆ THỐNG ĐỊNH LƯỢNG HOSE V4 - STREAMLIT WEB APP
Nested Walk-Forward FA/TA + Top N + EW / MPT / MinVar / HRP + Event-Driven Engine
Tác giả: Chuyên gia Phân tích & Lập trình Định lượng (Quantitative Finance)
"""

import os
import re
import math
import glob
import io
import warnings
from pathlib import Path
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

# ==========================================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="HOSE V4 | Kiểm Định Chiến Lược Kết Hợp",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-champion {
        background-color: #10B981;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HẰNG SỐ VÀ CẤU HÌNH MẶC ĐỊNH
# ==========================================================
BENCHMARK_ALIASES = {"VNINDEX", "VN-INDEX", "VNI", "^VNINDEX", "VNINDEXINDEX"}
TRADING_DAYS = 252

FA_ALIASES = {
    "ROE": ["roe", "return_on_equity", "returnonequity"],
    "ROA": ["roa", "return_on_assets", "returnonassets"],
    "RevenueGrowth": ["revenue_growth", "sales_growth", "revenuegrowth", "tangtruongdoanhthu"],
    "EPSGrowth": ["eps_growth", "epsgrowth", "tangtruongeps"],
    "GrossMargin": ["gross_margin", "grossmargin", "bienloinhuangop"],
    "OperatingMargin": ["operating_margin", "operatingmargin"],
    "NetMargin": ["net_margin", "netmargin", "bienloinhuanrong"],
    "CurrentRatio": ["current_ratio", "currentratio"],
    "DebtToEquity": ["debt_to_equity", "de", "d_e", "debttoequity"],
    "PE": ["pe", "p_e", "price_to_earnings", "pricetoearnings"],
    "PB": ["pb", "p_b", "price_to_book", "pricetobook"],
    "EVEBITDA": ["ev_ebitda", "evebitda"],
}

ACTIVE_MODEL_SPECS = {
    "DYN_EW_REGIME":      {"family": "EW",        "regime": True, "mask": True},
    "DYN_MPT_LW_REGIME":  {"family": "MPT_LW",    "regime": True, "mask": True},
    "DYN_MINVAR_REGIME":  {"family": "MINVAR_LW", "regime": True, "mask": True},
    "DYN_HRP_REGIME":     {"family": "HRP",       "regime": True, "mask": True},
    "DYN_BLEND50_REGIME": {"family": "BLEND50",   "regime": True, "mask": True},
}

# ==========================================================
# TIỆN ÍCH DỮ LIỆU & TIỀN XỬ LÝ (DATA PREPARATION)
# ==========================================================
def _canon(s):
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

def find_col(df, candidates):
    lookup = {_canon(c): c for c in df.columns}
    for c in candidates:
        if _canon(c) in lookup:
            return lookup[_canon(c)]
    return None

def parse_dates_safely(s):
    s = pd.Series(s)
    x = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return x

@st.cache_data(show_spinner=False)
def standardize_ohlcv(df_raw, use_adjusted=True):
    """Chuẩn hóa dữ liệu về định dạng Ticker-Date-OHLCV."""
    df = df_raw.copy()
    
    # Ánh xạ tên cột tiếng Việt thông dụng
    rename_dict = {
        "Mã": "Ticker",
        "Ngày": "Date",
        "Giá mở cửa": "Open",
        "Giá cao nhất": "High",
        "Giá thấp nhất": "Low",
        "Giá đóng cửa": "Close",
        "Giá điều chỉnh": "adj_close",
        "Khối lượng khớp lệnh": "Volume",
        "KL Khớp Lệnh": "Volume"
    }
    df = df.rename(columns=rename_dict)
    df.columns = [str(c).strip() for c in df.columns]

    ticker_col = find_col(df, ["ticker", "symbol", "code", "stock_code", "mack", "ma"])
    date_col = find_col(df, ["date", "datetime", "trading_date", "time", "ngay"])
    volume_col = find_col(df, ["volume", "vol", "trading_volume", "khoiluong", "klkhopl плеnh"])

    raw_cols = {
        "Open": find_col(df, ["open", "open_price", "giamocua"]),
        "High": find_col(df, ["high", "high_price", "giacaonhat"]),
        "Low": find_col(df, ["low", "low_price", "giathapnhat"]),
        "Close": find_col(df, ["close", "close_price", "giadongcua"]),
    }
    adj_cols = {
        "Open": find_col(df, ["adj_open", "adjusted_open"]),
        "High": find_col(df, ["adj_high", "adjusted_high"]),
        "Low": find_col(df, ["adj_low", "adjusted_low"]),
        "Close": find_col(df, ["adj_close", "adjusted_close", "adjclose", "giadieuchinh"]),
    }
    if ticker_col is None or date_col is None:
        raise ValueError(f"Thiếu cột Mã (Ticker) hoặc Ngày (Date). Các cột hiện có: {list(df.columns)}")

    use_adj = bool(use_adjusted and all(v is not None for v in adj_cols.values()))
    # Nếu chỉ có adj_close mà thiếu adj_open, adj_high, adj_low thì fallback raw OHLC kèm adj_close nếu cần
    if not use_adj and adj_cols["Close"] is not None and use_adjusted:
        pcols = {
            "Open": raw_cols["Open"] or adj_cols["Close"],
            "High": raw_cols["High"] or adj_cols["Close"],
            "Low": raw_cols["Low"] or adj_cols["Close"],
            "Close": adj_cols["Close"]
        }
    else:
        pcols = adj_cols if use_adj else raw_cols

    if not all(v is not None for v in pcols.values()):
        # Fallback cuối cùng
        pcols = {
            "Open": raw_cols["Open"] or find_col(df, ["open"]),
            "High": raw_cols["High"] or find_col(df, ["high"]),
            "Low": raw_cols["Low"] or find_col(df, ["low"]),
            "Close": raw_cols["Close"] or find_col(df, ["close"]),
        }

    if not all(v is not None for v in pcols.values()):
        raise ValueError(f"Thiếu các cột OHLC cần thiết trong dữ liệu: {list(df.columns)}")

    out = pd.DataFrame({
        "Ticker": df[ticker_col].astype(str).str.strip().str.upper(),
        "Date": parse_dates_safely(df[date_col]),
        "Open": pd.to_numeric(df[pcols["Open"]], errors="coerce"),
        "High": pd.to_numeric(df[pcols["High"]], errors="coerce"),
        "Low": pd.to_numeric(df[pcols["Low"]], errors="coerce"),
        "Close": pd.to_numeric(df[pcols["Close"]], errors="coerce"),
        "Volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0.0,
    })
    out = out.dropna(subset=["Ticker", "Date", "Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    out["Volume"] = out["Volume"].fillna(0).clip(lower=0)

    # Đảm bảo logic High/Low hợp lệ
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)

    out = out.drop_duplicates(["Ticker", "Date"], keep="last")
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def load_data_from_file_or_default(uploaded_file=None):
    """Tải dữ liệu từ file upload hoặc tự động tìm file Excel/CSV trong thư mục làm việc."""
    if uploaded_file is not None:
        fname = uploaded_file.name
        if fname.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)

    # Tìm file có sẵn trong thư mục
    excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
    if excel_files:
        return pd.read_excel(excel_files[0])
    csv_files = glob.glob("*.csv")
    if csv_files:
        return pd.read_csv(csv_files[0])
    return None

def data_quality_table(data):
    rows = []
    for t, g in data.groupby("Ticker"):
        rows.append({
            "Ticker": t,
            "Start": g["Date"].min().strftime("%Y-%m-%d"),
            "End": g["Date"].max().strftime("%Y-%m-%d"),
            "Sessions": g["Date"].nunique(),
            "MissingVolumePct": 100 * (g["Volume"] == 0).mean(),
            "MedianTradedValue": (g["Close"] * g["Volume"]).median(),
        })
    return pd.DataFrame(rows).sort_values("Sessions", ascending=False).reset_index(drop=True)

# ==========================================================
# CHỈ BÁO KỸ THUẬT & TA SCORE (0-10)
# ==========================================================
def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi_wilder(close, window=14):
    d = close.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    al = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - 100/(1+rs)
    rsi.loc[(al == 0) & (ag > 0)] = 100
    rsi.loc[(ag == 0) & (al > 0)] = 0
    return rsi

def atr_wilder(df, window=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

def adx_wilder(df, window=14):
    high, low = df["High"], df["Low"]
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = atr_wilder(df, window)
    plus_di = 100 * plus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return adx, plus_di, minus_di

def calculate_ta_features(df):
    out = df.copy().sort_index()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)

    ema12, ema26 = ema(out["Close"], 12), ema(out["Close"], 26)
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = ema(out["MACD"], 9)

    out["ADX"], out["Plus_DI"], out["Minus_DI"] = adx_wilder(out, 14)
    out["ATR"] = atr_wilder(out, 14)
    out["ATR_pct"] = 100 * out["ATR"] / out["Close"].replace(0, np.nan)
    out["ATR_pct_Q90"] = out["ATR_pct"].rolling(126, min_periods=40).quantile(0.90)

    out["Volume_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Volume_MA20"].replace(0, np.nan)
    out["OBV"] = (np.sign(out["Close"].diff()).fillna(0) * out["Volume"]).cumsum()
    out["OBV_EMA20"] = ema(out["OBV"], 20)

    out["RSI"] = rsi_wilder(out["Close"], 14)
    out["ROC10"] = out["Close"].pct_change(10) * 100
    out["High_52W"] = out["High"].rolling(252, min_periods=150).max()
    out["Close_to_52W_High"] = out["Close"] / out["High_52W"].replace(0, np.nan)
    out["Avg_Traded_Value_60"] = (out["Close"] * out["Volume"]).rolling(60, min_periods=20).mean()

    out["Momentum63"] = out["Close"].pct_change(63)
    out["Momentum126"] = out["Close"].pct_change(126)
    out["Momentum252"] = out["Close"].pct_change(252)
    out["Vol63"] = out["Close"].pct_change().rolling(63, min_periods=40).std() * np.sqrt(TRADING_DAYS)
    rollmax = out["Close"].rolling(126, min_periods=60).max()
    out["DD126"] = out["Close"] / rollmax - 1

    # TA Score 0-10 theo cấu trúc Group 1
    out["Trend_Score"] = (
        (out["Close"] > out["EMA50"]).astype(int)*2 +
        (out["EMA50"] > out["EMA200"]).astype(int)*2 +
        (out["MACD"] > out["MACD_Signal"]).astype(int)
    ).clip(0, 5)

    momentum_core = ((out["RSI"] >= 45) & (out["RSI"] <= 75)) | (out["ROC10"] > 0)
    out["Momentum_Score"] = (
        momentum_core.astype(int)*2 +
        (out["Close_to_52W_High"] >= 0.85).astype(int)
    ).clip(0, 3)

    out["Volume_Score"] = ((out["Volume_Ratio"] >= 1.0) | (out["OBV"] > out["OBV_EMA20"])).astype(int)
    out["Volatility_Score"] = (
        out["ATR_pct"].notna() & out["ATR_pct_Q90"].notna() &
        (out["ATR_pct"] <= out["ATR_pct_Q90"])
    ).astype(int)

    out["TA_Score"] = (
        out["Trend_Score"] + out["Momentum_Score"] +
        out["Volume_Score"] + out["Volatility_Score"]
    ).clip(0, 10).astype(int)
    return out

@st.cache_data(show_spinner=False)
def build_feature_store(data):
    """Tính toán TA feature store cho tất cả mã cổ phiếu."""
    price_dict = {
        t: g[["Date", "Open", "High", "Low", "Close", "Volume"]]
             .drop_duplicates("Date").sort_values("Date").set_index("Date")
        for t, g in data.groupby("Ticker")
    }

    # Benchmark proxy nếu không có VNINDEX
    benchmark_ticker = next((t for t in price_dict if t in BENCHMARK_ALIASES), None)
    if benchmark_ticker is None:
        benchmark_ticker = "__MARKET_PROXY__"
        closes = [px["Close"].rename(t) for t, px in price_dict.items()]
        m = pd.concat(closes, axis=1).sort_index()
        ret = m.pct_change(fill_method=None)
        ewret = ret.mean(axis=1, skipna=True).fillna(0)
        close = 1000 * (1 + ewret).cumprod()
        price_dict[benchmark_ticker] = pd.DataFrame({
            "Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0
        }, index=close.index)

    feature_store = {t: calculate_ta_features(px) for t, px in price_dict.items()}
    return price_dict, feature_store, benchmark_ticker

def stateful_active_from_score(score, buy_threshold, exit_threshold):
    state = 0
    vals = []
    for v in score.fillna(-np.inf):
        if v >= buy_threshold:
            state = 1
        elif v <= exit_threshold:
            state = 0
        vals.append(state)
    return pd.Series(vals, index=score.index, dtype=int)

# ==========================================================
# PHÂN CHIA THỜI GIAN (DATA SPLITS)
# ==========================================================
def resolve_splits(dates, split_mode="auto", train_prop=0.6, val_prop=0.2, test_prop=0.2):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates.unique())))
    n = len(dates)
    if n < 200:
        raise ValueError(f"Dữ liệu chỉ có {n} phiên, không đủ để phân chia Train/Val/Test.")

    if split_mode == "proportions":
        i_train = max(60, int(n * train_prop))
        i_val_end = max(i_train + 40, int(n * (train_prop + val_prop)))
        i_val_end = min(i_val_end, n - 40)
        core_end = dates[i_train - 1]
        val_start, val_end = dates[i_train], dates[i_val_end - 1]
        test_start, test_end = dates[i_val_end], dates[-1]
        return dates, core_end, val_start, val_end, test_start, test_end

    # Chế độ "auto": ưu tiên năm cuối cùng làm Test
    last_year = dates[-1].year
    last_year_dates = dates[dates.year == last_year]
    pre_last_year_count = int((dates < last_year_dates[0]).sum()) if len(last_year_dates) else 0

    if len(last_year_dates) >= 150 and pre_last_year_count >= 200:
        test_start, test_end = last_year_dates[0], last_year_dates[-1]
        pretest = dates[dates < test_start]
    else:
        test_n = max(60, min(252, int(round(test_prop * n))))
        test_start, test_end = dates[-test_n], dates[-1]
        pretest = dates[dates < test_start]

    val_n = max(40, min(252, int(round(0.25 * len(pretest)))))
    val_n = min(val_n, len(pretest) - 100)
    val_start, val_end = pretest[-val_n], pretest[-1]
    core_end = pretest[-val_n - 1]
    return dates, core_end, val_start, val_end, test_start, test_end

# ==========================================================
# ĐÁNH GIÁ CHỈ SỐ HIỆU SUẤT (METRICS)
# ==========================================================
def annualized_sharpe(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    return np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=1) if len(r) > 1 and r.std(ddof=1) > 1e-12 else np.nan

def annualized_sortino(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    neg = r[r < 0]
    return np.sqrt(TRADING_DAYS) * r.mean() / neg.std(ddof=1) if len(neg) > 1 and neg.std(ddof=1) > 1e-12 else np.nan

def metrics_from_equity(eq):
    eq = pd.Series(eq).dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1/365.25)
    cagr = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan
    return {
        "Return [%]": total * 100,
        "CAGR [%]": cagr * 100,
        "Sharpe": annualized_sharpe(r),
        "Sortino": annualized_sortino(r),
        "Volatility [%]": vol * 100 if pd.notna(vol) else np.nan,
        "MaxDD [%]": mdd * 100 if pd.notna(mdd) else np.nan,
        "Calmar": cagr / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan,
    }

# ==========================================================
# BACKTEST ĐƠN MÃ & TỐI ƯU HÓA NGƯỠNG BUY/EXIT
# ==========================================================
def simulate_single_stock(features, start, end, buy_threshold, exit_threshold,
                          initial_cash=1_000_000.0, slippage=0.0005, commission=0.0015):
    active_close = stateful_active_from_score(features["TA_Score"], buy_threshold, exit_threshold)
    active_open = active_close.shift(1).fillna(0).astype(int)
    x = features.loc[(features.index >= pd.Timestamp(start)) & (features.index <= pd.Timestamp(end))].copy()
    active_open = active_open.reindex(x.index).fillna(0).astype(int)
    if x.empty:
        return {}

    cash, shares, state, trades = float(initial_cash), 0.0, 0, 0
    eq = []
    for dt, row in x.iterrows():
        d = int(active_open.loc[dt])
        op, cl = float(row["Open"]), float(row["Close"])
        if d != state:
            if d == 1:
                px = op * (1 + slippage)
                qty = math.floor(cash / (px * (1 + commission)))
                if qty > 0:
                    cash -= qty * px * (1 + commission)
                    shares = qty
                    state = 1
                    trades += 1
            else:
                px = op * (1 - slippage)
                cash += shares * px * (1 - commission)
                shares = 0
                state = 0
                trades += 1
        eq.append((dt, cash + shares * cl))

    m = metrics_from_equity(pd.Series(dict(eq)).sort_index())
    m["Trades"] = trades
    return m

def build_universe(price_dict, asof, benchmark_ticker=None,
                   min_history_floor=180, min_history_cap=400,
                   use_liquidity_filter=True, liquidity_min_percentile=0.20):
    all_dates = set()
    for px in price_dict.values():
        all_dates.update(px.loc[:asof].index)
    pre_sessions = len(all_dates)
    min_hist = min(min_history_cap, max(min_history_floor, int(0.55 * pre_sessions)))

    rows = []
    for t, px in price_dict.items():
        if t == benchmark_ticker:
            continue
        h = px.loc[:asof]
        if h.empty:
            continue
        avg_value = (h["Close"] * h["Volume"]).tail(60).mean()
        rows.append({
            "Ticker": t,
            "HistorySessions": len(h),
            "AvgTradedValue": avg_value,
            "EligibleHistory": len(h) >= min_hist,
        })

    u = pd.DataFrame(rows)
    if u.empty:
        return u
    ok = u["EligibleHistory"] & u["AvgTradedValue"].notna()
    u.loc[ok, "LiquidityPercentile"] = u.loc[ok, "AvgTradedValue"].rank(pct=True)
    u["PassLiquidity"] = (
        u["LiquidityPercentile"].fillna(0) >= liquidity_min_percentile
        if use_liquidity_filter else True
    )
    u["Candidate"] = u["EligibleHistory"] & u["PassLiquidity"]
    return u.sort_values("AvgTradedValue", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def optimize_core_thresholds(feature_store_keys, _feature_store, candidates, dates_list, core_end):
    core_dates = pd.DatetimeIndex([d for d in dates_list if d <= pd.Timestamp(core_end)])
    inner_n = max(40, min(126, int(0.30 * len(core_dates))))
    inner_start = core_dates[-inner_n]
    
    # 3 Folds
    d_sub = core_dates[core_dates >= inner_start]
    chunks = np.array_split(d_sub, 3)
    folds = [{"Fold": f"F{i+1}", "start": c[0], "end": c[-1]} for i, c in enumerate(chunks) if len(c) >= 10]

    trials = []
    for b, e in product([5, 6, 7], [2, 3, 4]):
        if e >= b:
            continue
        stock_scores = []
        for t in candidates:
            ms = []
            for f in folds:
                m = simulate_single_stock(_feature_store[t], f["start"], f["end"], b, e)
                if m:
                    sh = m.get("Sharpe", np.nan)
                    sh = -3.0 if not np.isfinite(sh) else float(sh)
                    ms.append((sh, float(m.get("Return [%]", 0)), float(m.get("MaxDD [%]", -100))))
            if ms:
                a = np.array(ms, float)
                stock_scores.append([np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), np.nanmin(a[:, 2])])
        if not stock_scores:
            continue

        z = np.array(stock_scores, float)
        med_sh = float(np.nanmedian(z[:, 0]))
        sd_sh = float(np.nanstd(z[:, 0]))
        med_ret = float(np.nanmedian(z[:, 1]))
        med_dd = float(np.nanmedian(z[:, 2]))
        score = med_sh - 0.20 * sd_sh + 0.005 * med_ret + 0.003 * med_dd
        trials.append({
            "BuyThreshold": b,
            "ExitThreshold": e,
            "GlobalCVScore": score,
            "MedianSharpe": med_sh,
            "SharpeStd": sd_sh,
            "MedianReturnPct": med_ret,
            "MedianMaxDDPct": med_dd,
            "NStocks": len(stock_scores),
        })

    tr = pd.DataFrame(trials).sort_values("GlobalCVScore", ascending=False).reset_index(drop=True)
    if tr.empty:
        return 6, 3, tr
    best = tr.iloc[0]
    return int(best["BuyThreshold"]), int(best["ExitThreshold"]), tr

# ==========================================================
# NESTED WALK-FORWARD STOCK SELECTION & RANKING
# ==========================================================
def trailing_stock_quality(features, asof, b, e, lookback=126):
    h = features.loc[:asof]
    if len(h) < 40:
        return {"TrailingSharpe": -3.0, "TrailingReturnPct": -100.0, "TrailingMaxDDPct": -100.0}
    sub_idx = h.index[-min(lookback, len(h)):]
    m = simulate_single_stock(features, sub_idx[0], sub_idx[-1], b, e)
    sh = m.get("Sharpe", np.nan) if m else np.nan
    return {
        "TrailingSharpe": -3.0 if not np.isfinite(sh) else float(sh),
        "TrailingReturnPct": float(m.get("Return [%]", -100.0)) if m else -100.0,
        "TrailingMaxDDPct": float(m.get("MaxDD [%]", -100.0)) if m else -100.0,
    }

def rank_and_select_asof(feature_store, price_dict, benchmark_features, benchmark_ticker,
                         asof, b, e, top_n=5, max_stocks_per_sector=2, sector_map=None):
    sector_map = sector_map or {}
    universe = build_universe(price_dict, asof, benchmark_ticker)
    if universe.empty:
        raise ValueError(f"Universe rỗng tại {asof}")
    candidates = universe.loc[universe["Candidate"], "Ticker"].tolist()
    bench = benchmark_features.loc[:asof]
    bench_m126 = bench["Close"].pct_change(126).iloc[-1] if len(bench) > 126 else np.nan
    uidx = universe.set_index("Ticker")

    rows = []
    for t in candidates:
        feat = feature_store[t].loc[:asof]
        if feat.empty:
            continue
        snap = feat.iloc[-1]
        tq = trailing_stock_quality(feature_store[t], asof, b, e)
        rows.append({
            "Ticker": t,
            **tq,
            "TA_Score": float(snap.get("TA_Score", np.nan)),
            "Momentum63": float(snap.get("Momentum63", np.nan)),
            "Momentum126": float(snap.get("Momentum126", np.nan)),
            "Momentum252": float(snap.get("Momentum252", np.nan)),
            "RelativeStrength126": (
                float(snap.get("Momentum126", np.nan) - bench_m126)
                if pd.notna(bench_m126) else float(snap.get("Momentum126", np.nan))
            ),
            "Vol63": float(snap.get("Vol63", np.nan)),
            "RSI": float(snap.get("RSI", np.nan)),
            "LiquidityPercentile": float(uidx.loc[t, "LiquidityPercentile"]) if t in uidx.index else 0.5,
            "Sector": sector_map.get(t, "UNKNOWN"),
        })

    r = pd.DataFrame(rows)
    if r.empty:
        raise ValueError(f"Không có cổ phiếu đủ điều kiện xếp hạng tại {asof}")

    # Gate 1: ưu tiên TA Score >= 4
    r["PassTA"] = r["TA_Score"].fillna(0) >= 4
    if int(r["PassTA"].sum()) >= top_n:
        r = r[r["PassTA"]].copy()

    # Gate 2: Lọc biến động và sụt giảm sâu
    vol_cap = r["Vol63"].quantile(0.75)
    dd_floor = r["TrailingMaxDDPct"].quantile(0.15)
    r["PassRiskGate"] = (
        (r["Vol63"].isna() | (r["Vol63"] <= vol_cap)) &
        (r["TrailingMaxDDPct"].isna() | (r["TrailingMaxDDPct"] >= dd_floor)) &
        (r["RSI"].isna() | (r["RSI"] <= 80.0))
    )
    if int(r["PassRiskGate"].sum()) >= top_n:
        r = r[r["PassRiskGate"]].copy()

    # Điểm tổng hợp xếp hạng
    r["P_Trailing"] = r["TrailingSharpe"].rank(pct=True)
    r["P_TA"] = r["TA_Score"].rank(pct=True)
    r["P_RS"] = r["RelativeStrength126"].rank(pct=True)
    r["P_LowVol"] = (-r["Vol63"]).rank(pct=True)
    r["P_Liq"] = r["LiquidityPercentile"].rank(pct=True)
    mommix = 0.6 * r["Momentum126"].fillna(r["Momentum63"]) + 0.4 * r["Momentum63"].fillna(0)
    r["P_Momentum"] = mommix.rank(pct=True)

    r["FinalScore"] = (
        0.25 * r["P_Trailing"] +
        0.20 * r["P_TA"] +
        0.25 * r["P_RS"] +
        0.10 * r["P_LowVol"] +
        0.05 * r["P_Liq"] +
        0.15 * r["P_Momentum"]
    ) * 100

    r = r.sort_values(["FinalScore", "TrailingSharpe"], ascending=False).reset_index(drop=True)

    # Correlation cap & Sector cap
    corr_tickers = r["Ticker"].tolist()
    close_list = [feature_store[t].loc[:asof, "Close"].rename(t) for t in corr_tickers]
    corr_px = pd.concat(close_list, axis=1, join="inner").tail(127) if close_list else pd.DataFrame()
    corr = corr_px.pct_change(fill_method=None).dropna().corr() if len(corr_px) > 20 else pd.DataFrame()

    selected, sector_count = [], {}
    for _, row in r.iterrows():
        t, sec = row["Ticker"], row["Sector"]
        if sector_map and sec != "UNKNOWN" and sector_count.get(sec, 0) >= max_stocks_per_sector:
            continue
        if selected and not corr.empty:
            too_close = any(
                t in corr.index and x in corr.columns and
                pd.notna(corr.loc[t, x]) and abs(corr.loc[t, x]) > 0.80
                for x in selected
            )
            if too_close:
                continue
        selected.append(t)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for t in r["Ticker"]:
            if t not in selected:
                selected.append(t)
            if len(selected) >= top_n:
                break

    locked = r[r["Ticker"].isin(selected)].copy()
    locked["SelectedOrder"] = locked["Ticker"].map({t: i+1 for i, t in enumerate(selected)})
    locked["AsOfDate"] = pd.Timestamp(asof)
    return r, locked.sort_values("SelectedOrder"), selected

def first_session_each_period(calendar, start, end, freq="Q"):
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    key = idx.to_period(freq)
    firsts = s.groupby(key).first().tolist()
    if idx[0] not in firsts:
        firsts = [idx[0]] + firsts
    return sorted(set(pd.Timestamp(x) for x in firsts))

def previous_session(calendar, dt):
    idx = pd.DatetimeIndex(calendar)
    prev = idx[idx < pd.Timestamp(dt)]
    return pd.Timestamp(prev[-1]) if len(prev) else None

def build_selection_schedule(feature_store, price_dict, benchmark_ticker, benchmark_features,
                             calendar, start, end, b, e, top_n=5, freq="Q", sector_map=None):
    effective_dates = first_session_each_period(calendar, start, end, freq)
    rows = []
    details = {}

    for eff in effective_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        ranking, locked, selected = rank_and_select_asof(
            feature_store, price_dict, benchmark_features, benchmark_ticker,
            asof, b, e, top_n=top_n, sector_map=sector_map
        )
        details[pd.Timestamp(eff)] = {"ranking": ranking, "locked": locked}
        row = {
            "EffectiveDate": pd.Timestamp(eff),
            "AsOfDate": pd.Timestamp(asof),
        }
        for i in range(top_n):
            row[f"Top{i+1}"] = selected[i] if i < len(selected) else None
        rows.append(row)

    schedule = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    return schedule, details

def selected_tickers_asof(selection_schedule, effective_date, top_n=5):
    x = selection_schedule[selection_schedule["EffectiveDate"] <= pd.Timestamp(effective_date)]
    if x.empty:
        return []
    row = x.iloc[-1]
    return [row[f"Top{i+1}"] for i in range(top_n) if pd.notna(row.get(f"Top{i+1}"))]

# ==========================================================
# CÁC MÔ HÌNH PHÂN BỔ DANH MỤC (PORTFOLIO ALLOCATION)
# ==========================================================
def close_matrix(price_dict, tickers, end=None, start=None):
    series = [price_dict[t]["Close"].rename(t) for t in tickers if t in price_dict]
    if not series:
        return pd.DataFrame()
    m = pd.concat(series, axis=1, join="inner").sort_index().dropna(how="any")
    if start is not None:
        m = m.loc[m.index >= pd.Timestamp(start)]
    if end is not None:
        m = m.loc[m.index <= pd.Timestamp(end)]
    return m

def equal_weights(tickers):
    tickers = list(tickers)
    return pd.Series(1/len(tickers), index=tickers, dtype=float) if tickers else pd.Series(dtype=float)

def robust_mu(prices, lookback=252, mu_shrinkage=0.50):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if r.empty:
        return pd.Series(0.0, index=prices.columns)
    span = max(20, min(lookback, len(r)))
    raw = r.ewm(span=span, adjust=False).mean().iloc[-1] * TRADING_DAYS
    grand = float(raw.mean())
    return mu_shrinkage * raw + (1 - mu_shrinkage) * grand

def lw_cov(prices, lookback=252):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if len(r) < 30:
        return r.cov() * TRADING_DAYS
    est = LedoitWolf().fit(r.values)
    return pd.DataFrame(est.covariance_ * TRADING_DAYS, index=r.columns, columns=r.columns)

def max_sharpe_weights(mu, cov, max_weight=0.50, l2_gamma=0.03):
    tickers = list(mu.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1/n)
    muv = mu.values.astype(float)
    cv = cov.reindex(index=tickers, columns=tickers).fillna(0).values.astype(float) + np.eye(n)*1e-8

    def obj(w):
        ret = float(w @ muv)
        vol = math.sqrt(max(float(w @ cv @ w), 1e-12))
        return -(ret / vol) + l2_gamma * float(np.sum(w*w))

    x0 = np.repeat(1/n, n)
    bounds = [(0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-8})
    if not res.success or np.any(~np.isfinite(res.x)):
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=tickers) if w.sum() > 0 else equal_weights(tickers)

def min_variance_weights(cov, max_weight=0.50):
    tickers = list(cov.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1/n)
    cv = cov.values.astype(float) + np.eye(n)*1e-8

    def obj(w):
        return float(w @ cv @ w)

    x0 = np.repeat(1/n, n)
    bounds = [(0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-8})
    if not res.success:
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=tickers)

def _cluster_var(cov, items):
    sub = cov.loc[items, items].values
    ivp = 1 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)

def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i+1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def hrp_weights(prices, method="single"):
    r = prices.pct_change(fill_method=None).dropna()
    if len(r) < 30 or prices.shape[1] < 2:
        return equal_weights(prices.columns)
    cov = r.cov() * TRADING_DAYS
    corr = r.corr().clip(-1, 1)
    dist = np.sqrt((1 - corr) / 2)
    link = linkage(squareform(dist.values, checks=False), method=method)
    ordered = corr.index[_quasi_diag(link)].tolist()

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            s = len(c) // 2
            c0, c1 = c[:s], c[s:]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            a = 1 - v0/(v0+v1) if v0+v1 > 0 else 0.5
            w[c0] *= a
            w[c1] *= (1 - a)
            nxt += [c0, c1]
        clusters = nxt
    return w.reindex(prices.columns).fillna(0) / w.sum()

def compute_weights(model_family, prices, weight_lookback=252):
    tickers = list(prices.columns)
    ew = equal_weights(tickers)
    if len(tickers) == 0:
        return ew
    if model_family == "EW":
        return ew

    cov = lw_cov(prices, weight_lookback)
    if model_family == "MINVAR_LW":
        return min_variance_weights(cov)
    if model_family == "HRP":
        return hrp_weights(prices.tail(weight_lookback))

    mu = robust_mu(prices, weight_lookback)
    mpt = max_sharpe_weights(mu, cov)
    if model_family == "MPT_LW":
        return mpt
    if model_family == "BLEND50":
        w = 0.5 * mpt + 0.5 * ew
        return w / w.sum()
    return ew

# ==========================================================
# RISK REGIME, STOCK MASK VÀ EVENT-DRIVEN PORTFOLIO ENGINE
# ==========================================================
def build_market_regime(benchmark_features, ema_span=200, slope_lookback=20):
    x = benchmark_features.copy()
    ema_m = x["Close"].ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    slope = ema_m - ema_m.shift(slope_lookback)
    risk_close = (x["Close"] > ema_m) & (slope > 0)
    return pd.DataFrame({
        "EMA_Market": ema_m,
        "EMA_Slope": slope,
        "RiskOn_Close": risk_close.fillna(False),
        "RiskOn_Open": risk_close.shift(1).fillna(False),
    }, index=x.index)

def build_stock_open_masks(feature_store, tickers, b, e):
    masks = {}
    for t in tickers:
        if t in feature_store:
            active_close = stateful_active_from_score(feature_store[t]["TA_Score"], b, e)
            masks[t] = active_close.shift(1).fillna(0).astype(bool)
        else:
            masks[t] = pd.Series(True)
    return pd.DataFrame(masks).sort_index()

def build_dynamic_weight_schedule(price_dict, selection_schedule, calendar, start, end,
                                  model_family, allocation_freq="M", weight_lookback=252, top_n=5):
    rebalance_dates = first_session_each_period(calendar, start, end, allocation_freq)
    rows = []
    all_tickers = set()

    for eff in rebalance_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        top = selected_tickers_asof(selection_schedule, eff, top_n=top_n)
        if not top:
            continue
        all_tickers.update(top)
        prices = close_matrix(price_dict, top, end=asof)
        if prices.empty or len(prices) < 30:
            w = equal_weights(top)
        else:
            w = compute_weights(model_family, prices.tail(weight_lookback + 1), weight_lookback=weight_lookback)
        row = {"EffectiveDate": pd.Timestamp(eff), "AsOfDate": pd.Timestamp(asof)}
        row.update({t: float(w.get(t, 0.0)) for t in top})
        rows.append(row)

    if not rows:
        raise ValueError("Không tạo được weight schedule")

    meta = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    tickers = sorted(all_tickers)
    weights = meta.set_index("EffectiveDate").reindex(columns=tickers).fillna(0.0)
    return weights, meta[["EffectiveDate", "AsOfDate"]]

def run_portfolio_engine_dynamic(price_dict, weight_schedule, calendar, start, end,
                                 market_regime_open=None, stock_active_open=None,
                                 use_market_regime=True, use_stock_mask=True,
                                 initial_cash=1_000_000_000.0, commission=0.0015,
                                 slippage=0.0005, fractional=False):
    tickers = list(weight_schedule.columns)
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        raise ValueError("Không có ngày giao dịch trong kỳ backtest")

    opens = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    closes = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        if t in price_dict:
            opens[t] = price_dict[t]["Open"].reindex(idx)
            closes[t] = price_dict[t]["Close"].reindex(idx)

    ws = weight_schedule.sort_index().reindex(columns=tickers)
    ws = ws.reindex(ws.index.union(idx)).sort_index().ffill().reindex(idx).fillna(0.0)

    regime = (
        pd.Series(True, index=idx)
        if market_regime_open is None
        else pd.Series(market_regime_open).reindex(idx).ffill().fillna(False).astype(bool)
    )
    masks = (
        pd.DataFrame(True, index=idx, columns=tickers)
        if stock_active_open is None
        else stock_active_open.reindex(index=idx, columns=tickers).fillna(False).astype(bool)
    )

    cash = float(initial_cash)
    shares = pd.Series(0.0, index=tickers)
    last_close = pd.Series(np.nan, index=tickers)
    last_target = pd.Series(np.nan, index=tickers)

    orders, eq_rows, exposure_rows = [], [], []
    turnover, rebals = 0.0, 0

    for dt in idx:
        op = opens.loc[dt].astype(float)
        cl = closes.loc[dt].astype(float)

        valid_close = cl.notna()
        last_close.loc[valid_close] = cl.loc[valid_close]
        mark_open = op.where(op.notna(), last_close)
        mark_close = cl.where(cl.notna(), last_close)

        base = ws.loc[dt].clip(lower=0)
        if base.sum() > 1 + 1e-8:
            base = base / base.sum()

        risk_on = bool(regime.loc[dt]) if use_market_regime else True
        active = masks.loc[dt].astype(bool) if use_stock_mask else pd.Series(True, index=tickers)

        target = base.copy()
        if use_stock_mask:
            target *= active.astype(float)
        if use_market_regime and not risk_on:
            target[:] = 0.0

        target = target.clip(lower=0)
        if target.sum() > 1 + 1e-8:
            target /= target.sum()

        changed = last_target.isna().any() or (target - last_target.fillna(0)).abs().max() > 1e-10
        if changed:
            equity_open = cash + float((shares * mark_open.fillna(0)).sum())
            desired = shares.copy()

            for t in tickers:
                if pd.isna(op[t]) or op[t] <= 0:
                    continue
                if target[t] <= 0:
                    desired[t] = 0.0
                else:
                    px = op[t] * (1 + slippage)
                    raw_qty = equity_open * target[t] / (px * (1 + commission))
                    desired[t] = raw_qty if fractional else math.floor(raw_qty)

            # Bán trước
            for t in tickers:
                d = desired[t] - shares[t]
                if d < -1e-12 and pd.notna(op[t]) and op[t] > 0:
                    qty = -d
                    px = op[t] * (1 - slippage)
                    notional = qty * px
                    fee = notional * commission
                    cash += notional - fee
                    shares[t] -= qty
                    turnover += notional
                    orders.append({"Date": dt.strftime("%Y-%m-%d"), "Ticker": t, "Side": "SELL", "Shares": qty, "ExecPrice": px, "Fee": fee})

            # Mua sau
            for t in tickers:
                d = desired[t] - shares[t]
                if d > 1e-12 and pd.notna(op[t]) and op[t] > 0:
                    px = op[t] * (1 + slippage)
                    unit = px * (1 + commission)
                    qty = min(d, cash / unit) if fractional else min(d, math.floor(cash / unit))
                    if qty > 0:
                        notional = qty * px
                        fee = notional * commission
                        cash -= notional + fee
                        shares[t] += qty
                        turnover += notional
                        orders.append({"Date": dt.strftime("%Y-%m-%d"), "Ticker": t, "Side": "BUY", "Shares": qty, "ExecPrice": px, "Fee": fee})

            last_target = target.copy()
            rebals += 1

        equity = float(cash + (shares * mark_close.fillna(0)).sum())
        invested = float((shares * mark_close.fillna(0)).sum())
        exposure = invested / equity if equity > 0 else 0.0
        exposure_rows.append(exposure)
        eq_rows.append((dt, equity))

    eq = pd.Series(dict(eq_rows)).sort_index()
    st_res = metrics_from_equity(eq)
    st_res.update({
        "#Orders": len(orders),
        "#Rebalances": rebals,
        "Turnover [%]": 100 * turnover / initial_cash,
        "AvgExposure [%]": 100 * np.mean(exposure_rows) if exposure_rows else 0.0,
        "AvgCash [%]": 100 * (1 - np.mean(exposure_rows)) if exposure_rows else 100.0,
    })
    return st_res, eq, pd.DataFrame(orders)

def benchmark_nav(benchmark_px, start, end, initial_cash=1_000_000_000.0):
    x = benchmark_px.loc[(benchmark_px.index >= pd.Timestamp(start)) & (benchmark_px.index <= pd.Timestamp(end)), "Close"].dropna()
    if x.empty:
        return pd.Series(dtype=float)
    return initial_cash * x / x.iloc[0]

def model_validation_score(stats, benchmark_stats):
    sh = stats.get("Sharpe", np.nan)
    sh = -3 if not np.isfinite(sh) else sh
    so = stats.get("Sortino", np.nan)
    so = -3 if not np.isfinite(so) else so
    ret = stats.get("Return [%]", -100)
    dd = stats.get("MaxDD [%]", -100)
    turnover = stats.get("Turnover [%]", 0)
    bret = benchmark_stats.get("Return [%]", 0)
    return 0.45*sh + 0.15*so + 0.012*(ret - bret) + 0.004*dd - 0.0005*turnover

# ==========================================================
# GIAO DIỆN CHÍNH (STREAMLIT APP LAYOUT)
# ==========================================================
def main():
    st.markdown('<div class="main-title">📈 HỆ THỐNG ĐỊNH LƯỢNG HOSE V4</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Kiểm định chiến lược kết hợp FA/TA + Nested Walk-Forward Top N + Portfolio Allocation Models (EW / MPT / MinVar / HRP)</div>', unsafe_allow_html=True)

    # ------------------------------------------------------
    # SIDEBAR CẤU HÌNH
    # ------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Cấu Hình Tham Số")

        st.subheader("1. Nguồn Dữ Liệu")
        uploaded_file = st.file_uploader(
            "Tải file Excel / CSV dữ liệu OHLCV",
            type=["xlsx", "xls", "csv"],
            help="Nếu không tải file mới, hệ thống sẽ tự động dùng file dữ liệu có sẵn trong thư mục."
        )

        use_adjusted_prices = st.checkbox("Ưu tiên giá điều chỉnh (Adjusted)", value=True)

        st.subheader("2. Thiết Kế Danh Mục")
        top_n = st.slider("Số lượng cổ phiếu Top N:", min_value=3, max_value=15, value=5, step=1)
        selection_freq = st.selectbox("Tần suất chọn lại Top N:", options=["Q", "M"], format_func=lambda x: "Hàng Quý (Quarterly)" if x == "Q" else "Hàng Tháng (Monthly)")
        allocation_freq = st.selectbox("Tần suất tái cơ cấu tỷ trọng:", options=["M", "Q"], format_func=lambda x: "Hàng Tháng (Monthly)" if x == "M" else "Hàng Quý (Quarterly)")

        st.subheader("3. Ngưỡng Tín Hiệu (TA Score)")
        opt_mode = st.radio("Cách xác định BUY/EXIT:", ["Tối ưu tự động trên Train", "Thiết lập thủ công"], index=0)
        if opt_mode == "Thiết lập thủ công":
            manual_buy = st.slider("Ngưỡng BUY (vào lệnh):", min_value=4, max_value=9, value=6)
            manual_exit = st.slider("Ngưỡng EXIT (thoát lệnh):", min_value=1, max_value=manual_buy-1, value=3)
        else:
            manual_buy, manual_exit = 6, 3

        st.subheader("4. Quản Trị Rủi Ro & Khớp Lệnh")
        use_market_regime = st.checkbox("Bật Market Regime (EMA200)", value=True, help="Rút về 100% Cash khi VN-INDEX dưới EMA200 hoặc dốc âm")
        use_stock_mask = st.checkbox("Bật Stock Active Mask", value=True, help="Chuyển tỷ trọng sang Cash khi cổ phiếu rơi vào trạng thái Exit")
        commission = st.number_input("Phí giao dịch (%):", min_value=0.0, max_value=1.0, value=0.15, step=0.01) / 100
        slippage = st.number_input("Trượt giá ước tính (%):", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100
        initial_cash = st.number_input("Vốn khởi đầu (VND):", min_value=10_000_000, value=1_000_000_000, step=100_000_000, format="%d")

        st.markdown("---")
        run_btn = st.button("🚀 BẮT ĐẦU CHẠY KIỂM ĐỊNH", use_container_width=True, type="primary")

    # ------------------------------------------------------
    # TẢI & CHUẨN HÓA DỮ LIỆU
    # ------------------------------------------------------
    with st.spinner("Đang nạp và kiểm tra dữ liệu thị trường..."):
        try:
            raw_data = load_data_from_file_or_default(uploaded_file)
            if raw_data is None:
                st.error("❌ Không tìm thấy dữ liệu! Vui lòng tải file Excel (.xlsx) hoặc CSV dữ liệu giá OHLCV tại thanh bên trái.")
                st.stop()

            data = standardize_ohlcv(raw_data, use_adjusted=use_adjusted_prices)
            price_dict, feature_store, benchmark_ticker = build_feature_store(data)
            all_dates = pd.DatetimeIndex(sorted(data["Date"].unique()))
            dates, core_end, val_start, val_end, test_start, test_end = resolve_splits(all_dates)
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc dữ liệu: {e}")
            st.stop()

    # ------------------------------------------------------
    # CÁC TAB NỘI DUNG CHÍNH
    # ------------------------------------------------------
    tabs = st.tabs([
        "📊 1. Dữ Liệu & Universe",
        "⚙️ 2. Tối Ưu Ngưỡng TA (Train)",
        "🔍 3. Đấu Trường Validation",
        "🏆 4. Kiểm Định Thực Chiến (Final Test)",
        "📜 5. Lịch Sử Danh Mục & Lệnh",
        "🛡️ 6. Leakage Audit & Xuất Báo Cáo"
    ])

    # ======================================================
    # TAB 1: DỮ LIỆU & UNIVERSE
    # ======================================================
    with tabs[0]:
        st.subheader("Tổng Quan Bộ Dữ Liệu & Universe Cổ Phiếu")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng Số Mã Cổ Phiếu", f"{data['Ticker'].nunique()} mã")
        col2.metric("Tổng Số Phiên Giao Dịch", f"{len(all_dates):,} phiên")
        col3.metric("Benchmark Thị Trường", benchmark_ticker)
        col4.metric("Khoảng Thời Gian", f"{all_dates[0].strftime('%d/%m/%Y')} - {all_dates[-1].strftime('%d/%m/%Y')}")

        st.markdown("### Phân Chia Chuỗi Thời Gian Chống Rò Rỉ Dữ Liệu (Lookahead-Free)")
        split_df = pd.DataFrame([
            {"Giai đoạn": "Core Train", "Bắt đầu": all_dates[0].strftime("%d/%m/%Y"), "Kết thúc": core_end.strftime("%d/%m/%Y"), "Số phiên": int((all_dates <= core_end).sum()), "Mục đích": "Tối ưu hóa ngưỡng BUY/EXIT chung"},
            {"Giai đoạn": "Validation", "Bắt đầu": val_start.strftime("%d/%m/%Y"), "Kết thúc": val_end.strftime("%d/%m/%Y"), "Số phiên": int(((all_dates >= val_start) & (all_dates <= val_end)).sum()), "Mục đích": "Nested Walk-Forward so sánh các mô hình và khóa Champion"},
            {"Giai đoạn": "Final Test (OOS)", "Bắt đầu": test_start.strftime("%d/%m/%Y"), "Kết thúc": test_end.strftime("%d/%m/%Y"), "Số phiên": int((all_dates >= test_start).sum()), "Mục đích": "Kiểm định thực chiến ngoài mẫu hoàn toàn độc lập"}
        ])
        st.dataframe(split_df, use_container_width=True, hide_index=True)

        st.markdown("### Bảng Kiểm Tra Chất Lượng Dữ Liệu Từng Mã (Top 15)")
        dq = data_quality_table(data)
        st.dataframe(dq.head(15), use_container_width=True, hide_index=True)

    # Nếu người dùng chưa bấm nút chạy, hiển thị hướng dẫn
    if not run_btn and "champion" not in st.session_state:
        st.info("💡 Hãy kiểm tra cấu hình tại thanh bên trái và bấm nút **'🚀 BẮT ĐẦU CHẠY KIỂM ĐỊNH'** để hệ thống tiến hành tính toán toàn bộ pipeline.")
        return

    # ======================================================
    # TIẾN HÀNH CHẠY PIPELINE HOSE V4
    # ======================================================
    with st.spinner("Đang chạy mô phỏng Walk-Forward, tối ưu hóa danh mục và kiểm định..."):
        # 1. Tối ưu hoặc gán ngưỡng
        core_universe = build_universe(price_dict, core_end, benchmark_ticker)
        core_candidates = core_universe.loc[core_universe["Candidate"], "Ticker"].tolist()

        if opt_mode == "Tối ưu tự động trên Train":
            best_buy, best_exit, threshold_trials = optimize_core_thresholds(
                tuple(feature_store.keys()), feature_store, core_candidates, tuple(all_dates), core_end
            )
        else:
            best_buy, best_exit = manual_buy, manual_exit
            threshold_trials = pd.DataFrame([{
                "BuyThreshold": best_buy, "ExitThreshold": best_exit, "GlobalCVScore": 1.0, "NStocks": len(core_candidates)
            }])

        # 2. Validation Selection & Evaluation
        val_selection, val_details = build_selection_schedule(
            feature_store, price_dict, benchmark_ticker, feature_store[benchmark_ticker],
            calendar=all_dates, start=val_start, end=val_end,
            b=best_buy, e=best_exit, top_n=top_n, freq=selection_freq
        )

        regime_val = build_market_regime(feature_store[benchmark_ticker])
        union_val_tickers = sorted({
            val_selection.loc[i, f"Top{j+1}"]
            for i in val_selection.index for j in range(top_n)
            if pd.notna(val_selection.loc[i, f"Top{j+1}"])
        })
        masks_val = build_stock_open_masks(feature_store, union_val_tickers, best_buy, best_exit)

        bnav_val = benchmark_nav(price_dict[benchmark_ticker], val_start, val_end, initial_cash=initial_cash)
        bstats_val = metrics_from_equity(bnav_val)

        val_rows, val_curves = [], {"BENCHMARK": bnav_val}
        for name, spec in ACTIVE_MODEL_SPECS.items():
            sched, _ = build_dynamic_weight_schedule(
                price_dict, val_selection, all_dates, val_start, val_end,
                spec["family"], allocation_freq=allocation_freq, top_n=top_n
            )
            st_res, eq, _ = run_portfolio_engine_dynamic(
                price_dict, sched, all_dates, val_start, val_end,
                market_regime_open=regime_val["RiskOn_Open"],
                stock_active_open=masks_val,
                use_market_regime=(use_market_regime and spec["regime"]),
                use_stock_mask=(use_stock_mask and spec["mask"]),
                initial_cash=initial_cash, commission=commission, slippage=slippage
            )
            st_res["Model"] = name
            st_res["ValidationScore"] = model_validation_score(st_res, bstats_val)
            val_rows.append(st_res)
            val_curves[name] = eq

        bstats_val.update({"Model": "BENCHMARK", "ValidationScore": np.nan})
        val_rows.append(bstats_val)
        val_results = pd.DataFrame(val_rows).set_index("Model")
        champion = val_results.drop(index="BENCHMARK")["ValidationScore"].idxmax()
        st.session_state["champion"] = champion

        # 3. Final Test (Out-of-sample)
        test_selection, test_details = build_selection_schedule(
            feature_store, price_dict, benchmark_ticker, feature_store[benchmark_ticker],
            calendar=all_dates, start=test_start, end=test_end,
            b=best_buy, e=best_exit, top_n=top_n, freq=selection_freq
        )

        regime_test = build_market_regime(feature_store[benchmark_ticker])
        union_test_tickers = sorted({
            test_selection.loc[i, f"Top{j+1}"]
            for i in test_selection.index for j in range(top_n)
            if pd.notna(test_selection.loc[i, f"Top{j+1}"])
        })
        masks_test = build_stock_open_masks(feature_store, union_test_tickers, best_buy, best_exit)

        bnav_test = benchmark_nav(price_dict[benchmark_ticker], test_start, test_end, initial_cash=initial_cash)
        test_rows = []
        test_curves = {"VNINDEX": bnav_test}
        test_order_map = {}

        # a) VNINDEX
        bstats_test = metrics_from_equity(bnav_test)
        bstats_test["Model"] = "VNINDEX"
        test_rows.append(bstats_test)

        # b) Buy & Hold Top N (STATIC_EW_BH)
        top_start = selected_tickers_asof(test_selection, test_start, top_n=top_n)
        bh_sched = pd.DataFrame([equal_weights(top_start)], index=[pd.Timestamp(test_start)])
        bh_stats, bh_eq, bh_od = run_portfolio_engine_dynamic(
            price_dict, bh_sched, all_dates, test_start, test_end,
            market_regime_open=None, stock_active_open=None,
            use_market_regime=False, use_stock_mask=False,
            initial_cash=initial_cash, commission=commission, slippage=slippage
        )
        bh_stats["Model"] = "STATIC_EW_BH"
        test_rows.append(bh_stats)
        test_curves["STATIC_EW_BH"] = bh_eq
        test_order_map["STATIC_EW_BH"] = bh_od

        # c) DYN_EW_NO_REGIME
        ew_sched, _ = build_dynamic_weight_schedule(
            price_dict, test_selection, all_dates, test_start, test_end, "EW",
            allocation_freq=allocation_freq, top_n=top_n
        )
        ew_stats, ew_eq, ew_od = run_portfolio_engine_dynamic(
            price_dict, ew_sched, all_dates, test_start, test_end,
            market_regime_open=None, stock_active_open=None,
            use_market_regime=False, use_stock_mask=False,
            initial_cash=initial_cash, commission=commission, slippage=slippage
        )
        ew_stats["Model"] = "DYN_EW_NO_REGIME"
        test_rows.append(ew_stats)
        test_curves["DYN_EW_NO_REGIME"] = ew_eq
        test_order_map["DYN_EW_NO_REGIME"] = ew_od

        # d) Locked Champion & Các Model Active khác
        for name, spec in ACTIVE_MODEL_SPECS.items():
            sched, _ = build_dynamic_weight_schedule(
                price_dict, test_selection, all_dates, test_start, test_end,
                spec["family"], allocation_freq=allocation_freq, top_n=top_n
            )
            st_res, eq, od = run_portfolio_engine_dynamic(
                price_dict, sched, all_dates, test_start, test_end,
                market_regime_open=regime_test["RiskOn_Open"],
                stock_active_open=masks_test,
                use_market_regime=use_market_regime,
                use_stock_mask=use_stock_mask,
                initial_cash=initial_cash, commission=commission, slippage=slippage
            )
            label = "LOCKED_CHAMPION" if name == champion else "DIAG_" + name
            st_res["Model"] = label
            st_res["OriginalModel"] = name
            test_rows.append(st_res)
            test_curves[label] = eq
            test_order_map[label] = od

        test_results = pd.DataFrame(test_rows).set_index("Model")
        if "STATIC_EW_BH" in test_results.index:
            test_results["ExcessCAGR_vs_StaticEW [pp]"] = test_results["CAGR [%]"] - test_results.loc["STATIC_EW_BH", "CAGR [%]"]
            test_results["ExcessSharpe_vs_StaticEW"] = test_results["Sharpe"] - test_results.loc["STATIC_EW_BH", "Sharpe"]
        if "VNINDEX" in test_results.index:
            test_results["ExcessCAGR_vs_Benchmark [pp]"] = test_results["CAGR [%]"] - test_results.loc["VNINDEX", "CAGR [%]"]
            test_results["DD_Improvement_vs_Benchmark [pp]"] = test_results["MaxDD [%]"] - test_results.loc["VNINDEX", "MaxDD [%]"]

    # ======================================================
    # TAB 2: TỐI ƯU HÓA NGƯỠNG TA (TRAIN)
    # ======================================================
    with tabs[1]:
        st.subheader("Kết Quả Lựa Chọn Ngưỡng BUY / EXIT Chung Trên Core Train")
        c1, c2 = st.columns(2)
        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ngưỡng BUY Tối Ưu (Vào Lệnh)</div>
            <div class="metric-value" style="color: #10B981;">TA Score ≥ {best_buy}</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ngưỡng EXIT Tối Ưu (Thoát Lệnh)</div>
            <div class="metric-value" style="color: #EF4444;">TA Score ≤ {best_exit}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Bảng Kết Quả Cross-Validation Các Cặp Ngưỡng:")
        st.dataframe(threshold_trials.style.format({
            "GlobalCVScore": "{:.3f}",
            "MedianSharpe": "{:.2f}",
            "SharpeStd": "{:.2f}",
            "MedianReturnPct": "{:.1f}%",
            "MedianMaxDDPct": "{:.1f}%"
        }), use_container_width=True)

    # ======================================================
    # TAB 3: ĐẤU TRƯỜNG VALIDATION
    # ======================================================
    with tabs[2]:
        st.subheader("Khảo Sát & Khóa Champion Model Trên Giai Đoạn Validation")
        st.success(f"🏆 Mô hình chiến thắng đã được khóa từ Validation: **{champion}** (Không retune trên tập Test)")

        st.markdown("### Bảng Xếp Hạng Các Mô Hình Phân Bổ (Validation Score):")
        cols_val_show = ["Return [%]", "CAGR [%]", "Sharpe", "Sortino", "MaxDD [%]", "Turnover [%]", "ValidationScore"]
        st.dataframe(val_results[[c for c in cols_val_show if c in val_results.columns]].sort_values(
            "ValidationScore", ascending=False
        ).style.format({
            "Return [%]": "{:.1f}%", "CAGR [%]": "{:.1f}%", "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}", "MaxDD [%]": "{:.1f}%", "Turnover [%]": "{:.1f}%", "ValidationScore": "{:.3f}"
        }), use_container_width=True)

        # Plotly chart Validation NAV
        fig_val = go.Figure()
        for k, v in val_curves.items():
            norm_nav = v / v.iloc[0]
            width = 3 if k == champion else (1.5 if k == "BENCHMARK" else 1.2)
            dash = "dash" if k == "BENCHMARK" else "solid"
            fig_val.add_trace(go.Scatter(x=norm_nav.index, y=norm_nav, mode='lines', name=k, line=dict(width=width, dash=dash)))
        fig_val.update_layout(
            title="Đường Tích Lũy NAV Thời Kỳ Validation (Chuẩn hóa về 1.0)",
            xaxis_title="Thời gian", yaxis_title="Tỷ lệ NAV",
            hovermode="x unified", template="plotly_white"
        )
        st.plotly_chart(fig_val, use_container_width=True)

    # ======================================================
    # TAB 4: KIỂM ĐỊNH THỰC CHIẾN (FINAL TEST)
    # ======================================================
    with tabs[3]:
        st.subheader("🏆 Kết Quả Kiểm Định Thực Chiến Ngoài Mẫu (Out-of-Sample Final Test)")
        st.markdown(f"Giai đoạn kiểm định hoàn toàn độc lập từ **{test_start.strftime('%d/%m/%Y')}** đến **{test_end.strftime('%d/%m/%Y')}**.")

        champ_stats = test_results.loc["LOCKED_CHAMPION"] if "LOCKED_CHAMPION" in test_results.index else {}
        bench_stats = test_results.loc["VNINDEX"] if "VNINDEX" in test_results.index else {}
        bh_stats_test = test_results.loc["STATIC_EW_BH"] if "STATIC_EW_BH" in test_results.index else {}

        # KPI Metrics Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Lợi Nhuận Tích Lũy", f"{champ_stats.get('Return [%]', 0):.1f}%", f"{champ_stats.get('Return [%]', 0) - bench_stats.get('Return [%]', 0):.1f}% vs VNINDEX")
        k2.metric("CAGR (Tăng trưởng kép)", f"{champ_stats.get('CAGR [%]', 0):.1f}%", f"{champ_stats.get('ExcessCAGR_vs_Benchmark [pp]', 0):.1f} pp Alpha")
        k3.metric("Sharpe Ratio", f"{champ_stats.get('Sharpe', 0):.2f}", f"{champ_stats.get('Sharpe', 0) - bench_stats.get('Sharpe', 0):.2f}")
        k4.metric("Max Drawdown (Sụt giảm)", f"{champ_stats.get('MaxDD [%]', 0):.1f}%", f"{champ_stats.get('DD_Improvement_vs_Benchmark [pp]', 0):.1f} pp An toàn hơn")
        k5.metric("Tỷ Lệ Giữ Tiền Mặt TB", f"{champ_stats.get('AvgCash [%]', 0):.1f}%", f"Exposure: {champ_stats.get('AvgExposure [%]', 0):.1f}%")

        # Biểu đồ NAV tương tác bằng Plotly
        st.markdown("### Biểu Đồ Đường Tích Lũy NAV Thực Chiến")
        fig_test = go.Figure()
        
        # Thêm VNINDEX
        if "VNINDEX" in test_curves:
            v_b = test_curves["VNINDEX"] / test_curves["VNINDEX"].iloc[0]
            fig_test.add_trace(go.Scatter(x=v_b.index, y=v_b, mode='lines', name='📊 VNINDEX', line=dict(color='#000000', width=2, dash='dot')))

        # Thêm Buy & Hold
        if "STATIC_EW_BH" in test_curves:
            v_bh = test_curves["STATIC_EW_BH"] / test_curves["STATIC_EW_BH"].iloc[0]
            fig_test.add_trace(go.Scatter(x=v_bh.index, y=v_bh, mode='lines', name='📌 STATIC_EW_BH (Buy & Hold)', line=dict(color='#2563EB', width=2, dash='dash')))

        # Thêm DYN_EW_NO_REGIME
        if "DYN_EW_NO_REGIME" in test_curves:
            v_nr = test_curves["DYN_EW_NO_REGIME"] / test_curves["DYN_EW_NO_REGIME"].iloc[0]
            fig_test.add_trace(go.Scatter(x=v_nr.index, y=v_nr, mode='lines', name='⚡ DYN_EW_NO_REGIME (Không lọc thị trường)', line=dict(color='#F59E0B', width=1.8)))

        # Thêm LOCKED_CHAMPION
        if "LOCKED_CHAMPION" in test_curves:
            v_ch = test_curves["LOCKED_CHAMPION"] / test_curves["LOCKED_CHAMPION"].iloc[0]
            fig_test.add_trace(go.Scatter(x=v_ch.index, y=v_ch, mode='lines', name=f'🏆 LOCKED_CHAMPION ({champion})', line=dict(color='#10B981', width=3)))

        fig_test.add_hline(y=1.0, line_dash="dot", line_color="red", line_width=1)
        fig_test.update_layout(
            title=f"Hiệu Suất Thực Tế: Champion ({champion}) vs VNINDEX vs Buy & Hold",
            xaxis_title="Thời gian", yaxis_title="Tỷ lệ NAV Chuẩn hóa",
            hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_test, use_container_width=True)

        # Biểu đồ Drawdown
        st.markdown("### Mức Sụt Giảm Tài Sản Theo Thời Gian (Drawdown Profile)")
        fig_dd = go.Figure()
        for k in ["VNINDEX", "STATIC_EW_BH", "LOCKED_CHAMPION"]:
            if k in test_curves:
                nav = test_curves[k]
                dd = (nav / nav.cummax() - 1) * 100
                fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, mode='lines', name=k, fill='tozeroy' if k == "LOCKED_CHAMPION" else None))
        fig_dd.update_layout(
            title="So Sánh Drawdown (%) Qua Từng Giai Đoạn",
            xaxis_title="Thời gian", yaxis_title="Sụt giảm (%)",
            hovermode="x unified", template="plotly_white"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Bảng Tổng Kết Chỉ Số Chi Tiết
        st.markdown("### Bảng So Sánh Chỉ Số Hiệu Suất Toàn Diện")
        cols_summary = [
            "Return [%]", "CAGR [%]", "Sharpe", "Sortino", "MaxDD [%]", "Calmar",
            "Turnover [%]", "AvgExposure [%]", "AvgCash [%]",
            "ExcessCAGR_vs_Benchmark [pp]", "DD_Improvement_vs_Benchmark [pp]"
        ]
        st.dataframe(test_results[[c for c in cols_summary if c in test_results.columns]].style.format({
            "Return [%]": "{:.1f}%", "CAGR [%]": "{:.1f}%", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "MaxDD [%]": "{:.1f}%", "Calmar": "{:.2f}", "Turnover [%]": "{:.1f}%",
            "AvgExposure [%]": "{:.1f}%", "AvgCash [%]": "{:.1f}%",
            "ExcessCAGR_vs_Benchmark [pp]": "{:+.1f} pp", "DD_Improvement_vs_Benchmark [pp]": "{:+.1f} pp"
        }), use_container_width=True)

    # ======================================================
    # TAB 5: LỊCH SỬ DANH MỤC & LỆNH
    # ======================================================
    with tabs[4]:
        st.subheader("Lịch Chọn Cổ Phiếu Từng Kỳ & Sổ Lệnh Chi Tiết")

        st.markdown(f"### Danh Mục Top {top_n} Cổ Phiếu Trong Giai Đoạn Final Test (Mỗi Kỳ Rebalance)")
        disp_selection = test_selection.copy()
        disp_selection["EffectiveDate"] = disp_selection["EffectiveDate"].dt.strftime("%d/%m/%Y")
        disp_selection["AsOfDate"] = disp_selection["AsOfDate"].dt.strftime("%d/%m/%Y")
        st.dataframe(disp_selection, use_container_width=True, hide_index=True)

        st.markdown("### Lịch Sử Khớp Lệnh Mô Phỏng (LOCKED_CHAMPION)")
        champ_orders = test_order_map.get("LOCKED_CHAMPION", pd.DataFrame())
        if not champ_orders.empty:
            st.dataframe(champ_orders.style.format({
                "Shares": "{:,.0f}",
                "ExecPrice": "{:,.1f}",
                "Fee": "{:,.0f}"
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có lệnh giao dịch nào được ghi nhận.")

    # ======================================================
    # TAB 6: LEAKAGE AUDIT & XUẤT BÁO CÁO
    # ======================================================
    with tabs[5]:
        st.subheader("Kiểm Toán Rò Rỉ Dữ Liệu (Anti-Leakage Audit) & Tải Báo Cáo")

        leakage = pd.DataFrame([
            ["1. TA Score BUY/EXIT", core_end.strftime("%d/%m/%Y"), False, "Chỉ tối ưu trên Core Train, khóa cố định cho Validation và Test"],
            ["2. Model Family Selection", val_end.strftime("%d/%m/%Y"), False, "Khóa Champion trên Validation, tuyệt đối không retune trên Test"],
            ["3. Tái chọn Top N trong Test", "Phiên t-1", False, "Mỗi kỳ tái cơ cấu chỉ sử dụng dữ liệu giá/khối lượng kết thúc tại phiên t-1"],
            ["4. Trọng số MPT/MinVar/HRP", "Phiên t-1", False, "Ma trận hiệp phương sai và kỳ vọng tỷ suất chỉ tính trên cửa sổ quá khứ <= t-1"],
            ["5. Market Regime & Stock Mask", "Close(t) -> Open(t+1)", False, "Tín hiệu sinh từ giá đóng cửa phiên t, chỉ có hiệu lực giao dịch tại mở cửa t+1"],
            ["6. Final Test Performance", test_end.strftime("%d/%m/%Y"), True, "Chỉ dùng để đo lường hiệu suất thực tế, không can thiệp thuật toán"]
        ], columns=["Hạng mục kiểm toán", "Mốc thông tin tối đa", "Dùng dữ liệu Test để tinh chỉnh?", "Nguyên tắc tuân thủ"])
        st.dataframe(leakage, use_container_width=True, hide_index=True)

        st.markdown("### 📥 Tải Toàn Bộ Dữ Liệu Báo Cáo Ra Excel")
        
        # Tạo file Excel đa sheet tải về
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            split_df.to_excel(writer, sheet_name="TimeSplits", index=False)
            threshold_trials.to_excel(writer, sheet_name="CoreThresholds", index=False)
            val_results.reset_index().to_excel(writer, sheet_name="ValidationResults", index=False)
            test_results.reset_index().to_excel(writer, sheet_name="FinalTestResults", index=False)
            disp_selection.to_excel(writer, sheet_name="TestSelectionHistory", index=False)
            if not champ_orders.empty:
                champ_orders.to_excel(writer, sheet_name="ChampionOrders", index=False)
            leakage.to_excel(writer, sheet_name="LeakageAudit", index=False)
        output.seek(0)

        st.download_button(
            label="📊 TẢI BÁO CÁO TOÀN DIỆN (.XLSX)",
            data=output,
            file_name=f"HOSE_V4_QUANT_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

if __name__ == "__main__":
    main()
