import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ensemble Predictor", page_icon="▲", layout="wide")

# ─── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

header[data-testid="stHeader"],
#MainMenu, .stDeployButton, footer { display: none !important; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c14 !important;
    color: #d4d8e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,177,90,0.07) 0%, transparent 70%),
        linear-gradient(180deg, #080c14 0%, #0b1020 100%) !important;
}

.block-container {
    padding: 2.5rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0c1120 !important;
    border-right: 1px solid #1c2236 !important;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.25rem !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    font-family: 'DM Sans', sans-serif !important;
    color: #8892a4 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #111827 !important;
    border: 1px solid #1e2d48 !important;
    border-radius: 6px !important;
    color: #e8c97d !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s ease;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #e8c97d !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(232,201,125,0.08) !important;
}
[data-testid="stSidebar"] hr { border-color: #1c2236 !important; margin: 1.5rem 0 !important; }

/* ── Page Header ── */
.page-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2.5rem;
    padding-bottom: 1.75rem;
    border-bottom: 1px solid #1c2236;
}
.header-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #e8c97d;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.15rem;
    font-weight: 800;
    color: #eef0f4;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.header-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #5a6478;
    margin-top: 0.5rem;
    font-weight: 300;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 99px;
    padding: 0.4rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: #34d399;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 6px #34d399;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #0e1525;
    border: 1px solid #1c2236;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232,201,125,0.3), transparent);
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.55rem;
    font-weight: 500;
    color: #eef0f4;
    letter-spacing: -0.01em;
    line-height: 1;
}
.metric-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    margin-top: 0.4rem;
}
.delta-pos { color: #34d399; }
.delta-neg { color: #f87171; }

/* ── Performance Cards ── */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.perf-card {
    background: #0e1525;
    border: 1px solid #1c2236;
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
    text-align: center;
}
.perf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3d4d66;
    margin-bottom: 0.6rem;
}
.perf-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.7rem;
    font-weight: 500;
    color: #e8c97d;
}

/* ── Signal Card ── */
.signal-wrapper {
    display: flex;
    justify-content: center;
    padding: 3rem 1rem;
}
.signal-card {
    background: #0e1525;
    border: 1px solid #1c2236;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    text-align: center;
    width: 100%;
    max-width: 480px;
    position: relative;
    overflow: hidden;
}
.signal-card.bullish {
    border-color: rgba(52,211,153,0.25);
    box-shadow: 0 0 60px rgba(52,211,153,0.06);
}
.signal-card.bullish::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(52,211,153,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.signal-card.bearish {
    border-color: rgba(248,113,113,0.25);
    box-shadow: 0 0 60px rgba(248,113,113,0.06);
}
.signal-card.bearish::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(248,113,113,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.signal-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.signal-type {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 0.4rem;
}
.signal-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.signal-heading.bullish { color: #34d399; }
.signal-heading.bearish { color: #f87171; }
.signal-desc {
    font-size: 0.82rem;
    color: #4a5568;
    font-weight: 300;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3d4d66;
    margin-bottom: 0.75rem;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1c2236 20%, #1c2236 80%, transparent);
    margin: 2rem 0;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid #1c2236 !important;
    padding: 0 !important;
    margin-bottom: 2rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.25rem !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #3d4d66 !important;
    border-radius: 0 !important;
    margin-bottom: -1px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: #8892a4 !important; background: transparent !important; }
[data-testid="stTabs"] [aria-selected="true"] { color: #e8c97d !important; border-bottom-color: #e8c97d !important; background: transparent !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1c2236; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
            color:#c8cdd8;letter-spacing:-0.01em;margin-bottom:0.25rem;">
  Ensemble Predictor
</div>
<div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.18em;
            color:#3d4d66;text-transform:uppercase;margin-bottom:1.5rem;">
  Configuration
</div>
""", unsafe_allow_html=True)

ticker = st.sidebar.text_input("Ticker Symbol", "AAPL", help="Examples: AAPL, TSLA, RELIANCE.NS")
days_history = st.sidebar.slider("Historical Range (Days)", 365, 1825, 1095)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.1em;
            color:#2a3550;text-align:center;padding-top:0.5rem;">
  DSN2099 · Exhibition II
</div>
""", unsafe_allow_html=True)


# ─── Data & Feature Engineering ───────────────────────────────────────────────
@st.cache_data
def load_and_preprocess_data(ticker, days):
    import yfinance as yf
    import numpy as np

    symbol = "₹" if ticker.upper().endswith((".NS", ".BO")) else "$"

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{days}d")

        # 🔥 Fallback if API fails or empty data
        if df is None or df.empty:
            st.warning("Using default data (AAPL)")
            stock = yf.Ticker("AAPL")
            df = stock.history(period="365d")

    except Exception:
        # 🔥 Fallback if error occurs
        st.warning("API issue, loading default data (AAPL)")
        stock = yf.Ticker("AAPL")
        df = stock.history(period="365d")

    # --- Basic columns ---
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # --- Feature Engineering ---
    df['5MA'] = df['Close'].rolling(window=5).mean()
    df['10MA'] = df['Close'].rolling(window=10).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range'] = df['High'] - df['Low']
    df['Volatility'] = df['Close'].rolling(window=5).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA'] + (df['BB_std'] * 2)
    df['BB_Lower'] = df['BB_MA'] - (df['BB_std'] * 2)

    # Target
    df['Price_Change'] = (
    0.2 * ((df['Open'].shift(-1) - df['Open']) / df['Open']) +
    0.4 * ((df['Close'].shift(-1) - df['Close']) / df['Close']) +
    0.2 * ((df['High'].shift(-1) - df['High']) / df['High']) +
    0.2 * ((df['Low'].shift(-1) - df['Low']) / df['Low'])
)

    df['Target'] = np.where(df['Price_Change'] > 0.01, 1, 0)

    return df.dropna(), symbol


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div>
    <div class="header-eyebrow">Multi-Model ML · Market Analysis</div>
    <div class="header-title">{ticker.upper()}&nbsp;&nbsp;Trend Predictor</div>
    <div class="header-sub">Ensemble of Random Forest, Gradient Boosting, AdaBoost &amp; Logistic Regression</div>
  </div>
  <div>
    <div class="status-badge">
      <span class="status-dot"></span>Models Online
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Data Load ────────────────────────────────────────────────────────────────
with st.spinner("Loading market data…"):
    data, symbol = load_and_preprocess_data(ticker, days_history)

if data is None:
    st.error("Invalid ticker or connection issue. Check the symbol and try again.")
    st.stop()

# --- ML Pipeline ---

features = [
    'Open', 'High', 'Low', 'Close', 'Volume', '5MA', '10MA',
    'Daily_Return', 'Price_Range', 'Volatility', 'RSI',
    'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower'
]

X = data[features]
y = data['Target']

# Time-based split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# --- Models (with balancing where possible) ---
rf  = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
lr  = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# --- Ensemble (FIXED weights for stability) ---
ensemble = VotingClassifier(
    estimators=[('RF', rf), ('GB', gb), ('ADA', ada), ('LR', lr)],
    voting='soft',
    weights=[2, 3, 2, 2]   # stable + slightly favor GB
)

# --- Train models ---
rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)
ada.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)
ensemble.fit(X_train_scaled, y_train)

# ─── Shared Plotly theme ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#4a5568", size=11),
    margin=dict(t=20, b=10, l=0, r=0),
    xaxis=dict(gridcolor="#111827", linecolor="#1c2236"),
    yaxis=dict(gridcolor="#111827", linecolor="#1c2236"),
)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Market Dashboard", "Trading Signal", "Model Performance", "AI Insights"])


# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    daily_pct = data['Daily_Return'].iloc[-1] * 100
    dc = "delta-pos" if daily_pct >= 0 else "delta-neg"
    da = "▲" if daily_pct >= 0 else "▼"
    rsi_val = data['RSI'].iloc[-1]
    rsi_label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral zone")

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Latest Close</div>
        <div class="metric-value">{symbol}{data['Close'].iloc[-1]:,.2f}</div>
        <div class="metric-delta {dc}">{da} {abs(daily_pct):.2f}% today</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">RSI · 14-Day</div>
        <div class="metric-value">{rsi_val:.1f}</div>
        <div class="metric-delta" style="color:#3d4d66;">{rsi_label}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">MACD</div>
        <div class="metric-value">{data['MACD'].iloc[-1]:.3f}</div>
        <div class="metric-delta" style="color:#3d4d66;">Signal {data['Signal_Line'].iloc[-1]:.3f}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Volatility · 5-Day</div>
        <div class="metric-value">{data['Volatility'].iloc[-1]:.2f}</div>
        <div class="metric-delta" style="color:#3d4d66;">Rolling std dev</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Price History</div>', unsafe_allow_html=True)
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines',
        line=dict(color='#e8c97d', width=1.5), fill='tozeroy',
        fillcolor='rgba(232,201,125,0.04)', name='Close'))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['5MA'], mode='lines',
        line=dict(color='#60a5fa', width=1, dash='dot'), name='5-MA'))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['10MA'], mode='lines',
        line=dict(color='#a78bfa', width=1, dash='dot'), name='10-MA'))
    
    # Applied layout fixes
    fig_price.update_layout(PLOT_LAYOUT)
    fig_price.update_layout(height=320,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Bollinger Bands</div>', unsafe_allow_html=True)
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines',
        line=dict(color='rgba(96,165,250,0.3)', width=1), name='Upper Band'))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines',
        line=dict(color='rgba(96,165,250,0.3)', width=1), fill='tonexty',
        fillcolor='rgba(96,165,250,0.04)', name='Lower Band'))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines',
        line=dict(color='#e8c97d', width=1.2), name='Close'))
    
    # Applied layout fixes
    fig_bb.update_layout(PLOT_LAYOUT)
    fig_bb.update_layout(height=240,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_bb, use_container_width=True)


# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    prediction = ensemble.predict(X_scaled[-1].reshape(1, -1))[0]
    is_bull = prediction == 1
    card_cls = "bullish" if is_bull else "bearish"
    icon     = "↑" if is_bull else "↓"
    heading  = "BUY SIGNAL" if is_bull else "SELL / HOLD"
    desc     = "Models project an upward move for the next session." if is_bull else "Models project a downward or flat move for the next session."

    st.markdown(f"""
    <div class="signal-wrapper">
      <div class="signal-card {card_cls}">
        <div class="signal-icon">{icon}</div>
        <div class="signal-type">Next-Day Projection · Ensemble Consensus</div>
        <div class="signal-heading {card_cls}">{heading}</div>
        <div class="signal-desc">{desc}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Individual Model Accuracy</div>', unsafe_allow_html=True)

    acc_dict = {
        "Random Forest":  accuracy_score(y_test, rf.predict(X_test_scaled)),
        "Gradient Boost": accuracy_score(y_test, gb.predict(X_test_scaled)),
        "AdaBoost":       accuracy_score(y_test, ada.predict(X_test_scaled)),
        "Logistic Reg":   accuracy_score(y_test, lr.predict(X_test_scaled)),
        "Ensemble":       accuracy_score(y_test, ensemble.predict(X_test_scaled)),
    }
    acc_df  = pd.DataFrame(list(acc_dict.items()), columns=['Model', 'Accuracy'])
    bar_clr = ['#1e3a5f'] * 4 + ['#e8c97d']

    fig_acc = go.Figure(go.Bar(
        x=acc_df['Model'], y=acc_df['Accuracy'],
        marker=dict(color=bar_clr, line=dict(width=0)),
        text=[f"{v:.1%}" for v in acc_df['Accuracy']],
        textposition='outside',
        textfont=dict(family="DM Mono, monospace", size=11, color="#8892a4"),
    ))
    
    # Applied layout fixes
    fig_acc.update_layout(PLOT_LAYOUT)
    fig_acc.update_layout(height=300,
        yaxis=dict(range=[0, 1.12], tickformat=".0%"),
        bargap=0.35)
    st.plotly_chart(fig_acc, use_container_width=True)


# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    y_pred = ensemble.predict(X_test_scaled)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    st.markdown(f"""
    <div class="perf-grid">
      <div class="perf-card">
        <div class="perf-label">Accuracy</div>
        <div class="perf-value">{acc:.1%}</div>
      </div>
      <div class="perf-card">
        <div class="perf-label">Precision</div>
        <div class="perf-value">{prec:.1%}</div>
      </div>
      <div class="perf-card">
        <div class="perf-label">Recall</div>
        <div class="perf-value">{rec:.1%}</div>
      </div>
      <div class="perf-card">
        <div class="perf-label">F1 Score</div>
        <div class="perf-value">{f1:.1%}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">ROC Curve — Receiver Operating Characteristic</div>', unsafe_allow_html=True)

    fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble.predict_proba(X_test_scaled)[:, 1])
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf.predict_proba(X_test_scaled)[:, 1])
    fpr_gb,  tpr_gb,  _ = roc_curve(y_test, gb.predict_proba(X_test_scaled)[:, 1])
    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline',
        line=dict(color='#1c2236', width=1, dash='dot')))
    fig_roc.add_trace(go.Scatter(x=fpr_rf,  y=tpr_rf,  mode='lines',
        name=f'Random Forest  {auc(fpr_rf, tpr_rf):.2f}',
        line=dict(color='#3b82f6', width=1.5, dash='dash')))
    fig_roc.add_trace(go.Scatter(x=fpr_gb,  y=tpr_gb,  mode='lines',
        name=f'Gradient Boost  {auc(fpr_gb, tpr_gb):.2f}',
        line=dict(color='#a78bfa', width=1.5, dash='dot')))
    fig_roc.add_trace(go.Scatter(x=fpr_lr,  y=tpr_lr,  mode='lines',
        name=f'Logistic Reg  {auc(fpr_lr, tpr_lr):.2f}',
        line=dict(color='#60a5fa', width=1.5, dash='dashdot')))
    fig_roc.add_trace(go.Scatter(x=fpr_ens, y=tpr_ens, mode='lines',
        name=f'Ensemble  {auc(fpr_ens, tpr_ens):.2f}',
        line=dict(color='#e8c97d', width=2.5)))
        
    # Applied layout fixes
    fig_roc.update_layout(PLOT_LAYOUT)
    fig_roc.update_layout(height=380,
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98,
                    font=dict(size=11), bgcolor="rgba(14,21,37,0.8)",
                    bordercolor="#1c2236", borderwidth=1))
    st.plotly_chart(fig_roc, use_container_width=True)


# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-label">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#4a5568;
                font-weight:300;margin-bottom:1.5rem;max-width:520px;line-height:1.6;">
      Relative weight each technical indicator contributes to the Random Forest's
      directional prediction. Higher weight = stronger influence on the model's output.
    </div>
    """, unsafe_allow_html=True)

    importances = rf.feature_importances_
    indices     = np.argsort(importances)
    bar_colors  = ['#e8c97d' if importances[i] == max(importances) else '#1e3a5f' for i in indices]

    fig_feat = go.Figure(go.Bar(
        x=importances[indices],
        y=np.array(features)[indices],
        orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in importances[indices]],
        textposition='outside',
        textfont=dict(family="DM Mono, monospace", size=10, color="#3d4d66"),
    ))
    
    # Applied layout fixes
    fig_feat.update_layout(PLOT_LAYOUT)
    fig_feat.update_layout(height=500,
        xaxis_title="Relative Importance",
        xaxis=dict(tickformat=".2f"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(family="DM Mono, monospace", size=11, color="#8892a4")),
        margin=dict(l=0, r=60, t=10, b=10))
    st.plotly_chart(fig_feat, use_container_width=True)
