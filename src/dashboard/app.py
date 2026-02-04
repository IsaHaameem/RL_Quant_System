# src/dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

# --- PAGE CONFIG ---
st.set_page_config(page_title="Regime-Aware Quant System", layout="wide", initial_sidebar_state="expanded")

# --- CSS FOR "ACADEMIC LOOK" ---
st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    h1 { color: #003366; }
    h3 { color: #444; }
    div[data-testid="metric-container"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🤖 Mixture-of-Experts Trading System")
    st.markdown("""
    **Architecture:** `Hierarchical RL (Meta-Controller)` | **Regimes:** `3 (Bear, Sideways, Bull)`
    
    A research-grade system that adapts to non-stationary market conditions using **Regime-Specific Agents**.
    """)
with col2:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)

# --- LOAD DATA ---
@st.cache_data
def load_backtest_results():
    csv_path = os.path.join(config.PROJECT_ROOT, 'backtest_detailed_results.csv')
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # 1. FIX DATE PARSING
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    # 2. FIX REGIME TYPE (FLOAT -> INT)
    if 'Regime' in df.columns:
        # Drop any rows where Regime is NaN (prevents int casting errors)
        df = df.dropna(subset=['Regime'])
        df['Regime'] = df['Regime'].astype(int)
        
    return df

df_raw = load_backtest_results()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Simulation Settings")

# 1. Custom Initial Capital
custom_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, step=1000, min_value=1000)

# 2. Date Range Filter (Crisis Zoom)
if df_raw is not None:
    min_date = df_raw['Date'].min().date()
    max_date = df_raw['Date'].max().date()
    
    st.sidebar.subheader("📅 Crisis Zoom")
    date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
else:
    date_range = []

# --- MAIN LOGIC ---
if df_raw is None:
    st.error("⚠️ No Backtest Data Found. Please run `python src/backtesting/backtest.py` first.")
else:
    # --- FILTER DATA BASED ON DATE RANGE ---
    if len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df_raw[(df_raw['Date'] >= start_d) & (df_raw['Date'] <= end_d)].copy()
    else:
        df = df_raw.copy()

    # --- RECALCULATE EQUITY CURVE BASED ON CUSTOM CAPITAL ---
    if len(df) > 0:
        original_start = df['Net_Worth'].iloc[0]
        df['Net_Worth'] = (df['Net_Worth'] / original_start) * custom_capital
        
        # --- SYSTEM HEALTH PANEL ---
        final_val = df['Net_Worth'].iloc[-1]
        total_ret = (final_val - custom_capital) / custom_capital * 100
        
        # Drawdown Calculation
        rolling_max = df['Net_Worth'].cummax()
        drawdown = (df['Net_Worth'] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # Win Rate (Days positive)
        daily_rets = df['Net_Worth'].pct_change().fillna(0)
        win_rate = (daily_rets > 0).mean() * 100
        
        # Confidence
        avg_conf = df['Confidence'].mean()

        st.markdown("### 🛡️ System Health & Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Final Portfolio", f"${final_val:,.2f}", f"{total_ret:.2f}%")
        m2.metric("📉 Max Drawdown", f"{max_dd:.2f}%", help="Maximum loss from a peak. Lower is better.")
        m3.metric("🎯 Win Rate (Daily)", f"{win_rate:.1f}%", help="% of days ending in profit")
        m4.metric("🧠 Avg Confidence", f"{avg_conf:.2f}", help="HMM Model Certainty")

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📈 Performance & Regimes", "🧠 Logic & Transition", "📊 Regime Statistics"])

        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            row_heights=[0.7, 0.3], subplot_titles=("Equity Curve", "Market Regimes"))
            
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Net_Worth'], name='AI Portfolio', 
                                    line=dict(color='#0052CC', width=2)), row=1, col=1)
            
            colors = ['red', 'orange', 'green']
            regime_labels = ['Crash/Bear', 'Sideways', 'Bull/Trend']
            
            for r in range(3):
                mask = df['Regime'] == r
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=df[mask]['Date'], 
                        y=df[mask]['Net_Worth'],
                        mode='markers',
                        name=regime_labels[r],
                        marker=dict(color=colors[r], size=4, opacity=0.6)
                    ), row=1, col=1)

            fig.add_trace(go.Scatter(x=df['Date'], y=df['Confidence'], name='HMM Confidence',
                                    line=dict(color='purple', width=1), fill='tozeroy'), row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True) # use_container_width=True is deprecated but works; warnings are annoying but harmless.

        with tab2:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### 🔥 Regime Transition Heatmap")
                st.info("Visualizes how often the market switches between states.")
                
                transitions = pd.crosstab(
                    df['Regime'], 
                    df['Regime'].shift(-1), 
                    normalize='index'
                )
                
                # FIX: Force Integer Conversion inside the list comprehension
                # This guarantees it works even if pandas sees "1.0"
                x_labels = [regime_labels[int(i)] for i in transitions.columns]
                y_labels = [regime_labels[int(i)] for i in transitions.index]
                
                fig_heat = px.imshow(
                    transitions, 
                    labels=dict(x="To Regime", y="From Regime", color="Probability"),
                    x=x_labels,
                    y=y_labels,
                    color_continuous_scale="Blues",
                    text_auto='.2f'
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
            with c2:
                st.markdown("#### ⚖️ Confidence vs Exposure")
                st.info("Does the agent deleverage when uncertain?")
                
                fig_scat = px.scatter(
                    df, 
                    x="Confidence", 
                    y=df['Action'].abs(), 
                    color="Regime",
                    color_continuous_scale=["red", "orange", "green"],
                    labels={"y": "Abs Position Size (Leverage)"},
                    title="Risk Scaling Validation"
                )
                st.plotly_chart(fig_scat, use_container_width=True)

        with tab3:
            st.markdown("#### 🏆 Specialist Performance Breakdown")
            
            stats = df.groupby('Regime').agg({
                'Net_Worth': lambda x: (x.iloc[-1] - x.iloc[0]), 
                'Action': 'mean',
                'Confidence': 'mean',
                'Date': 'count'
            })
            
            stats.columns = ['Approx PnL ($)', 'Avg Exposure', 'Avg Confidence', 'Days Active']
            
            idx_map = {0: 'Regime 0 (Bear)', 1: 'Regime 1 (Sideways)', 2: 'Regime 2 (Bull)'}
            stats.index = [idx_map.get(int(i), str(i)) for i in stats.index]
            
            st.table(stats)
    else:
        st.warning("No data found for the selected date range.")