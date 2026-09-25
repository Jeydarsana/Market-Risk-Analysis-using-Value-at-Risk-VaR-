import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm, t, linregress
from datetime import datetime, timedelta
import io

# Optional Ollama Import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# --------------------------------------------------
# APP CONFIG & STYLING
# --------------------------------------------------
st.set_page_config(page_title="Finance Lab: Risk Analyzer", layout="wide")
st.title("📈 Portfolio Risk & Performance Analytics")
st.markdown("---")

# --------------------------------------------------
# SIDEBAR - USER INPUTS
# --------------------------------------------------
st.sidebar.header("User Parameters")
ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", "INFY.NS, TCS.NS, RELIANCE.NS")
investment = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=1000)
conf_level = st.sidebar.selectbox("Confidence Level", [0.95, 0.99], index=0)
time_horizon = st.sidebar.slider("Time Horizon (Days)", 1, 30, 1)
rf_rate = st.sidebar.number_input("Risk-Free Rate (Decimal)", 0.0, 0.15, 0.06)
use_benchmark = st.sidebar.checkbox("Use NIFTY 50 Benchmark", value=True)

tickers = [t.strip() for t in ticker_input.split(",")]

# --------------------------------------------------
# CORE LOGIC FUNCTIONS
# --------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_data(tickers, benchmark_flag):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=730)
    data = yf.download(tickers, start=start_date, end=end_date)
    prices = data["Adj Close"] if "Adj Close" in data else data["Close"]
    
    # Handle single ticker
    if len(tickers) == 1:
        prices = prices.to_frame(name=tickers[0])
        
    bench_returns = None
    bench_prices = None
    if benchmark_flag:
        bench_data = yf.download("^NSEI", start=start_date, end=end_date)
        if not bench_data.empty:
            bench_prices = bench_data["Adj Close"] if "Adj Close" in bench_data else bench_data["Close"]
            bench_returns = bench_prices.pct_change().dropna()
            
    return prices, bench_returns, bench_prices

def get_metrics(returns_series, inv_amt, conf, rf):
    mean = returns_series.mean()
    std = returns_series.std()
    
    # Drawdown
    cum_ret = (1 + returns_series).cumprod()
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max
    
    # VaR / CVaR
    hist_var = -np.percentile(returns_series, (1-conf)*100) * inv_amt
    var_cutoff = np.percentile(returns_series, (1-conf)*100)
    cvar = -returns_series[returns_series <= var_cutoff].mean() * inv_amt
    
    # Ratios
    ann_ret = mean * 252
    ann_vol = std * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else 0
    
    return {
        "hist_var": hist_var, "cvar": cvar, "max_dd": drawdown.min(),
        "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "drawdown_series": drawdown
    }

# --------------------------------------------------
# DATA PROCESSING
# --------------------------------------------------

prices, benchmark_returns, benchmark_prices = fetch_data(tickers, use_benchmark)
returns = prices.pct_change().dropna()

# Optimization (Markowitz)
if len(tickers) > 1:
    mean_rets = returns.mean()
    cov_mat = returns.cov()
    num_ports = 2000
    results = np.zeros((3, num_ports))
    weights_record = []
    
    for i in range(num_ports):
        w = np.random.random(len(tickers))
        w /= np.sum(w)
        weights_record.append(w)
        p_ret = np.sum(mean_rets * w) * 252
        p_std = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * np.sqrt(252)
        results[0,i] = p_std
        results[1,i] = p_ret
        results[2,i] = (p_ret - rf_rate) / p_std
    
    max_sharpe_idx = np.argmax(results[2])
    best_weights = weights_record[max_sharpe_idx]
    vol_arr, ret_arr, sharpe_arr = results[0], results[1], results[2]
else:
    best_weights = np.array([1.0])

portfolio_returns = returns.dot(best_weights)
m = get_metrics(portfolio_returns, investment, conf_level, rf_rate)

# --------------------------------------------------
# UI LAYOUT - DASHBOARD
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("1-Day VaR", f"₹{m['hist_var']:,.0f}")
col2.metric("CVaR (Worst Case)", f"₹{m['cvar']:,.0f}")
col3.metric("Annual Return", f"{m['ann_ret']:.2%}")
col4.metric("Max Drawdown", f"{m['max_dd']:.2%}")

# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Trends & Correlation", "Optimization", "Stress Tests", "Risk Distribution"])

with tab1:
    st.subheader("Price Trend vs Benchmark")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for col in prices.columns:
        ax1.plot(prices[col], label=col)
    if use_benchmark and benchmark_prices is not None:
        ax1.plot(benchmark_prices, label='NIFTY 50', linestyle='--', color='black', alpha=0.6)
    ax1.legend()
    st.pyplot(fig1)

    if len(tickers) > 1:
        st.subheader("Asset Correlation")
        fig2, ax2 = plt.subplots()
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

with tab2:
    if len(tickers) > 1:
        st.subheader("Efficient Frontier")
        fig3, ax3 = plt.subplots()
        scatter = ax3.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
        ax3.scatter(vol_arr[max_sharpe_idx], ret_arr[max_sharpe_idx], color='red', marker='*', s=200, label="Optimal")
        plt.colorbar(scatter, label="Sharpe Ratio")
        ax3.set_xlabel("Volatility")
        ax3.set_ylabel("Return")
        st.pyplot(fig3)
    
    st.subheader("Optimal Portfolio Weights")
    weight_df = pd.DataFrame({'Asset': tickers, 'Weight': best_weights})
    st.dataframe(weight_df.style.format({'Weight': '{:.2%}'}))

with tab3:
    st.subheader("Stress Test Scenarios")
    scenarios = {'Mild (-10%)': -0.10, 'Moderate (-20%)': -0.20, 'Crash (-30%)': -0.30, 'Worst Day': portfolio_returns.min()}
    s_losses = {k: abs(v) * investment for k, v in scenarios.items()}
    st.bar_chart(pd.Series(s_losses))

with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Return Distribution")
        fig4, ax4 = plt.subplots()
        ax4.hist(portfolio_returns, bins=50, color='skyblue', edgecolor='black')
        ax4.axvline(-m['hist_var']/investment, color='red', linestyle='--', label='VaR')
        st.pyplot(fig4)
    with col_b:
        st.subheader("Portfolio Drawdown")
        fig5, ax5 = plt.subplots()
        ax5.fill_between(m['drawdown_series'].index, m['drawdown_series'], 0, color='red', alpha=0.3)
        st.pyplot(fig5)

# --------------------------------------------------
# AI CHATBOT SECTION (MODIFIED PART)
# --------------------------------------------------
st.markdown("---")
st.header("🤖 AI Risk Assistant")

# Prepare a structured context for the AI
portfolio_context = f"""
Portfolio Data Context:
- Tickers: {ticker_input}
- Total Investment: ₹{investment:,.2f}
- 1-Day VaR: ₹{m['hist_var']:,.2f}
- Expected Shortfall (CVaR): ₹{m['cvar']:,.2f}
- Annualized Return: {m['ann_ret']:.2%}
- Annualized Volatility: {m['ann_vol']:.2%}
- Sharpe Ratio: {m['sharpe']:.2f}
- Max Drawdown: {m['max_dd']:.2%}
- Benchmark: {'NIFTY 50' if use_benchmark else 'None'}
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your portfolio risk..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if OLLAMA_AVAILABLE:
            try:
                # Use system prompts to anchor the AI's behavior and context
                res = ollama.chat(
                    model='tinyllama',
                    messages=[
                        {'role': 'system', 'content': 'You are a professional Financial Risk Analyst. Provide concise, data-driven advice based on the metrics provided. Be realistic about risks and avoid generic fluff.'},
                        {'role': 'system', 'content': portfolio_context},
                        {'role': 'user', 'content': prompt}
                    ],
                    options={'temperature': 0.3} # Low temperature for more factual responses
                )
                response = res['message']['content']
            except Exception as e:
                response = f"I encountered an error connecting to the AI model. However, I can see your portfolio has an annualized volatility of {m['ann_vol']:.2%}. What specific risk metric can I help explain?"
        else:
            # Enhanced fallback logic without AI
            p_low = prompt.lower()
            if any(x in p_low for x in ["risk", "safe", "danger"]):
                response = f"Based on the data, your 1-day Value at Risk (VaR) is ₹{m['hist_var']:,.2f}. This means there is a {100-conf_level*100:.0f}% chance you could lose more than this amount in a single day."
            elif any(x in p_low for x in ["return", "performance", "profit"]):
                response = f"Your portfolio is showing an annualized return of {m['ann_ret']:.2%}, with a Sharpe Ratio of {m['sharpe']:.2f}."
            else:
                response = "AI mode is offline (Ollama not found). I can provide stats on 'risk' or 'returns' if you ask about them specifically."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --------------------------------------------------
# EXPORT REPORT
# --------------------------------------------------
st.sidebar.markdown("---")
# Clean up the dict for export
export_metrics = {k: v for k, v in m.items() if k != 'drawdown_series'}
report_csv = pd.DataFrame([export_metrics]).to_csv().encode('utf-8')
st.sidebar.download_button("📥 Download Risk Report", data=report_csv, file_name="portfolio_report.csv", mime="text/csv")