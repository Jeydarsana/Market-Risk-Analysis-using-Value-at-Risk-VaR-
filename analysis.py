import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
from scipy.stats import skew, linregress

#  import ollama 
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

tickers = input("Enter stock tickers (e.g., INFY.NS,TCS.NS,RELIANCE.NS): ").split(",")
tickers = [t.strip() for t in tickers]

investment = float(input("Enter investment amount (e.g., 100000): ").replace(",", "").replace("₹", ""))
confidence_level = float(input("Enter confidence level (0.95 or 0.99): "))
time_horizon = int(input("Enter time horizon (days): "))
risk_free_rate = float(input("Enter risk-free rate (as decimal, e.g., 0.06 for 6%): ") or "0.06")

# Market benchmark for Beta/Alpha (NIFTY 50)
use_benchmark = input("Use NIFTY 50 as market benchmark? (y/n): ").lower() == 'y'

# --------------------------------------------------
# FETCH DATA
# --------------------------------------------------

end_date = datetime.today()
start_date = end_date - timedelta(days=730)

print("\nFetching market data...\n")

data = yf.download(tickers, start=start_date, end=end_date)

# Fetch benchmark data (NIFTY 50)
benchmark_returns = None
benchmark_prices = None
if use_benchmark:
    try:
        benchmark_data = yf.download("^NSEI", start=start_date, end=end_date)
        if not benchmark_data.empty:
            benchmark_prices = benchmark_data["Adj Close"] if "Adj Close" in benchmark_data else benchmark_data["Close"]
            benchmark_returns = benchmark_prices.pct_change().dropna()
            print("✓ Benchmark data (NIFTY 50) loaded successfully")
        else:
            print("⚠️ Could not load NIFTY 50 data. Continuing without benchmark.")
            use_benchmark = False
    except Exception as e:
        print(f"⚠️ Error loading benchmark: {e}")
        use_benchmark = False

# Improved validation
if data.empty or len(data.columns) == 0:
    print("\n❌ Invalid ticker(s). Try:")
    print("INFY.NS, TCS.NS, RELIANCE.NS\n")
    exit()

prices = data["Adj Close"] if "Adj Close" in data else data["Close"]

if len(prices.shape) == 1:
    prices = prices.to_frame(name=tickers[0])

# Handle single stock case
single_stock = len(tickers) == 1

# --------------------------------------------------
# GRAPH 1 — PRICE TREND
# --------------------------------------------------

plt.figure(figsize=(12,6))
for ticker in prices.columns:
    plt.plot(prices[ticker], label=ticker, linewidth=1.5)

if use_benchmark and benchmark_prices is not None:
    plt.plot(benchmark_prices, label='NIFTY 50', linewidth=2, linestyle='--', color='black', alpha=0.7)

plt.title("Stock Price Trend vs Benchmark")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --------------------------------------------------
# RETURNS
# --------------------------------------------------

returns = prices.pct_change().dropna()

# Align benchmark returns with portfolio returns
if use_benchmark and benchmark_returns is not None:
    common_idx = returns.index.intersection(benchmark_returns.index)
    if len(common_idx) > 0:
        returns = returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
    else:
        print("⚠️ No overlapping dates with benchmark. Disabling benchmark features.")
        use_benchmark = False

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

if not single_stock:
    plt.figure(figsize=(8,6))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0, 
                xticklabels=tickers, yticklabels=tickers, fmt='.2f')
    plt.title("Asset Correlation Matrix")
    plt.show()

# --------------------------------------------------
# PORTFOLIO OPTIMIZATION (MARKOWITZ)
# --------------------------------------------------

if not single_stock:
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    weight_array = []

    for i in range(num_portfolios):
        weights_rand = np.random.random(len(tickers))
        weights_rand /= np.sum(weights_rand)
        weight_array.append(weights_rand)

        portfolio_return = np.sum(mean_returns * weights_rand) * 252
        portfolio_std = np.sqrt(np.dot(weights_rand.T, np.dot(cov_matrix, weights_rand))) * np.sqrt(252)
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std != 0 else 0

        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = portfolio_sharpe

    volatility_arr = results[0]
    returns_arr = results[1]
    sharpe_arr = results[2]

    # Optimal portfolios
    min_vol_idx = np.argmin(volatility_arr)
    max_return_idx = np.argmax(returns_arr)
    max_sharpe_idx = np.argmax(sharpe_arr)

    min_vol_weights = weight_array[min_vol_idx]
    max_return_weights = weight_array[max_return_idx]
    max_sharpe_weights = weight_array[max_sharpe_idx]
    
    # Use max sharpe portfolio
    weights = max_sharpe_weights
    
else:
    # For single stock, weights = 1.0
    weights = np.array([1.0])
    min_vol_weights = max_return_weights = max_sharpe_weights = weights

# --------------------------------------------------
# PORTFOLIO RETURNS
# --------------------------------------------------

portfolio_returns = returns.dot(weights)
mean = portfolio_returns.mean()
std = portfolio_returns.std()

# --------------------------------------------------
# BETA AND ALPHA CALCULATION
# --------------------------------------------------

def calculate_beta_alpha(portfolio_returns_series, benchmark_returns_series, risk_free_rate_annual):
    """Calculate Beta and Alpha of portfolio"""
    if benchmark_returns_series is None:
        return None, None
    
    try:
        # Align returns
        portfolio_clean = portfolio_returns_series.dropna()
        benchmark_clean = benchmark_returns_series.dropna()
        
        # Align indices
        common_idx = portfolio_clean.index.intersection(benchmark_clean.index)
        if len(common_idx) < 30:
            return None, None
        
        portfolio_aligned = portfolio_clean[common_idx]
        benchmark_aligned = benchmark_clean[common_idx]
        
        # Calculate excess returns
        daily_rf = (1 + risk_free_rate_annual) ** (1/252) - 1
        portfolio_excess = portfolio_aligned - daily_rf
        benchmark_excess = benchmark_aligned - daily_rf
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(benchmark_excess, portfolio_excess)
        
        beta = slope
        alpha_annual = intercept * 252
        
        return beta, alpha_annual
    except Exception as e:
        print(f"⚠️ Could not calculate Beta/Alpha: {e}")
        return None, None

beta, alpha = calculate_beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate)

# --------------------------------------------------
# STRESS TESTING
# --------------------------------------------------

def stress_test(portfolio_returns_series, investment_amt, current_portfolio_value=None):
    """Simulate market crash scenarios"""
    if current_portfolio_value is None:
        current_portfolio_value = investment_amt
    
    scenarios = {
        'Mild Crash (-10%)': -0.10,
        'Moderate Crash (-15%)': -0.15,
        'Severe Crash (-20%)': -0.20,
        'Extreme Crash (-30%)': -0.30,
        '2008 Style Crisis (-40%)': -0.40,
        'High Volatility (2x std)': -2 * portfolio_returns_series.std(),
        'Worst Historical Day': portfolio_returns_series.min()
    }
    
    results = {}
    for scenario_name, shock in scenarios.items():
        if isinstance(shock, (int, float)):
            loss = current_portfolio_value * abs(shock)
            new_value = current_portfolio_value * (1 + shock)
            loss_pct = abs(shock)
        else:
            loss = current_portfolio_value * abs(shock)
            new_value = current_portfolio_value * (1 + shock)
            loss_pct = abs(shock)
        
        results[scenario_name] = {
            'loss': loss,
            'new_value': new_value,
            'loss_pct': loss_pct
        }
    
    return results

stress_results = stress_test(portfolio_returns, investment)

# --------------------------------------------------
# INDIVIDUAL STOCK METRICS
# --------------------------------------------------

def calculate_stock_metrics(stock_returns_series, benchmark_returns_series=None, rf_rate=0.06):
    """Calculate key metrics for individual stocks"""
    try:
        annual_return = stock_returns_series.mean() * 252
        annual_volatility = stock_returns_series.std() * np.sqrt(252)
        sharpe = (annual_return - rf_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Sortino
        downside_returns = stock_returns_series[stock_returns_series < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino = (annual_return - rf_rate) / downside_std if downside_std != 0 else 0
        else:
            sortino = 0
        
        # Max Drawdown
        cumulative = (1 + stock_returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Beta
        beta = None
        if benchmark_returns_series is not None:
            try:
                aligned = pd.DataFrame({
                    'stock': stock_returns_series,
                    'benchmark': benchmark_returns_series
                }).dropna()
                if len(aligned) > 30:
                    slope, _, _, _, _ = linregress(aligned['benchmark'], aligned['stock'])
                    beta = slope
            except:
                beta = None
        
        return {
            'return': annual_return,
            'volatility': annual_volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'beta': beta
        }
    except Exception as e:
        return {
            'return': 0, 'volatility': 0, 'sharpe': 0, 
            'sortino': 0, 'max_drawdown': 0, 'beta': None
        }

# Calculate metrics for each stock
stock_metrics = {}
for ticker in tickers:
    stock_metrics[ticker] = calculate_stock_metrics(returns[ticker], benchmark_returns, risk_free_rate)

# --------------------------------------------------
# ADDITIONAL RISK METRICS
# --------------------------------------------------

def calculate_additional_metrics(returns_series, investment_amt, rf_rate):
    # Maximum Drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sortino Ratio
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (mean * 252 - rf_rate) / downside_std if downside_std != 0 else 0
    else:
        sortino_ratio = 0
    
    # Value at Risk
    historical_var = float(-np.percentile(returns_series, (1-confidence_level)*100) * investment_amt)
    
    z_score = norm.ppf(confidence_level)
    parametric_var = float((z_score * std - mean) * investment_amt)
    
    # Monte Carlo with t-distribution
    df = 5
    simulated_returns_t = np.random.standard_t(df, 10000) * std + mean
    monte_carlo_var_t = float(-np.percentile(simulated_returns_t, (1-confidence_level)*100) * investment_amt)
    
    # Conditional VaR
    var_cutoff = np.percentile(returns_series, (1-confidence_level)*100)
    cvar = float(-returns_series[returns_series <= var_cutoff].mean() * investment_amt)
    
    return {
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'historical_var': historical_var,
        'parametric_var': parametric_var,
        'monte_carlo_var': monte_carlo_var_t,
        'cvar': cvar
    }

metrics = calculate_additional_metrics(portfolio_returns, investment, risk_free_rate)

# --------------------------------------------------
# VAR CONFIDENCE INTERVALS (BOOTSTRAP)
# --------------------------------------------------

n_bootstrap = 1000
bootstrap_var = []

for _ in range(n_bootstrap):
    sample = np.random.choice(portfolio_returns, len(portfolio_returns), replace=True)
    bootstrap_var.append(-np.percentile(sample, (1-confidence_level)*100) * investment)

var_ci_lower = np.percentile(bootstrap_var, 2.5)
var_ci_upper = np.percentile(bootstrap_var, 97.5)

# --------------------------------------------------
# BACKTESTING
# --------------------------------------------------

var_threshold = -metrics['historical_var'] / investment
breaches = portfolio_returns < var_threshold
breach_count = int(breaches.sum())
expected_breaches = len(portfolio_returns) * (1-confidence_level)

# --------------------------------------------------
# RISK ENGINE
# --------------------------------------------------

def final_risk_assessment(var_ratio, cvar_ratio, breach_count_exp, expected_breaches_exp, beta_value=None):
    score = 0

    if var_ratio < 0.03:
        score += 1
    elif var_ratio < 0.07:
        score += 2
    else:
        score += 3

    if cvar_ratio > 1.5 * var_ratio:
        score += 2

    if breach_count_exp > expected_breaches_exp:
        score += 2
        
    if metrics['max_drawdown'] < -0.3:
        score += 2
    elif metrics['max_drawdown'] < -0.15:
        score += 1
    
    if beta_value and beta_value > 1.2:
        score += 2
    elif beta_value and beta_value < 0.8:
        score -= 1

    if score <= 2:
        return "LOW"
    elif score <= 5:
        return "MODERATE"
    else:
        return "HIGH"

var_ratio = metrics['historical_var'] / investment
cvar_ratio = metrics['cvar'] / investment
risk_level = final_risk_assessment(var_ratio, cvar_ratio, breach_count, expected_breaches, beta)
reliability = "GOOD" if breach_count <= expected_breaches * 1.2 else "POOR"

# --------------------------------------------------
# GRAPH 2 — EFFICIENT FRONTIER
# --------------------------------------------------

if not single_stock:
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(volatility_arr, returns_arr, c=sharpe_arr, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label="Sharpe Ratio")

    plt.scatter(volatility_arr[min_vol_idx], returns_arr[min_vol_idx], marker="*", s=300, 
               color='green', label="Min Risk", edgecolors='black', linewidth=2)
    plt.scatter(volatility_arr[max_return_idx], returns_arr[max_return_idx], marker="*", s=300, 
               color='red', label="Max Return", edgecolors='black', linewidth=2)
    plt.scatter(volatility_arr[max_sharpe_idx], returns_arr[max_sharpe_idx], marker="*", s=300, 
               color='gold', label="Max Sharpe", edgecolors='black', linewidth=2)

    plt.title("Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# GRAPH 3 — INDIVIDUAL STOCK RISK vs RETURN
# --------------------------------------------------

plt.figure(figsize=(10,8))

# Plot each stock
for ticker, metrics_stock in stock_metrics.items():
    if metrics_stock['volatility'] > 0 and metrics_stock['return'] != 0:
        plt.scatter(metrics_stock['volatility'], metrics_stock['return'], 
                   s=200, label=ticker, alpha=0.7, edgecolors='black', linewidth=2)
        plt.annotate(ticker, (metrics_stock['volatility'], metrics_stock['return']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# Plot portfolio
portfolio_vol = std * np.sqrt(252)
portfolio_ret = mean * 252
plt.scatter(portfolio_vol, portfolio_ret, s=400, color='red', marker='*', 
           label='Your Portfolio', edgecolors='black', linewidth=2, zorder=5)

plt.xlabel("Annualized Volatility (Risk)", fontsize=12)
plt.ylabel("Annualized Return", fontsize=12)
plt.title("Risk vs Return: Individual Stocks vs Portfolio", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# GRAPH 4 — STRESS TEST VISUALIZATION
# --------------------------------------------------

scenarios = list(stress_results.keys())
losses = [stress_results[s]['loss'] for s in scenarios]

plt.figure(figsize=(12,6))
bars = plt.bar(scenarios, losses, color=['green', 'yellow', 'orange', 'red', 'darkred', 'purple', 'black'])
plt.axhline(y=investment * 0.1, color='blue', linestyle='--', label='10% Loss Threshold')
plt.title(f"Stress Test: Portfolio Losses Under Different Scenarios (Investment: ₹{investment:,.0f})", fontsize=14)
plt.xlabel("Scenario", fontsize=12)
plt.ylabel("Expected Loss (₹)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar, loss in zip(bars, losses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'₹{loss:,.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# GRAPH 5 — RETURN DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(portfolio_returns, bins=40, edgecolor="black", alpha=0.7, color='skyblue')
plt.axvline(var_threshold, linestyle="dashed", linewidth=2, color='red', label="VaR Threshold")
plt.axvline(0, linestyle="solid", linewidth=1, color='black', alpha=0.5)
plt.title("Return Distribution with VaR")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

from scipy import stats
plt.subplot(1,2,2)
stats.probplot(portfolio_returns, dist="norm", plot=plt)
plt.title("Q-Q Plot (Normality Check)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# GRAPH 6 — DRAWDOWN PLOT
# --------------------------------------------------

cumulative_returns = (1 + portfolio_returns).cumprod()
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max * 100

plt.figure(figsize=(10,5))
plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
plt.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
plt.title("Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(alpha=0.3)
plt.show()

# --------------------------------------------------
# GRAPH INTERPRETATION
# --------------------------------------------------

def interpret_graph(returns_series):
    skewness = returns_series.skew()
    kurtosis = returns_series.kurtosis()

    if skewness < -0.5:
        skew_msg = "⚠️ Left-skewed (higher risk of losses)"
    elif skewness > 0.5:
        skew_msg = "✅ Right-skewed (higher potential gains)"
    else:
        skew_msg = "📊 Balanced distribution"

    if kurtosis > 3:
        kurt_msg = "⚠️ Fat tails present (extreme event risk)"
    elif kurtosis < 1:
        kurt_msg = "✅ Thin tails (lower extreme risk)"
    else:
        kurt_msg = "📊 Normal tail behavior"

    return f"{skew_msg} | {kurt_msg}"

graph_insight = interpret_graph(portfolio_returns)

# --------------------------------------------------
# OLLAMA CHATBOT WITH FALLBACK
# --------------------------------------------------

def local_ai(prompt):
    """Try to use Ollama, return None if not available"""
    if not OLLAMA_AVAILABLE:
        return None
    
    try:
        # Try tinyllama first (lightweight)
        response = ollama.chat(
            model="tinyllama",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 150, "temperature": 0.7}
        )
        return response["message"]["content"]
    except:
        try:
            # Try llama2 if tinyllama fails
            response = ollama.chat(
                model="llama2",
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 150, "temperature": 0.7}
            )
            return response["message"]["content"]
        except:
            return None

def advanced_chatbot(q):
    """Advanced chatbot with AI capabilities"""
    q_lower = q.lower()

    # Exit command
    if "exit" in q_lower:
        return "Goodbye! Remember: diversification is the only free lunch in finance 📈"

    # Quick commands (no AI needed)
    if "risk" in q_lower and "level" in q_lower:
        beta_text = f" | Beta: {beta:.2f}" if beta else ""
        return f"📊 Risk Level: {risk_level} | Max Drawdown: {metrics['max_drawdown']:.2%}{beta_text}"

    if "stress" in q_lower or "crash" in q_lower:
        return (f"💥 STRESS TEST RESULTS:\n"
                f"• Mild Crash (-10%): ₹{stress_results['Mild Crash (-10%)']['loss']:,.0f}\n"
                f"• Moderate Crash (-15%): ₹{stress_results['Moderate Crash (-15%)']['loss']:,.0f}\n"
                f"• Severe Crash (-20%): ₹{stress_results['Severe Crash (-20%)']['loss']:,.0f}\n"
                f"• Worst Day: ₹{stress_results['Worst Historical Day']['loss']:,.0f}")

    if "loss" in q_lower or "cvar" in q_lower:
        return f"💀 Worst-case expected loss (CVaR): ₹{metrics['cvar']:,.2f}"

    if "var" in q_lower:
        return f"📉 VaR: ₹{metrics['historical_var']:,.2f} (95% CI: ₹{var_ci_lower:,.2f} - ₹{var_ci_upper:,.2f})"

    if "summary" in q_lower:
        beta_text = f" | Beta: {beta:.2f}" if beta else ""
        return (f"{risk_level} risk | VaR: ₹{metrics['historical_var']:,.2f} | "
                f"CVaR: ₹{metrics['cvar']:,.2f} | Drawdown: {metrics['max_drawdown']:.2%}{beta_text}")

    if "weights" in q_lower or "optimize" in q_lower:
        if not single_stock:
            return f"🎯 Optimal Weights (Max Sharpe): {dict(zip(tickers, max_sharpe_weights.round(3)))}"
        else:
            return "Single stock portfolio - weights are 100%"

    if "compare" in q_lower:
        comparison = "\n".join([f"{t}: Return {stock_metrics[t]['return']:.1%} | Risk {stock_metrics[t]['volatility']:.1%} | Sharpe {stock_metrics[t]['sharpe']:.2f}" 
                               for t in tickers if stock_metrics[t]['return'] != 0])
        return f"📊 STOCK COMPARISON:\n{comparison}"

    if "beta" in q_lower:
        if beta:
            return f"📈 Portfolio Beta: {beta:.2f} ({'More volatile than market' if beta > 1 else 'Less volatile than market' if beta < 1 else 'Same as market'})"
        else:
            return "Beta not calculated (no benchmark selected)"

    if "alpha" in q_lower:
        if alpha:
            return f"📊 Portfolio Alpha: {alpha:.2%} ({'Outperforming' if alpha > 0 else 'Underperforming'} benchmark)"
        else:
            return "Alpha not calculated (no benchmark selected)"

    if "invest" in q_lower:
        if risk_level == "LOW":
            return "✅ You can invest more. Risk is low with controlled drawdowns."
        elif risk_level == "MODERATE":
            return "⚠️ Invest cautiously and diversify. Consider reducing position size."
        else:
            return "❌ High risk with significant drawdowns. Avoid increasing investment now."
    
    if "sharpe" in q_lower:
        return f"⚡ Sharpe Ratio: {(mean * 252 - risk_free_rate) / (std * np.sqrt(252)):.3f}"
    
    if "sortino" in q_lower:
        return f"🎯 Sortino Ratio: {metrics['sortino_ratio']:.3f}"

    if "drawdown" in q_lower:
        return f"📉 Maximum Drawdown: {metrics['max_drawdown']:.2%}"
    
    if "volatility" in q_lower:
        return f"📊 Annualized Volatility: {std*np.sqrt(252):.2%}"

    # For complex questions, try AI if available
    if any(word in q_lower for word in ["explain", "analysis", "recommend", "suggest", "advice", "evaluate"]):
        ai_prompt = f"""
As a financial risk analyst, analyze this portfolio:

Portfolio Metrics:
- Risk Level: {risk_level}
- Investment: ₹{investment:,.0f}
- VaR (99%): ₹{metrics['historical_var']:,.0f}
- CVaR: ₹{metrics['cvar']:,.0f}
- Max Drawdown: {metrics['max_drawdown']:.1%}
- Sharpe Ratio: {(mean * 252 - risk_free_rate) / (std * np.sqrt(252)):.2f}
- Volatility: {std*np.sqrt(252):.1%}
- Return: {mean*252:.1%}

Stocks in portfolio:
{chr(10).join([f"- {t}: {stock_metrics[t]['return']:.1%} return, {stock_metrics[t]['volatility']:.1%} risk" for t in tickers])}

Optimal Weights:
{chr(10).join([f"- {t}: {w:.1%}" for t, w in zip(tickers, max_sharpe_weights)]) if not single_stock else "- Single stock"}

User Question: {q}

Provide a concise, practical answer (2-3 sentences) for an investor.
"""
        
        ai_response = local_ai(ai_prompt)
        if ai_response:
            return f"🤖 AI Analysis: {ai_response}"
    
    # Fallback response for unknown questions
    return (f"📊 Portfolio Summary:\n"
            f"• Risk Level: {risk_level}\n"
            f"• VaR (1-day): ₹{metrics['historical_var']:,.2f}\n"
            f"• CVaR: ₹{metrics['cvar']:,.2f}\n"
            f"• Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"• Sharpe Ratio: {(mean * 252 - risk_free_rate) / (std * np.sqrt(252)):.3f}\n\n"
            f"💡 Try asking: risk, var, cvar, stress, weights, compare, explain, recommend")

# --------------------------------------------------
# EXPORT FULL REPORT
# --------------------------------------------------

def export_full_report():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main metrics report
    report_data = {
        'Metric': [
            'Risk Level', 'VaR (1-day)', 'CVaR (1-day)', 'VaR Multi-day', 
            'CVaR Multi-day', 'Volatility (Annual)', 'Return (Annual)',
            'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio', 
            'Beta' if beta else 'Beta (N/A)', 'Alpha' if alpha else 'Alpha (N/A)',
            'Model Reliability', 'VaR 95% CI Lower', 'VaR 95% CI Upper', 
            'Breach Count', 'Expected Breaches'
        ],
        'Value': [
            risk_level, f'₹{metrics["historical_var"]:,.2f}', f'₹{metrics["cvar"]:,.2f}',
            f'₹{metrics["historical_var"] * np.sqrt(time_horizon):,.2f}', 
            f'₹{metrics["cvar"] * np.sqrt(time_horizon):,.2f}',
            f'{std*np.sqrt(252):.2%}', f'{mean*252:.2%}',
            f'{metrics["max_drawdown"]:.2%}',
            f'{(mean * 252 - risk_free_rate) / (std * np.sqrt(252)):.3f}',
            f'{metrics["sortino_ratio"]:.3f}',
            f'{beta:.3f}' if beta else 'N/A',
            f'{alpha:.2%}' if alpha else 'N/A',
            reliability, f'₹{var_ci_lower:,.2f}', f'₹{var_ci_upper:,.2f}',
            breach_count, f'{expected_breaches:.1f}'
        ]
    }
    
    df_main = pd.DataFrame(report_data)
    df_main.to_csv(f'portfolio_report_{timestamp}.csv', index=False)
    print(f"✅ Main report exported: portfolio_report_{timestamp}.csv")
    
    # Stress test report
    stress_df = pd.DataFrame([
        {'Scenario': s, 'Loss (₹)': f'₹{stress_results[s]["loss"]:,.0f}', 
         'Loss %': f'{stress_results[s]["loss_pct"]:.1%}', 
         'Portfolio Value': f'₹{stress_results[s]["new_value"]:,.0f}'}
        for s in stress_results
    ])
    stress_df.to_csv(f'stress_test_{timestamp}.csv', index=False)
    print(f"✅ Stress test exported: stress_test_{timestamp}.csv")
    
    # Individual stock comparison
    stock_comparison = []
    for ticker in tickers:
        m = stock_metrics[ticker]
        stock_comparison.append({
            'Stock': ticker,
            'Annual Return': f'{m["return"]:.2%}' if m["return"] != 0 else 'N/A',
            'Volatility': f'{m["volatility"]:.2%}' if m["volatility"] != 0 else 'N/A',
            'Sharpe Ratio': f'{m["sharpe"]:.3f}' if m["sharpe"] != 0 else 'N/A',
            'Sortino Ratio': f'{m["sortino"]:.3f}' if m["sortino"] != 0 else 'N/A',
            'Max Drawdown': f'{m["max_drawdown"]:.2%}' if m["max_drawdown"] != 0 else 'N/A',
            'Beta': f'{m["beta"]:.3f}' if m["beta"] else 'N/A'
        })
    
    df_stocks = pd.DataFrame(stock_comparison)
    df_stocks.to_csv(f'stock_comparison_{timestamp}.csv', index=False)
    print(f"✅ Stock comparison exported: stock_comparison_{timestamp}.csv")
    
    # Optimal weights
    if not single_stock:
        weights_df = pd.DataFrame({
            'Ticker': tickers,
            'Min Risk': min_vol_weights,
            'Max Return': max_return_weights,
            'Max Sharpe': max_sharpe_weights
        })
        weights_df.to_csv(f'optimal_weights_{timestamp}.csv', index=False)
        print(f"✅ Optimal weights exported: optimal_weights_{timestamp}.csv")
    
    print(f"\n📁 All reports saved with timestamp: {timestamp}")

# --------------------------------------------------
# PRINT FULL REPORT
# --------------------------------------------------

print("\n" + "="*70)
print("PORTFOLIO RISK ANALYSIS REPORT")
print("="*70)
print(f"\n📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💰 Investment Amount: ₹{investment:,.2f}")
print(f"🎯 Confidence Level: {confidence_level:.0%}")
print(f"⏰ Time Horizon: {time_horizon} days")
print(f"📊 Risk-Free Rate: {risk_free_rate:.1%}")
print(f"📈 Benchmark: {'NIFTY 50' if use_benchmark else 'None selected'}")

print("\n" + "-"*70)
print("RISK METRICS")
print("-"*70)
print(f"⚠️  Risk Level: {risk_level}")
print(f"📉 VaR (1-day): ₹{metrics['historical_var']:,.2f}")
print(f"📊 VaR (1-day) 95% CI: [₹{var_ci_lower:,.2f}, ₹{var_ci_upper:,.2f}]")
print(f"💀 CVaR (1-day): ₹{metrics['cvar']:,.2f}")
print(f"📈 VaR ({time_horizon}-day): ₹{metrics['historical_var'] * np.sqrt(time_horizon):,.2f}")
print(f"💀 CVaR ({time_horizon}-day): ₹{metrics['cvar'] * np.sqrt(time_horizon):,.2f}")
print(f"📉 Parametric VaR: ₹{metrics['parametric_var']:,.2f}")
print(f"🎲 Monte Carlo VaR (t-dist): ₹{metrics['monte_carlo_var']:,.2f}")

print("\n" + "-"*70)
print("PERFORMANCE METRICS")
print("-"*70)
print(f"📊 Annualized Volatility: {std*np.sqrt(252):.2%}")
print(f"📈 Annualized Return: {mean*252:.2%}")
print(f"⚡ Sharpe Ratio: {(mean * 252 - risk_free_rate) / (std * np.sqrt(252)):.3f}")
print(f"🎯 Sortino Ratio: {metrics['sortino_ratio']:.3f}")
if beta:
    print(f"📊 Beta: {beta:.3f} ({'More volatile' if beta > 1 else 'Less volatile' if beta < 1 else 'Same as'} market)")
if alpha:
    print(f"⭐ Alpha: {alpha:.2%} ({'Outperforming' if alpha > 0 else 'Underperforming'} benchmark)")
print(f"📉 Maximum Drawdown: {metrics['max_drawdown']:.2%}")

print("\n" + "-"*70)
print("🔥 STRESS TEST RESULTS")
print("-"*70)
for scenario, data in stress_results.items():
    print(f"{scenario}: ₹{data['loss']:,.0f} loss ({data['loss_pct']:.1%}) → New value: ₹{data['new_value']:,.0f}")

print("\n" + "-"*70)
print("📊 INDIVIDUAL STOCK COMPARISON")
print("-"*70)
print(f"{'Stock':<12} {'Return':<10} {'Risk':<10} {'Sharpe':<8} {'Beta':<8} {'Max DD':<10}")
print("-"*70)
for ticker in tickers:
    m = stock_metrics[ticker]
    return_str = f'{m["return"]:.1%}' if m["return"] != 0 else 'N/A'
    risk_str = f'{m["volatility"]:.1%}' if m["volatility"] != 0 else 'N/A'
    sharpe_str = f'{m["sharpe"]:.2f}' if m["sharpe"] != 0 else 'N/A'
    beta_str = f'{m["beta"]:.2f}' if m["beta"] else 'N/A'
    dd_str = f'{m["max_drawdown"]:.1%}' if m["max_drawdown"] != 0 else 'N/A'
    print(f"{ticker:<12} {return_str:<10} {risk_str:<10} {sharpe_str:<8} {beta_str:<8} {dd_str:<10}")

print("\n" + "-"*70)
print("MODEL VALIDATION")
print("-"*70)
print(f"✅ Model Reliability: {reliability}")
print(f"📊 Breach Count: {breach_count} / {expected_breaches:.1f} expected")
print(f"📐 Return Skewness: {portfolio_returns.skew():.3f}")
print(f"📏 Return Kurtosis: {portfolio_returns.kurtosis():.3f}")

print("\n" + "-"*70)
print("GRAPH INSIGHTS")
print("-"*70)
print(f"📊 {graph_insight}")

if not single_stock:
    print("\n" + "-"*70)
    print("OPTIMAL PORTFOLIO WEIGHTS (MAX SHARPE)")
    print("-"*70)
    for t, w in zip(tickers, max_sharpe_weights):
        print(f"{t}: {w:.2%}")

# Export prompt
print("\n" + "-"*70)
export_choice = input("💾 Export all reports to CSV? (y/n): ").lower()
if export_choice == 'y':
    export_full_report()

# --------------------------------------------------
# CHAT LOOP
# --------------------------------------------------

print("\n" + "="*70)
print("🤖 AI RISK ASSISTANT READY")
print("="*70)
print("💬 Ask me about:")
print("  • risk, var, cvar, stress, crash")
print("  • beta, alpha, sharpe, sortino, drawdown")
print("  • weights, compare, summary, invest")
print("  • explain, analysis, recommend (uses AI if available)")
print("  • Type 'exit' to quit")
print("="*70)

# Check if Ollama is available
if OLLAMA_AVAILABLE:
    try:
        test_response = local_ai("Hello")
        if test_response:
            print("✅ AI Assistant: ACTIVE (Ollama detected)")
        else:
            print("⚠️ AI Assistant: Ollama installed but model not loaded")
            print("   Run: ollama pull tinyllama")
    except:
        print("⚠️ AI Assistant: Ollama not responding")
        print("   Make sure ollama serve is running")
else:
    print("⚠️ AI Assistant: FALLBACK MODE (Install Ollama for AI responses)")
    print("   To enable AI: pip install ollama && ollama pull tinyllama")
print("="*70)

while True:
    q = input("\nYou: ")
    
    if q.lower() == "exit":
        print("AI: Goodbye! Remember: diversification is the only free lunch in finance 📈")
        break
    
    print("AI:", advanced_chatbot(q))