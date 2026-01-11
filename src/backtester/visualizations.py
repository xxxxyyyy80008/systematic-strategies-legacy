"""Visualization utilities for backtest analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from matplotlib.gridspec import GridSpec


def plot_equity_curve(daily_values, initial_capital: float, 
                      title: str = "Portfolio Equity Curve"):
    """
    Plot portfolio equity curve with profit/loss shading.
    
    Args:
        daily_values: DataFrame with 'date' and 'total_value' columns
        initial_capital: Starting capital
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df = pd.DataFrame(daily_values)
    df.set_index('date', inplace=True)
    
    # Plot equity curve
    ax.plot(df.index, df['total_value'], linewidth=2, color='darkblue', 
            label='Portfolio Value', zorder=3)
    
    # Initial capital line
    ax.axhline(y=initial_capital, color='red', linestyle='--', 
               alpha=0.7, linewidth=1.5, label='Initial Capital')
    
    # Profit/loss shading
    ax.fill_between(df.index, initial_capital, df['total_value'],
                     where=(df['total_value'] >= initial_capital),
                     alpha=0.3, color='green', label='Profit Zone')
    ax.fill_between(df.index, initial_capital, df['total_value'],
                     where=(df['total_value'] < initial_capital),
                     alpha=0.3, color='red', label='Loss Zone')
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()


def plot_drawdown(daily_values, title: str = "Portfolio Drawdown"):
    """
    Plot drawdown chart showing underwater periods.
    
    Args:
        daily_values: DataFrame with 'date' and 'total_value' columns
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df = pd.DataFrame(daily_values)
    df.set_index('date', inplace=True)
    
    # Calculate drawdown
    running_max = df['total_value'].cummax()
    drawdown = (df['total_value'] - running_max) / running_max * 100
    
    # Plot
    ax.fill_between(df.index, 0, drawdown, alpha=0.7, color='red', 
                     label='Drawdown')
    ax.plot(df.index, drawdown, linewidth=1.5, color='darkred', zorder=3)
    
    # Mark maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.scatter(max_dd_idx, max_dd_val, color='darkred', s=100, 
               zorder=5, label=f'Max DD: {max_dd_val:.2f}%')
    ax.annotate(f'{max_dd_val:.2f}%', 
                xy=(max_dd_idx, max_dd_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(daily_values, 
                              title: str = "Daily Returns Distribution"):
    """
    Plot distribution of daily returns with statistics.
    
    Args:
        daily_values: DataFrame with 'date' and 'total_value' columns
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = pd.DataFrame(daily_values)
    df.set_index('date', inplace=True)
    
    # Calculate returns
    returns = df['total_value'].pct_change().dropna() * 100
    
    # Plot histogram
    n, bins, patches = ax.hist(returns, bins=50, alpha=0.7, color='steelblue', 
                                edgecolor='black', density=True)
    
    # Color bars by positive/negative
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('red')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('green')
            patch.set_alpha(0.6)
    
    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sigma), 'k--', linewidth=2, 
            label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
    
    # Statistics lines
    ax.axvline(x=mu, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mu:.3f}%')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Text box with statistics
    stats_text = (
        f'Statistics:\n'
        f'Mean: {mu:.3f}%\n'
        f'Median: {returns.median():.3f}%\n'
        f'Std Dev: {sigma:.3f}%\n'
        f'Skewness: {returns.skew():.3f}\n'
        f'Kurtosis: {returns.kurtosis():.3f}'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Daily Return (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()


def plot_trade_analysis(trades, 
                       title: str = "Trade Analysis Dashboard"):
    """
    Plot comprehensive trade analysis with multiple subplots.
    
    Args:
        trades: DataFrame with trade data
        title: Main title
    """
    df = pd.DataFrame(trades)
    closed_trades = df[df['action'] == 'CLOSE'].copy()
    closed_trades['net_pnl'] = closed_trades['pnl']
    if len(closed_trades) == 0:
        print("⚠️  No closed trades to analyze")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. P&L by Ticker
    ax1 = fig.add_subplot(gs[0, 0])
    pnl_by_ticker = closed_trades.groupby('ticker')['net_pnl'].sum().sort_values()
    colors = ['green' if x >= 0 else 'red' for x in pnl_by_ticker.values]
    pnl_by_ticker.plot(kind='barh', ax=ax1, color=colors, edgecolor='black')
    ax1.set_title('Total P&L by Ticker', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Net P&L ($)', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Win Rate by Ticker
    ax2 = fig.add_subplot(gs[0, 1])
    win_rate = closed_trades.groupby('ticker').apply(
        lambda x: (x['net_pnl'] > 0).sum() / len(x) * 100,
        include_groups=False
    ).sort_values(ascending=False)
    win_rate.plot(kind='bar', ax=ax2, color='steelblue', edgecolor='black')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_title('Win Rate by Ticker', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)', fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Cumulative P&L
    ax3 = fig.add_subplot(gs[1, :])
    closed_trades_sorted = closed_trades.sort_values('execution_date')
    closed_trades_sorted['cumulative_pnl'] = closed_trades_sorted['net_pnl'].cumsum()
    ax3.plot(closed_trades_sorted['execution_date'], 
             closed_trades_sorted['cumulative_pnl'],
             linewidth=2.5, color='darkgreen', marker='o', markersize=3)
    ax3.fill_between(closed_trades_sorted['execution_date'], 
                      0, closed_trades_sorted['cumulative_pnl'],
                      alpha=0.3, color='green')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_title('Cumulative P&L Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_ylabel('Cumulative P&L ($)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 4. Trade P&L Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(closed_trades['net_pnl'], bins=30, alpha=0.7, 
             color='steelblue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                label='Break-even')
    ax4.axvline(x=closed_trades['net_pnl'].mean(), color='orange', 
                linestyle='--', linewidth=2, 
                label=f"Mean: ${closed_trades['net_pnl'].mean():.2f}")
    ax4.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('P&L per Trade ($)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Win/Loss Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    winners = closed_trades[closed_trades['net_pnl'] > 0]
    losers = closed_trades[closed_trades['net_pnl'] < 0]
    
    categories = ['Winners', 'Losers']
    counts = [len(winners), len(losers)]
    avg_pnl = [winners['net_pnl'].mean() if len(winners) > 0 else 0,
               abs(losers['net_pnl'].mean()) if len(losers) > 0 else 0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, counts, width, label='Count', 
                    color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x + width/2, avg_pnl, width, label='Avg P&L', 
                         color=['darkgreen', 'darkred'], alpha=0.7, 
                         edgecolor='black')
    
    ax5.set_xlabel('Trade Type', fontweight='bold')
    ax5.set_ylabel('Count', fontweight='bold', color='black')
    ax5_twin.set_ylabel('Average P&L ($)', fontweight='bold', color='black')
    ax5.set_title('Win/Loss Statistics', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories)
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.show()


def plot_monthly_returns(daily_values,
                        title: str = "Monthly Returns Heatmap"):
    """
    Plot monthly returns heatmap.
    
    Args:
        daily_values: DataFrame with 'date' and 'total_value' columns
        title: Chart title
    """
    df = pd.DataFrame(daily_values)
    df.set_index('date', inplace=True)
    
    # Calculate daily returns
    df['returns'] = df['total_value'].pct_change()
    
    # Resample to monthly
    monthly = df['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create pivot table for heatmap
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Return (%)'}, linewidths=1, 
                linecolor='gray', ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Year', fontsize=12, fontweight='bold')
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names, rotation=0)
    
    plt.tight_layout()
    plt.show()


def plot_rolling_metrics(daily_values, window: int = 60,
                         title: str = "Rolling Performance Metrics"):
    """
    Plot rolling Sharpe ratio and volatility.
    
    Args:
        daily_values: DataFrame with 'date' and 'total_value' columns
        window: Rolling window in days
        title: Chart title
    """
    df = pd.DataFrame(daily_values)
    df.set_index('date', inplace=True)
    
    # Calculate returns
    returns = df['total_value'].pct_change().dropna()
    
    # Rolling metrics
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
    )
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Rolling Sharpe
    ax1.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, 
             color='darkblue', label=f'{window}-Day Rolling Sharpe')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, 
                label='Sharpe = 1.0')
    ax1.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                      where=(rolling_sharpe >= 0), alpha=0.3, color='green')
    ax1.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                      where=(rolling_sharpe < 0), alpha=0.3, color='red')
    ax1.set_title(f'{window}-Day Rolling Sharpe Ratio', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Rolling Volatility
    ax2.plot(rolling_vol.index, rolling_vol, linewidth=2, 
             color='darkred', label=f'{window}-Day Rolling Volatility')
    ax2.fill_between(rolling_vol.index, 0, rolling_vol, alpha=0.3, color='red')
    ax2.set_title(f'{window}-Day Rolling Annualized Volatility', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Volatility (%)', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_underwater_chart(daily_values,
                          title: str = "Underwater (Drawdown) Chart"):
    """
    Plot underwater chart showing time underwater.
    
    Args:
        daily_values: DataFrame with 'date' and 'total_value' columns
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df = pd.DataFrame(daily_values)
    df.set_index('date', inplace=True)
    
    # Calculate drawdown
    running_max = df['total_value'].cummax()
    drawdown = (df['total_value'] - running_max) / running_max * 100
    
    # Find underwater periods
    underwater = drawdown < -0.1  # More than 0.1% underwater
    
    # Plot
    ax.fill_between(df.index, -100, 0, where=underwater, 
                     alpha=0.5, color='red', label='Underwater')
    ax.plot(df.index, drawdown, linewidth=1.5, color='darkred')
    
    # Calculate underwater statistics
    underwater_periods = (underwater != underwater.shift()).cumsum()
    underwater_lengths = underwater.groupby(underwater_periods).sum()
    max_underwater = underwater_lengths.max() if len(underwater_lengths) > 0 else 0
    
    # Add statistics text
    stats_text = (
        f'Max Underwater Period: {max_underwater} days\n'
        f'Current DD: {drawdown.iloc[-1]:.2f}%\n'
        f'Max DD: {drawdown.min():.2f}%'
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([drawdown.min() * 1.1, 5])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_per_ticker_performance(portfolio: Dict,
                                title: str = "Per-Ticker Performance"):
    """
    Plot performance breakdown by ticker.
    
    Args:
        portfolio: Portfolio state dictionary
        title: Chart title
    """
    # Extract ticker data
    ticker_data = []
    for ticker, account in portfolio['ticker_accounts'].items():
        ticker_data.append({
            'ticker': ticker,
            'pnl': account['total_pnl'],
            'trades': account['trade_count'],
            'return_pct': (account['total_pnl'] / account['starting_capital']) * 100
        })
    
    df = pd.DataFrame(ticker_data).sort_values('pnl', ascending=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # P&L chart
    colors = ['green' if x >= 0 else 'red' for x in df['pnl']]
    df.plot(x='ticker', y='pnl', kind='barh', ax=ax1, color=colors, 
            legend=False, edgecolor='black')
    ax1.set_title('Total P&L by Ticker', fontsize=12, fontweight='bold')
    ax1.set_xlabel('P&L ($)', fontweight='bold')
    ax1.set_ylabel('Ticker', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Return % chart
    colors = ['green' if x >= 0 else 'red' for x in df['return_pct']]
    df.plot(x='ticker', y='return_pct', kind='barh', ax=ax2, color=colors,
            legend=False, edgecolor='black')
    ax2.set_title('Return % by Ticker', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Return (%)', fontweight='bold')
    ax2.set_ylabel('')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_performance_dashboard(results, initial_capital: float):
    """
    Create comprehensive performance dashboard.
    
    Combines multiple visualizations into one display.
    
    Args:
        results: BacktestResults object
        initial_capital: Starting capital
    """
     
    # 1. Equity Curve
    plot_equity_curve(results["daily_values"], initial_capital)
    
    # 2. Drawdown
    plot_drawdown(results["daily_values"])
    
    # 3. Returns Distribution
    plot_returns_distribution(results["daily_values"])
    
    # 4. Trade Analysis
    plot_trade_analysis(results["trades"])
    
    # 5. Monthly Returns Heatmap
    if len(results["daily_values"]) > 30:
        plot_monthly_returns(results["daily_values"])
    
    # 6. Rolling Metrics
    if len(results["daily_values"]) > 60:
        plot_rolling_metrics(results["daily_values"], window=60)
    
    # 7. Underwater Chart
    plot_underwater_chart(results["daily_values"])
    
    # 8. Per-Ticker Performance
    plot_per_ticker_performance(results)


def save_all_charts(results, initial_capital: float, output_dir: str = 'results/'):
    """
    Save all charts to files instead of displaying.
    
    Args:
        results: BacktestResults object
        initial_capital: Starting capital
        output_dir: Directory to save charts
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporarily disable interactive mode
    plt.ioff()
    
    print(f"\n Saving charts to {output_dir}...")
    
    # Save each chart
    charts = [
        ('equity_curve.png', lambda: plot_equity_curve(results.daily_values, initial_capital)),
        ('drawdown.png', lambda: plot_drawdown(results.daily_values)),
        ('returns_dist.png', lambda: plot_returns_distribution(results.daily_values)),
        ('trade_analysis.png', lambda: plot_trade_analysis(results.trades)),
        ('monthly_returns.png', lambda: plot_monthly_returns(results.daily_values)),
        ('rolling_metrics.png', lambda: plot_rolling_metrics(results.daily_values)),
        ('underwater.png', lambda: plot_underwater_chart(results.daily_values)),
        ('per_ticker.png', lambda: plot_per_ticker_performance(results.portfolio_state))
    ]
    
    for filename, plot_func in charts:
        try:
            plot_func()
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   Saved {filename}")
        except Exception as e:
            print(f"   Failed to save {filename}: {e}")
    
    # Re-enable interactive mode
    plt.ion()
    
    print(f"\n Charts saved to {output_dir}\n")
