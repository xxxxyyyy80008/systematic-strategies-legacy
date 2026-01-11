"""Report generation utilities."""
from typing import Dict
import pandas as pd


def print_performance_summary(metrics: Dict):
    """Print formatted performance summary."""
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n*** RETURNS ***")
    print(f"  Initial Capital:    ${metrics['initial_capital']:>12,.2f}")
    print(f"  Final Value:        ${metrics['final_value']:>12,.2f}")
    print(f"  Total P&L:          ${metrics['total_pnl']:>12,.2f}")
    print(f"  Total Return:       {metrics['total_return_pct']:>12.2f}%")
    
    print(f"\n*** RISK METRICS ***")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>12.2f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>12.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:>12.2f}%")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>12.2f}")
    print(f"  Annual Volatility:  {metrics['volatility_annual_pct']:>12.2f}%")
    
    print(f"\n*** TRADE STATISTICS ***")
    print(f"  Total Trades:       {metrics['total_trades']:>12}")
    print(f"  Win Rate:           {metrics['win_rate_pct']:>12.2f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']:>12.2f}")
    print(f"  Expectancy:         ${metrics['expectancy']:>12.2f}")
    print(f"  Best Trade:         ${metrics['best_trade']:>12.2f}")
    print(f"  Worst Trade:        ${metrics['worst_trade']:>12.2f}")
    print(f"  Average Trade:      ${metrics['avg_trade']:>12.2f}")
    
    print(f"\n*** COSTS ***")
    print(f"  Total Commission:   ${metrics['total_commission']:>12.2f}")
    print(f"  Total Slippage:     ${metrics['total_slippage']:>12.2f}")
    print(f"  Total Costs:        ${metrics['total_commission'] + metrics['total_slippage']:>12.2f}")
    
    print(f"\n{'='*80}\n")


def print_per_ticker_summary(portfolio: Dict):
    """Print per-ticker performance summary."""
    print(f"\n*** PER-TICKER SUMMARY ***")
    print("-" * 80)
    print(f"{'Ticker':<10} {'Starting':>12} {'Final':>12} {'P&L':>12} {'Return':>10} {'Trades':>8}")
    print("-" * 80)
    
    for ticker, account in portfolio['ticker_accounts'].items():
        starting = account['starting_capital']
        final = account['available_capital']
        pnl = account['total_pnl']
        ret = (pnl / starting) * 100
        trades = account['trade_count']
        
        print(f"{ticker:<10} ${starting:>10,.2f} ${final:>10,.2f} "
              f"${pnl:>10,.2f} {ret:>9.2f}% {trades:>8}")
    
    print("-" * 80)


def export_to_csv(results, output_dir: str = 'results/'):
    """Export results to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export trades
    trades_file = os.path.join(output_dir, 'trades.csv')
    results.trades.to_csv(trades_file, index=False)
    print(f"  Trades exported: {trades_file}")
    
    # Export daily values
    daily_file = os.path.join(output_dir, 'daily_values.csv')
    results.daily_values.to_csv(daily_file, index=False)
    print(f"  Daily values exported: {daily_file}")
    
    # Export metrics
    metrics_file = os.path.join(output_dir, 'metrics.csv')
    pd.DataFrame([results.metrics]).to_csv(metrics_file, index=False)
    print(f"  Metrics exported: {metrics_file}")
