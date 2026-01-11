"""Performance metrics calculations """
import numpy as np
import pandas as pd
from typing import Dict


def calculate_total_return(initial: float, final: float) -> float:
    """Total return percentage (pure)."""
    return ((final / initial) - 1) * 100


def calculate_sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio (pure)."""
    if returns.std() == 0:
        return 0.0
    excess = returns - (risk_free / 252)
    return np.sqrt(252) * excess.mean() / returns.std()

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as negative percentage (pure)."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    return drawdown.min()

def calculate_sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio (pure)."""
    excess = returns - (risk_free / 252)
    downside = returns[returns < 0]
    
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    
    return np.sqrt(252) * excess.mean() / downside.std()


def calculate_calmar_ratio(total_return: float, max_dd: float) -> float:
    """Calmar ratio (pure)."""
    if max_dd == 0:
        return float('inf')
    return total_return / abs(max_dd)


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """Win rate percentage (pure)."""
    if len(trades) == 0:
        return 0.0
    closed = trades[trades['action'] == 'CLOSE']
    winning = closed[closed['net_pnl'] > 0]
    return (len(winning) / len(closed)) * 100 if len(closed) > 0 else 0.0


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """Profit factor (pure)."""
    closed = trades[trades['action'] == 'CLOSE']
    if len(closed) == 0:
        return 0.0
    
    gross_profit = closed[closed['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(closed[closed['net_pnl'] < 0]['net_pnl'].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expectancy(trades: pd.DataFrame) -> float:
    """Average trade expectancy (pure)."""
    closed = trades[trades['action'] == 'CLOSE']
    if len(closed) == 0:
        return 0.0
    
    win_rate = calculate_win_rate(trades) / 100
    
    winning = closed[closed['net_pnl'] > 0]
    losing = closed[closed['net_pnl'] < 0]
    
    avg_win = winning['net_pnl'].mean() if len(winning) > 0 else 0
    avg_loss = abs(losing['net_pnl'].mean()) if len(losing) > 0 else 0
    
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


def calculate_log_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns"""
    if method == 'log':
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def calculate_cumulative_returns(returns_series: pd.Series) -> pd.Series:
    """Cumulative returns"""
    return (1 + returns_series).cumprod() - 1


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Drawdown series ."""
    running_max = equity_curve.expanding().max()
    return (equity_curve - running_max) / running_max

def calculate_all_metrics(portfolio: Dict, initial_capital: float) -> Dict:
    """
    Calculate all performance metrics .
    
    Returns dict with all metrics.
    """
    daily_values = pd.DataFrame(portfolio['daily_values'])
    trades_df = pd.DataFrame([
        {
            'action': t.action,
            'net_pnl': t.pnl,
            'ticker': t.ticker,
            **t.metadata
        } for t in portfolio['trades']
    ])
    
    if daily_values.empty:
        return {}
    
    # Calculate returns
    equity = daily_values['total_value']
    returns = equity.pct_change().dropna()
    
    final_value = equity.iloc[-1]
    total_return = calculate_total_return(initial_capital, final_value)
    
    # Risk metrics
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(equity)
    calmar = calculate_calmar_ratio(total_return, max_dd)
    
    # Trade metrics
    closed_trades = trades_df[trades_df['action'] == 'CLOSE']
    win_rate = calculate_win_rate(trades_df)
    profit_factor = calculate_profit_factor(trades_df)
    expectancy = calculate_expectancy(trades_df)
    
    # Costs
    total_commission = trades_df['commission'].sum() if 'commission' in trades_df else 0
    total_slippage = trades_df['slippage_cost'].sum() if 'slippage_cost' in trades_df else 0
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_pnl': final_value - initial_capital,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': max_dd,
        'calmar_ratio': calmar,
        'total_trades': len(closed_trades),
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'total_commission': total_commission,
        'total_slippage': total_slippage,
        'volatility_annual_pct': returns.std() * np.sqrt(252) * 100,
        'best_trade': closed_trades['net_pnl'].max() if len(closed_trades) > 0 else 0,
        'worst_trade': closed_trades['net_pnl'].min() if len(closed_trades) > 0 else 0,
        'avg_trade': closed_trades['net_pnl'].mean() if len(closed_trades) > 0 else 0
    }
