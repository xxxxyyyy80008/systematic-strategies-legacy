"""Trade execution and cost calculations"""
from typing import Dict, Tuple, Optional, List
import pandas as pd
from .types_core import Trade, PositionSide, TradeConfig


def calculate_slippage(price: float, quantity: float, side: str,
                      pct_slippage: float, fixed_slippage: float) -> float:
    """Calculate slippage-adjusted price ."""
    pct_component = price * pct_slippage
    fixed_component = fixed_slippage * (abs(quantity) ** 0.5)
    total_slippage = pct_component + fixed_component
    
    if side.lower() == 'buy':
        return price + total_slippage
    else:
        return price - total_slippage


def calculate_slippage_cost(price: float, quantity: float,
                           pct: float, fixed: float) -> float:
    """Calculate dollar cost of slippage ."""
    pct_component = price * pct
    fixed_component = fixed * (abs(quantity) ** 0.5)
    return (pct_component + fixed_component) * abs(quantity)


def calculate_commission(position_value: float, commission_pct: float) -> float:
    """Calculate commission ."""
    return position_value * commission_pct


def calculate_trade_costs(price: float, shares: float, side: str,
                         config: TradeConfig) -> Tuple[float, float, float]:
    """
    Calculate all trade costs .
    
    Returns:
        Tuple of (execution_price, commission, slippage_cost)
    """
    exec_price = calculate_slippage(
        price, shares, side,
        config.slippage_pct, config.slippage_fixed
    )
    
    position_value = shares * exec_price
    commission = calculate_commission(position_value, config.commission_pct)
    slippage = calculate_slippage_cost(
        price, shares,
        config.slippage_pct, config.slippage_fixed
    )
    
    return exec_price, commission, slippage


def execute_entry(portfolio: Dict, ticker: str, price: float,
                 signal_date: pd.Timestamp, exec_date: pd.Timestamp,
                 config: TradeConfig) -> Optional[Trade]:
    """
    Execute entry trade (modifies portfolio state).
    
    Returns Trade object if successful.
    """
    account = portfolio['ticker_accounts'][ticker]
    
    # Check if already has position
    if account['position'] is not None:
        return None
    
    # Calculate position size
    from .portfolio_manager import calculate_position_size, calculate_shares
    
    target_size, can_trade = calculate_position_size(
        account['available_capital'],
        price,
        config.position_size_pct,
        config.max_trade_size,
        config.min_trade_size
    )
    
    if not can_trade:
        return None
    
    # Calculate costs
    shares = calculate_shares(
        target_size, price,
        config.commission_pct, config.slippage_pct
    )
    
    exec_price, commission, slippage = calculate_trade_costs(
        price, shares, 'buy', config
    )
    
    # Total cost
    total_cost = (shares * exec_price) + commission
    
    if total_cost > account['available_capital']:
        return None
    
    # Update account
    account['available_capital'] -= total_cost
    account['position'] = {
        'shares': shares,
        'entry_price': exec_price,
        'entry_date': exec_date
    }
    
    # Create trade record
    trade = Trade(
        ticker=ticker,
        action='OPEN',
        signal_date=signal_date,
        execution_date=exec_date,
        side=PositionSide.LONG,
        shares=shares,
        price=exec_price,
        commission=commission,
        slippage_cost=slippage,
        pnl=None,
        metadata={'total_cost': total_cost}
    )
    
    portfolio['trades'].append(trade)
    account['trade_count'] += 1
    
    return trade


def execute_exit(portfolio: Dict, ticker: str, price: float,
                signal_date: pd.Timestamp, exec_date: pd.Timestamp,
                config: TradeConfig) -> Optional[Trade]:
    """
    Execute exit trade (modifies portfolio state).
    
    Returns Trade object if successful.
    """
    account = portfolio['ticker_accounts'][ticker]
    
    if account['position'] is None:
        return None
    
    position = account['position']
    shares = position['shares']
    entry_price = position['entry_price']
    
    # Calculate exit costs
    exec_price, commission, slippage = calculate_trade_costs(
        price, shares, 'sell', config
    )
    
    # Calculate P&L
    gross_pnl = shares * (exec_price - entry_price)
    net_pnl = gross_pnl - commission
    
    # Update account
    proceeds = (shares * exec_price) - commission
    account['available_capital'] += proceeds
    account['total_pnl'] += net_pnl
    account['position'] = None
    
    # Create trade record
    trade = Trade(
        ticker=ticker,
        action='CLOSE',
        signal_date=signal_date,
        execution_date=exec_date,
        side=PositionSide.LONG,
        shares=shares,
        price=exec_price,
        commission=commission,
        slippage_cost=slippage,
        pnl=net_pnl,
        metadata={
            'entry_price': entry_price,
            'entry_date': position['entry_date'],
            'gross_pnl': gross_pnl,
            'proceeds': proceeds,
            'return_pct': (net_pnl / (shares * entry_price)) * 100
        }
    )
    
    portfolio['trades'].append(trade)
    
    return trade
