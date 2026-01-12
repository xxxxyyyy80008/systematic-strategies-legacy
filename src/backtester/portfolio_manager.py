"""Portfolio and capital management (functional approach)."""
from typing import Dict, Tuple


def allocate_per_ticker(total_capital: float, num_tickers: int) -> float:
    """Allocate equal capital per ticker ."""
    return total_capital / num_tickers


def calculate_position_size(available_capital: float, price: float,
                           position_size_pct: float, max_size: float,
                           min_size: float) -> Tuple[float, bool]:
    """
    Calculate position size .
    
    Returns:
        Tuple of (target_size, can_trade)
    """
    target_size = available_capital * position_size_pct
    target_size = max(min(target_size, max_size), min_size)
    
    can_trade = (target_size >= min_size and 
                target_size <= available_capital * 0.95)
    
    return target_size, can_trade


def calculate_shares(target_size: float, price: float, commission_pct: float,
                     slippage_pct: float) -> float:
    """Calculate shares accounting for costs ."""
    safety_buffer = 0.02 
    total_cost_factor = 1 + commission_pct + slippage_pct + safety_buffer
    return target_size / (price * total_cost_factor)
    

def initialize_portfolio(tickers: list, capital_per_ticker: float) -> Dict:
    """
    Initialize portfolio state .
    
    Returns immutable-style state dict.
    """
    return {
        'ticker_accounts': {
            ticker: {
                'starting_capital': capital_per_ticker,
                'available_capital': capital_per_ticker,
                'position': None,
                'total_pnl': 0.0,
                'trade_count': 0
            }
            for ticker in tickers
        },
        'capital_per_ticker': capital_per_ticker,
        'daily_values': [],
        'trades': []
    }


def get_portfolio_value(portfolio: Dict, current_prices: Dict) -> float:
    """Calculate total portfolio value ."""
    total = 0.0
    
    for ticker, account in portfolio['ticker_accounts'].items():
        # Cash
        total += account['available_capital']
        
        # Position value
        if account['position'] is not None and ticker in current_prices:
            position = account['position']
            total += position['shares'] * current_prices[ticker]
    
    return total
