"""Main backtest engine."""
from typing import Dict, List
import pandas as pd
from .types_core import BacktestResults, TradeConfig
from .strategy_base import Strategy
from .portfolio_manager import allocate_per_ticker, initialize_portfolio, get_portfolio_value
from .trade_executor import execute_entry, execute_exit


class BacktestEngine:
    """Core backtesting engine."""
    
    def __init__(self, strategy: Strategy, trade_config: TradeConfig):
        self.strategy = strategy
        self.config = trade_config
    
    def run(self, data_dict: Dict[str, pd.DataFrame]) -> BacktestResults:
        """Run backtest on multiple tickers."""
        
        if not data_dict or len(data_dict) == 0:
            print("Error: No data provided")
            return self._empty_results()
        
        # Prepare data with indicators
        prepared_data = {}
        for ticker, df in data_dict.items():
            if df.empty:
                continue
            
            try:
                prepared = self.strategy.prepare_data(df)
                if prepared.empty:
                    continue
                    
                prepared['ticker'] = ticker
                prepared_data[ticker] = prepared
            except Exception as e:
                print(f"Error preparing {ticker}: {str(e)}")
                continue
        
        if not prepared_data:
            print("Error: No valid data after preparation")
            return self._empty_results()
        
        # Combine data
        combined = self._combine_data(prepared_data)
        
        if combined.empty:
            print("Error: Combined dataset is empty")
            return self._empty_results()
        
        # Initialize portfolio
        num_tickers = len(prepared_data)
        capital_per_ticker = allocate_per_ticker(
            self.config.initial_capital, num_tickers
        )
        
        portfolio = initialize_portfolio(list(prepared_data.keys()), capital_per_ticker)
        
        # Generate all signals
        all_signals = []
        for ticker, df in prepared_data.items():
            try:
                signals = self.strategy.generate_signals(df)
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                print(f"Error generating signals for {ticker}: {str(e)}")
                continue
        
        all_signals.sort(key=lambda s: s.date)
        
        # Run backtest
        dates = sorted(combined.index.unique())
        
        if not dates or len(dates) == 0:
            print("Error: No valid dates in dataset")
            return self._empty_results()
        
        pending_entries = {}
        pending_exits = {}
        
        for i, date in enumerate(dates):
            day_data = self._get_day_data(combined, date)
            current_prices = {row['ticker']: row['Open'] 
                            for _, row in day_data.iterrows()}
            
            # Execute pending exits
            for ticker in list(pending_exits.keys()):
                if ticker in current_prices:
                    pending = pending_exits[ticker]
                    execute_exit(
                        portfolio, ticker, current_prices[ticker],
                        pending['signal_date'], date, self.config
                    )
                    del pending_exits[ticker]
            
            # Execute pending entries
            for ticker in list(pending_entries.keys()):
                if ticker in current_prices:
                    pending = pending_entries[ticker]
                    execute_entry(
                        portfolio, ticker, current_prices[ticker],
                        pending['signal_date'], date, self.config
                    )
                    del pending_entries[ticker]
            
            # Process today's signals
            day_signals = [s for s in all_signals if s.date == date]
            for signal in day_signals:
                ticker = signal.ticker
                account = portfolio['ticker_accounts'][ticker]
                
                if signal.signal_type.value == 1:
                    if account['position'] is None and ticker not in pending_entries:
                        pending_entries[ticker] = {'signal_date': date}
                
                elif signal.signal_type.value == -1:
                    if account['position'] is not None and ticker not in pending_exits:
                        pending_exits[ticker] = {'signal_date': date}
            
            # Record daily value
            close_prices = {row['ticker']: row['Close'] 
                          for _, row in day_data.iterrows()}
            total_value = get_portfolio_value(portfolio, close_prices)
            
            portfolio['daily_values'].append({
                'date': date,
                'total_value': total_value
            })
        
        # Close remaining positions
        if dates and len(dates) > 0:
            final_date = dates[-1]
            
            try:
                final_data = self._get_day_data(combined, final_date)
                final_prices = {row['ticker']: row['Close'] 
                               for _, row in final_data.iterrows()}
                
                for ticker, account in portfolio['ticker_accounts'].items():
                    if account['position'] is not None:
                        if ticker in final_prices:
                            execute_exit(
                                portfolio, ticker, final_prices[ticker],
                                final_date, final_date, self.config
                            )
                        else:
                            print(f"Warning: No price data for {ticker}")
                    
            except Exception as e:
                print(f"Warning: Error closing positions: {str(e)}")
        else:
            print("Warning: No dates available for position closure")
        
        # Create results
        if portfolio['trades']:
            trades_df = pd.DataFrame([self._trade_to_dict(t) for t in portfolio['trades']])
        else:
            trades_df = pd.DataFrame()
        
        if portfolio['daily_values']:
            daily_df = pd.DataFrame(portfolio['daily_values'])
        else:
            daily_df = pd.DataFrame()
        
        # Calculate metrics
        try:
            from .metrics import calculate_all_metrics
            metrics = calculate_all_metrics(portfolio, self.config.initial_capital)
        except Exception as e:
            print(f"Warning: Error calculating metrics: {str(e)}")
            metrics = {}
        
        return BacktestResults(
            portfolio_state=portfolio,
            metrics=metrics,
            trades=trades_df,
            daily_values=daily_df
        )
    
    def _empty_results(self) -> BacktestResults:
        """Return empty results."""
        empty_portfolio = {
            'ticker_accounts': {},
            'trades': [],
            'daily_values': []
        }
        
        return BacktestResults(
            portfolio_state=empty_portfolio,
            metrics={},
            trades=pd.DataFrame(),
            daily_values=pd.DataFrame()
        )
    
    def _combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine ticker DataFrames."""
        if not data_dict:
            return pd.DataFrame()
        
        combined = pd.concat(list(data_dict.values()))
        return combined.sort_index()
    
    def _get_day_data(self, combined: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """Extract data for specific date."""
        day_data = combined.loc[date]
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T
        return day_data
    
    def _trade_to_dict(self, trade) -> Dict:
        """Convert Trade object to dict."""
        return {
            'ticker': trade.ticker,
            'action': trade.action,
            'signal_date': trade.signal_date,
            'execution_date': trade.execution_date,
            'side': trade.side.value,
            'shares': trade.shares,
            'price': trade.price,
            'commission': trade.commission,
            'slippage_cost': trade.slippage_cost,
            'net_pnl': trade.pnl,
            **trade.metadata
        }
