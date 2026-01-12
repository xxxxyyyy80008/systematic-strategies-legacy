"""AMA-KAMA Strategy implementation."""
from typing import List
import pandas as pd
from backtester.strategy_base import (
    Strategy, 
    create_signal, 
    detect_crossover_above,
    detect_crossover_below
)
from backtester.types_core import Signal, StrategyConfig
from backtester.indicators import ama, kama, ema, rsi, atr


class AMAKAMAStrategy(Strategy):
    """
    Adaptive Moving Average + Kaufman Adaptive Moving Average strategy.
    
    Entry Logic:
        - AMA crosses above EMA (bullish momentum)
        - RSI < 35 (oversold condition)
    
    Exit Logic:
        - AMA crosses below EMA (bearish momentum)
        - RSI > 65 (overbought condition)
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # AMA Parameters
        self.ama_period = self.params.get('ama_period', 10)
        self.ama_fast = self.params.get('ama_fast', 2)
        self.ama_slow = self.params.get('ama_slow', 30)
        
        # KAMA Parameters (optional, for analysis)
        self.kama_er_period = self.params.get('kama_er_period', 10)
        self.kama_fast = self.params.get('kama_fast', 2)
        self.kama_slow = self.params.get('kama_slow', 30)
        self.kama_period = self.params.get('kama_period', 20)
        
        # EMA Parameters
        self.ema_period = self.params.get('ema_period', 20)
        
        # RSI Parameters
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_entry_max = self.params.get('rsi_entry_max', 35)
        self.rsi_exit_min = self.params.get('rsi_exit_min', 65)
        
        # ATR for stops (optional)
        self.atr_period = self.params.get('atr_period', 14)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add AMA-KAMA indicators to data."""
        result = data.copy()
        
        # Add AMA (requires OHLC data)
        result['AMA'] = ama(
            result[['High', 'Low', 'Close']], 
            period=self.ama_period,
            fast_period=self.ama_fast,
            slow_period=self.ama_slow
        )
        
        # Add KAMA (for comparison/analysis)
        result['KAMA'] = kama(
            result['Close'],
            er_period=self.kama_er_period,
            ema_fast=self.kama_fast,
            ema_slow=self.kama_slow,
            period=self.kama_period
        )
        
        # Add EMA
        result['EMA'] = ema(result['Close'], self.ema_period)
        
        # Add RSI
        result['RSI'] = rsi(result['Close'], self.rsi_period)
        
        # Add ATR (optional for stops)
        result['ATR'] = atr(
            result['High'], 
            result['Low'], 
            result['Close'], 
            self.atr_period
        )
        
        # Drop NaN rows
        # result = result.dropna()
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate AMA-KAMA signals .
        
        Entry: AMA crosses above EMA + RSI < 35
        Exit: AMA crosses below EMA + RSI > 65
        """
        signals = []
        
        # Entry condition
        entry_cond = self._detect_entry(data)
        
        # Exit condition
        exit_cond = self._detect_exit(data)
        
        # Create entry signals
        for idx in data.index[entry_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=1,
                price=row['Close'],
                signal_name='AMA_KAMA_ENTRY'
            ))
        
        # Create exit signals
        for idx in data.index[exit_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=-1,
                price=row['Close'],
                signal_name='AMA_KAMA_EXIT'
            ))
        
        return signals
    
    def _detect_entry(self, data: pd.DataFrame) -> pd.Series:
        """Detect entry conditions (pure helper)."""
        # AMA crosses above EMA
        ama_cross_above = detect_crossover_above(data, 'AMA', 'EMA')
        
        # RSI oversold
        # rsi_oversold = data['RSI'] < self.rsi_entry_max
        rsi_condition = (data['RSI'] < self.rsi_entry_max).rolling(3).max() > 0
        
        return ama_cross_above & rsi_condition
    
    def _detect_exit(self, data: pd.DataFrame) -> pd.Series:
        """Detect exit conditions (pure helper)."""
        # AMA crosses below EMA
        ama_cross_below = detect_crossover_below(data, 'AMA', 'EMA')
        
        # RSI overbought
        rsi_overbought = data['RSI'] > self.rsi_exit_min
        
        return ama_cross_below & rsi_overbought
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required = ['ama_period', 'ema_period', 'rsi_period']
        return all(param in self.params for param in required)
