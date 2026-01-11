"""MABW Strategy implementation."""
from typing import List
import pandas as pd
from backtester.strategy_base import (
    Strategy, 
    create_signal, 
    detect_crossover_above,
    is_at_lowest
)
from backtester.types_core import Signal, StrategyConfig
from backtester.indicators import mabw, ema, atr


class MABWStrategy(Strategy):
    """Moving Average Band Width strategy."""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Extract parameters with defaults
        self.fast_period = self.params.get('fast_period', 10)
        self.slow_period = self.params.get('slow_period', 60)
        self.multiplier = self.params.get('multiplier', 1.0)
        self.ema_period = self.params.get('ema_period', 20)
        self.atr_period = self.params.get('atr_period', 14)
        self.mabw_llv_period = self.params.get('mabw_llv_period', 10)
        self.mab_width_critical = self.params.get('mab_width_critical', 30)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add MABW indicators to data."""
        result = data.copy()
        
        # Add MABW bands
        mabw_result = mabw(result['Close'], self.fast_period, 
                          self.slow_period, self.multiplier)
        result['MAB_UPPER'] = mabw_result['upper']
        result['MAB_MIDDLE'] = mabw_result['middle']
        result['MAB_LOWER'] = mabw_result['lower']
        result['MAB_WIDTH'] = mabw_result['width']
        result['MAB_LLV'] = mabw_result['llv']
        
        # Add EMA
        result['EMA'] = ema(result['Close'], self.ema_period)
        
        # Add ATR
        result['ATR'] = atr(result['High'], result['Low'], 
                           result['Close'], self.atr_period)
        
        # Drop NaN rows
        result = result.dropna()
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate MABW signals .
        
        Entry: EMA crosses above MAB_UPPER + MAB_WIDTH is at LLV
        Exit: MAB_WIDTH crosses above critical level
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
                signal_name='MABW_ENTRY'
            ))
        
        # Create exit signals
        for idx in data.index[exit_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=-1,
                price=row['Close'],
                signal_name='MABW_EXIT'
            ))
        
        return signals
    
    def _detect_entry(self, data: pd.DataFrame) -> pd.Series:
        """Detect entry conditions (pure helper)."""
        # EMA crosses above upper band
        cross_cond = detect_crossover_above(data, 'EMA', 'MAB_UPPER')
        
        # MAB_WIDTH is at LLV
        llv_cond = is_at_lowest(data, 'MAB_WIDTH', self.mabw_llv_period)
        
        return cross_cond & llv_cond
    
    def _detect_exit(self, data: pd.DataFrame) -> pd.Series:
        """Detect exit conditions (pure helper)."""
        return (
            (data['MAB_WIDTH'] > self.mab_width_critical) & 
            (data['MAB_WIDTH'].shift(1) <= self.mab_width_critical)
        )
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required = ['fast_period', 'slow_period', 'ema_period']
        return all(param in self.params for param in required)
