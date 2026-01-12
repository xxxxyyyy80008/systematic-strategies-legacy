"""
Bollinger Bands Breakout Strategy with Bullish Engulfing Pattern.
"""
from typing import List
import pandas as pd
from backtester.strategy_base import Strategy, create_signal
from backtester.types_core import Signal, StrategyConfig
from backtester.indicators import (
    bollinger_bands,
    bullish_engulfing,
    bbands_bounce,
    bbands_breakout,
    rsi,
    atr
)


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Breakout Strategy with Candlestick Confirmation.
    
    Entry Logic:
        - Bullish Engulfing pattern detected
        - Bollinger Bands bounce signal (price bounces off lower band)
        - Optional: RSI confirmation
    
    Exit Logic:
        - Price breaks above upper Bollinger Band
        - Or stop loss triggered
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Bollinger Bands Parameters
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std_dev = self.params.get('bb_std_dev', 2.0)
        
        # RSI Parameters (optional filter)
        self.use_rsi_filter = self.params.get('use_rsi_filter', False)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_oversold = self.params.get('rsi_oversold', 30)
        
        # ATR for stops
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_stop_multiplier = self.params.get('atr_stop_multiplier', 2.0)
        
        # Pattern confirmation
        self.require_engulfing = self.params.get('require_engulfing', True)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands indicators to data."""
        result = data.copy()
        
        # Add Bollinger Bands
        bb = bollinger_bands(result['Close'], self.bb_period, self.bb_std_dev)
        for key, series in bb.items():
            result[key.upper()] = series
        
        # Add Bullish Engulfing pattern
        result['BULL_ENGULF'] = bullish_engulfing(result['Open'], result['Close'])
        
        # Add BBands bounce signal
        result['BBANDS_BOUNCE'] = bbands_bounce(
            result['Close'], 
            result['Low'], 
            result['BB_LOWER']
        )
        
        # Add BBands breakout signal
        result['BBANDS_BREAKOUT'] = bbands_breakout(
            result['High'], 
            result['BB_UPPER']
        )
        
        # Add RSI (optional filter)
        if self.use_rsi_filter:
            result['RSI'] = rsi(result['Close'], self.rsi_period)
        
        # Add ATR for stops
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
        Generate Bollinger Bands signals .
        
        Entry: Bullish Engulfing + BBands Bounce (+ optional RSI)
        Exit: Price breaks above upper band
        """
        signals = []
        
        # Entry conditions
        entry_cond = self._detect_entry(data)
        
        # Exit conditions
        exit_cond = self._detect_exit(data)
        
        # Create entry signals
        for idx in data.index[entry_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=1,
                price=row['Close'],
                signal_name='BBANDS_ENTRY'
            ))
        
        # Create exit signals
        for idx in data.index[exit_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=-1,
                price=row['Close'],
                signal_name='BBANDS_EXIT'
            ))
        
        return signals
    
    def _detect_entry(self, data: pd.DataFrame) -> pd.Series:
        """Detect entry conditions (pure helper)."""
        # BBands bounce (required)
        bounce = data['BBANDS_BOUNCE']
        
        # Bullish Engulfing (optional)
        if self.require_engulfing:
            pattern = data['BULL_ENGULF']
            entry = bounce & pattern
        else:
            entry = bounce
        
        # RSI filter (optional)
        if self.use_rsi_filter and 'RSI' in data.columns:
            rsi_oversold = data['RSI'] < self.rsi_oversold
            entry = entry & rsi_oversold
        
        return entry
    
    def _detect_exit(self, data: pd.DataFrame) -> pd.Series:
        """Detect exit conditions (pure helper)."""
        # Price breaks above upper band
        breakout = data['BBANDS_BREAKOUT']
        
        return breakout
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required = ['bb_period', 'bb_std_dev']
        return all(param in self.params for param in required)
