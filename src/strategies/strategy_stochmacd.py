"""
Stochastic MACD Oscillator Strategy.
"""
from typing import List
import pandas as pd
from backtester.strategy_base import Strategy, create_signal
from backtester.types_core import Signal, StrategyConfig
from backtester.indicators import (
    stochastic_macd,
    rsi,
    atr,
    ema
)


class StochasticMACDStrategy(Strategy):
    """
    Stochastic MACD Oscillator Crossover Strategy.
    
    The Stochastic MACD combines stochastic oscillator normalization
    with MACD momentum signals, providing clearer overbought/oversold
    zones compared to traditional MACD.
    
    Entry Logic:
        - STMACD crosses above Signal line (bullish crossover)
        - Optional: STMACD is below oversold threshold
        - Optional: Price above trend filter (EMA)
    
    Exit Logic:
        - STMACD crosses below Signal line (bearish crossover)
        - Optional: STMACD is above overbought threshold
    
    References:
        Apirine, V. (2019). "Stochastic MACD Oscillator"
        TASC Magazine, November 2019
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Stochastic MACD Parameters
        self.period = self.params.get('period', 45)
        self.fast_period = self.params.get('fast_period', 12)
        self.slow_period = self.params.get('slow_period', 26)
        self.signal_period = self.params.get('signal_period', 9)
        
        # Overbought/Oversold Levels
        self.overbought = self.params.get('overbought', 10)
        self.oversold = self.params.get('oversold', -10)
        
        # Optional Filters
        self.use_threshold_filter = self.params.get('use_threshold_filter', True)
        self.use_trend_filter = self.params.get('use_trend_filter', False)
        self.trend_ema_period = self.params.get('trend_ema_period', 50)
        
        # Optional RSI Filter
        self.use_rsi_filter = self.params.get('use_rsi_filter', False)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_threshold = self.params.get('rsi_threshold', 50)
        
        # Risk Management
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_stop_multiplier = self.params.get('atr_stop_multiplier', 2.0)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic MACD indicators to data."""
        result = data.copy()
        
        # Add Stochastic MACD
        stmacd = stochastic_macd(
            result,
            period=self.period,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period
        )
        
        result['STMACD'] = stmacd['stmacd']
        result['STMACD_SIGNAL'] = stmacd['signal']
        result['STMACD_HISTOGRAM'] = stmacd['histogram']
        
        # Add trend filter (optional)
        if self.use_trend_filter:
            result['TREND_EMA'] = ema(result['Close'], self.trend_ema_period)
        
        # Add RSI filter (optional)
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
        Generate Stochastic MACD signals .
        
        Entry: STMACD crosses above Signal line
        Exit: STMACD crosses below Signal line
        """
        signals = []
        
        # Entry conditions (bullish crossover)
        entry_cond = self._detect_entry(data)
        
        # Exit conditions (bearish crossover)
        exit_cond = self._detect_exit(data)
        
        # Create entry signals
        for idx in data.index[entry_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=1,
                price=row['Close'],
                signal_name='STMACD_ENTER'
            ))
        
        # Create exit signals
        for idx in data.index[exit_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=-1,
                price=row['Close'],
                signal_name='STMACD_EXIT'
            ))
        
        return signals
    
    def _detect_entry(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect entry conditions (pure helper).
        
        Entry: STMACD crosses above Signal line (bullish crossover)
        """
        # Basic crossover: STMACD crosses above Signal
        bullish_cross = (
            (data['STMACD'] >= data['STMACD_SIGNAL']) &
            (data['STMACD'].shift(1) < data['STMACD_SIGNAL'].shift(1)) 
        )
        
        entry = bullish_cross
        
        # Optional: Only enter when STMACD is below oversold (coming from oversold)
        if self.use_threshold_filter:
            oversold_filter = data['STMACD'] < self.overbought  # Not already overbought
            entry = entry & oversold_filter
        
        # Optional: Trend filter (only long when price > EMA)
        if self.use_trend_filter and 'TREND_EMA' in data.columns:
            trend_filter = data['Close'] > data['TREND_EMA']
            entry = entry & trend_filter
        
        # Optional: RSI filter (momentum confirmation)
        if self.use_rsi_filter and 'RSI' in data.columns:
            rsi_filter = data['RSI'] > self.rsi_threshold
            entry = entry & rsi_filter
        
        return entry
    
    def _detect_exit(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect exit conditions (pure helper).
        
        Exit: STMACD crosses below Signal line (bearish crossover)
        """
        # Basic crossover: STMACD crosses below Signal
        bearish_cross = (
            (data['STMACD'] < data['STMACD_SIGNAL']) &
            (data['STMACD'].shift(1) >= data['STMACD_SIGNAL'].shift(1))&
            (data['STMACD'] > 0) 
        )
        
        exit_signal = bearish_cross
        
        # Optional: Exit early if extremely overbought
        if self.use_threshold_filter:
            overbought_exit = (
                (data['STMACD'].shift(1) >= self.overbought) &
                (data['STMACD'] < data['STMACD'].shift(1))  # Started declining
            )
            exit_signal = exit_signal | overbought_exit
        
        return exit_signal
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required = ['period', 'fast_period', 'slow_period', 'signal_period']
        return all(param in self.params for param in required)
