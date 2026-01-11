"""VPN Strategy implementation."""
from typing import List
import pandas as pd
from backtester.strategy_base import (
    Strategy, 
    create_signal, 
    detect_threshold_cross_above,
    detect_crossover_below
)
from backtester.types_core import Signal, StrategyConfig
from backtester.indicators import vpn, rsi, sma, ema, atr


class VPNStrategy(Strategy):
    """Volume Pressure Number strategy."""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Extract parameters with defaults
        self.vpn_period = self.params.get('vpn_period', 30)
        self.vpn_smooth = self.params.get('vpn_smooth', 3)
        self.vpn_ma_period = self.params.get('vpn_ma_period', 30)
        self.vpn_critical = self.params.get('vpn_critical', 5.0)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_max_value = self.params.get('rsi_max_value', 70)
        self.rsi_min_value = self.params.get('rsi_min_value', 30)
        self.price_ma_period = self.params.get('price_ma_period', 50)
        self.price_ma_type = self.params.get('price_ma_type', 'sma')
        self.atr_period = self.params.get('atr_period', 14)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add VPN indicators to data."""
        result = data.copy()
        
        # Add VPN indicators
        vpn_result = vpn(result, self.vpn_period, 
                        self.vpn_smooth, self.vpn_ma_period)
        result['VPN'] = vpn_result['vpn']
        result['VPN_SMOOTHED'] = vpn_result['vpn_smoothed']
        result['VPN_MA'] = vpn_result['vpn_ma']
        
        # Add RSI
        result['RSI'] = rsi(result['Close'], self.rsi_period)
        
        # Add Price MA
        if self.price_ma_type == 'ema':
            result['PRICE_MA'] = ema(result['Close'], self.price_ma_period)
        else:
            result['PRICE_MA'] = sma(result['Close'], self.price_ma_period)
        
        # Add ATR
        result['ATR'] = atr(result['High'], result['Low'], 
                           result['Close'], self.atr_period)
        
        # Drop NaN rows
        result = result.dropna()
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate VPN signals (pure function).
        
        Entry: VPN crosses above critical + RSI < max + Price > MA
        Exit: VPN crosses below MA + RSI > min
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
                signal_name='VPN_ENTRY'
            ))
        
        # Create exit signals
        for idx in data.index[exit_cond]:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=-1,
                price=row['Close'],
                signal_name='VPN_EXIT'
            ))
        
        return signals
    
    def _detect_entry(self, data: pd.DataFrame) -> pd.Series:
        """Detect entry conditions (pure helper)."""
        # VPN crosses above critical level
        vpn_cross = detect_threshold_cross_above(
            data, 'VPN_SMOOTHED', self.vpn_critical
        )
        
        # RSI not overbought
        rsi_cond = data['RSI'] < self.rsi_max_value
        
        # Price above MA
        price_cond = data['Close'] > data['PRICE_MA']
        
        return vpn_cross & rsi_cond & price_cond
    
    def _detect_exit(self, data: pd.DataFrame) -> pd.Series:
        """Detect exit conditions (pure helper)."""
        # VPN crosses below MA
        vpn_cross = detect_crossover_below(data, 'VPN_SMOOTHED', 'VPN_MA')
        
        # RSI has momentum
        rsi_cond = data['RSI'] > self.rsi_min_value
        
        return vpn_cross & rsi_cond
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        required = ['vpn_period', 'rsi_period', 'price_ma_period']
        return all(param in self.params for param in required)
