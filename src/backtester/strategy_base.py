from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
from .types_core import Signal, SignalType, StrategyConfig


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.params = config.parameters
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add required indicators to data.
        
        Must return new DataFrame (immutable approach).
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from data.
        
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        return True


# Helper functions (pure)
def create_signal(ticker: str, date: pd.Timestamp, signal_type: int,
                 price: float, signal_name: str) -> Signal:
    """Factory function for creating signals (pure)."""
    signal_map = {1: SignalType.ENTRY, -1: SignalType.EXIT, 0: SignalType.HOLD}
    
    return Signal(
        ticker=ticker,
        date=date,
        signal_type=signal_map.get(signal_type, SignalType.HOLD),
        price=price,
        metadata={'signal_name': signal_name}
    )


def detect_crossover_above(df: pd.DataFrame, col1: str, col2: str) -> pd.Series:
    """Detect when col1 crosses above col2 ."""
    return (df[col1] > df[col2]) & (df[col1].shift(1) <= df[col2].shift(1))


def detect_crossover_below(df: pd.DataFrame, col1: str, col2: str) -> pd.Series:
    """Detect when col1 crosses below col2 ."""
    return (df[col1] < df[col2]) & (df[col1].shift(1) >= df[col2].shift(1))


def detect_threshold_cross_above(df: pd.DataFrame, col: str, threshold: float) -> pd.Series:
    """Detect when column crosses above threshold ."""
    return (df[col] >= threshold) & (df[col].shift(1) < threshold)


def detect_threshold_cross_below(df: pd.DataFrame, col: str, threshold: float) -> pd.Series:
    """Detect when column crosses below threshold ."""
    return (df[col] < threshold) & (df[col].shift(1) >= threshold)


def is_at_lowest(df: pd.DataFrame, col: str, period: int) -> pd.Series:
    """Check if column is at lowest value in period ."""
    return df[col].rolling(window=period, min_periods=period).min() == df[col]


def is_at_highest(df: pd.DataFrame, col: str, period: int) -> pd.Series:
    """Check if column is at highest value in period ."""
    return df[col].rolling(window=period, min_periods=period).max() == df[col]
