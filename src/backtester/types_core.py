"""Core type definitions and dataclasses."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import pandas as pd


class SignalType(Enum):
    """Signal types."""
    ENTRY = 1
    EXIT = -1
    HOLD = 0


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class Signal:
    """Immutable signal data."""
    ticker: str
    date: datetime
    signal_type: SignalType
    price: float
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class Position:
    """Immutable position data."""
    ticker: str
    side: PositionSide
    shares: float
    entry_price: float
    entry_date: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class Trade:
    """Immutable trade record."""
    ticker: str
    action: str
    signal_date: datetime
    execution_date: datetime
    side: PositionSide
    shares: float
    price: float
    commission: float
    slippage_cost: float
    pnl: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    parameters: Dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'StrategyConfig':
        return cls(
            name=config_dict['name'],
            parameters=config_dict.get('parameters', {})
        )


@dataclass
class TradeConfig:
    """Trading configuration."""
    initial_capital: float
    commission_pct: float
    slippage_pct: float
    slippage_fixed: float
    position_size_pct: float = 0.95
    max_trade_size: float = 50000
    min_trade_size: float = 100
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TradeConfig':
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class BacktestResults:
    """Backtest results container."""
    portfolio_state: Dict
    metrics: Dict
    trades: pd.DataFrame
    daily_values: pd.DataFrame
