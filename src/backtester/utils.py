"""Utility functions."""
import yaml
import json
from pathlib import Path
from typing import Dict, Union
from .types_core import StrategyConfig, TradeConfig


def load_yaml(filepath: Union[str, Path]) -> Dict:
    """Load YAML file (pure I/O function)."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file (pure I/O function)."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict, filepath: Union[str, Path]):
    """Save dict to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def save_json(data: Dict, filepath: Union[str, Path]):
    """Save dict to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def create_strategy_config(config_dict: Dict) -> StrategyConfig:
    """Factory function for StrategyConfig."""
    return StrategyConfig.from_dict(config_dict)


def create_trade_config(config_dict: Dict) -> TradeConfig:
    """Factory function for TradeConfig."""
    return TradeConfig.from_dict(config_dict)


def load_configs(strategy_config_path: str, 
                default_config_path: str = 'configs/default_settings.yaml'):
    """
    Load and merge configurations.
    
    Returns tuple of (StrategyConfig, TradeConfig, full_config_dict)
    """
    # Load default settings
    default = load_yaml(default_config_path)
    
    # Load strategy config
    strategy_dict = load_yaml(strategy_config_path)
    
    # Create config objects
    strategy_config = create_strategy_config(strategy_dict)
    trade_config = create_trade_config(default['trade'])
    
    # Merge for full config
    full_config = {
        'strategy': strategy_dict,
        'trade': default['trade'],
        'data': default.get('data', {}),
        'tickers': strategy_dict.get('tickers', [])
    }
    
    return strategy_config, trade_config, full_config
