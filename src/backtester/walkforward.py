"""
Walk-Forward Analysis with Optuna Optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import optuna
from optuna.samplers import TPESampler
import warnings
import random
warnings.filterwarnings('ignore')

from .backtest_engine import BacktestEngine
from .types_core import StrategyConfig, TradeConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

def set_random_seed(seed: int = 2570):
    random.seed(seed)
    np.random.seed(seed)
  

def create_wf_config(training_days: int = 252,
                     testing_days: int = 63,
                     holdout_days: int = 21,
                     min_trades: int = 10,
                     n_trials: int = 100,
                     n_jobs: int = 1,
                     timeout: int = 3600,
                     random_seed: int = 42) -> Dict:
    return {
        'training_days': training_days,
        'testing_days': testing_days,
        'holdout_days': holdout_days,
        'min_trades': min_trades,
        'n_trials': n_trials,
        'n_jobs': n_jobs,
        'timeout': timeout,
        'random_seed': random_seed
    }


# ============================================================================
# DATA FILTERING
# ============================================================================

def filter_data_by_date(data_dict: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> Dict[str, pd.DataFrame]:
    filtered = {}
    for ticker, df in data_dict.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df[mask].copy()
        if not filtered_df.empty:
            filtered[ticker] = filtered_df
    return filtered


def get_date_range(data_dict: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime]:
    all_dates = set()
    for df in data_dict.values():
        all_dates.update(df.index)
    dates = sorted(all_dates)
    return dates[0], dates[-1]


# ============================================================================
# WINDOW GENERATION
# ============================================================================

def generate_windows(start_date: datetime,
                     end_date: datetime,
                     training_days: int,
                     testing_days: int,
                     holdout_days: int) -> List[Dict]:
    windows = []
    current_start = start_date
    
    while True:
        train_start = current_start
        train_end = train_start + timedelta(days=training_days)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=testing_days)
        holdout_start = test_end + timedelta(days=1)
        holdout_end = holdout_start + timedelta(days=holdout_days)
        
        if holdout_end > end_date:
            break
        
        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'holdout_start': holdout_start,
            'holdout_end': holdout_end
        })
        
        current_start = test_start
    
    return windows


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def extract_closed_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    
    if 'action' in trades_df.columns:
        closed = trades_df[trades_df['action'] == 'CLOSE'].copy()
        if not closed.empty:
            return closed
    
    pnl_col = None
    for col in ['net_pnl', 'pnl', 'profit', 'return']:
        if col in trades_df.columns:
            pnl_col = col
            break
    
    if pnl_col:
        closed = trades_df[trades_df[pnl_col].notna()].copy()
        if not closed.empty:
            return closed
    
    return trades_df.copy()


def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict:
    if trades_df is None or trades_df.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'total_pnl': 0.0
        }
    
    closed_trades = extract_closed_trades(trades_df)
    
    if closed_trades.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'total_pnl': 0.0
        }
    
    pnl_col = None
    for col in ['net_pnl', 'pnl', 'profit', 'return', 'profit_loss']:
        if col in closed_trades.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        return {
            'total_trades': len(closed_trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'total_pnl': 0.0
        }
    
    pnl_values = closed_trades[pnl_col].replace([np.inf, -np.inf], np.nan).dropna()
    
    if pnl_values.empty:
        return {
            'total_trades': len(closed_trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'total_pnl': 0.0
        }
    
    total_trades = len(pnl_values)
    winning_trades = len(pnl_values[pnl_values > 0])
    losing_trades = len(pnl_values[pnl_values < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    
    total_wins = wins.sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else (float('inf') if total_wins > 0 else 0.0)
    expectancy = pnl_values.mean()
    total_pnl = pnl_values.sum()
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': min(profit_factor, 999.99),
        'expectancy': expectancy,
        'total_pnl': total_pnl
    }


def calculate_returns_metrics(daily_values: pd.DataFrame, initial_capital: float) -> Dict:
    if daily_values is None or daily_values.empty:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0
        }
    
    if 'total_value' not in daily_values.columns:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0
        }
    
    values = daily_values['total_value'].replace([np.inf, -np.inf], np.nan).dropna().values
    
    if len(values) < 2:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0
        }
    
    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0
        }
    
    total_return = ((values[-1] - initial_capital) / initial_capital) * 100
    
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
    
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
    sortino_ratio = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    calmar_ratio = (total_return / max_drawdown) if max_drawdown > 0 else 0.0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }


# ============================================================================
# BACKTEST EXECUTION
# ============================================================================

def run_backtest(data_dict: Dict[str, pd.DataFrame],
                params: Dict,
                strategy_class: type,
                strategy_name: str,
                trade_config: TradeConfig) -> Dict:
    try:
        strategy_config = StrategyConfig(name=strategy_name, parameters=params)
        strategy = strategy_class(strategy_config)
        engine = BacktestEngine(strategy, trade_config)
        results = engine.run(data_dict)
        
        trade_metrics = calculate_trade_metrics(results.trades)
        returns_metrics = calculate_returns_metrics(results.daily_values, trade_config.initial_capital)
        
        return {**trade_metrics, **returns_metrics}
        
    except Exception as e:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'total_pnl': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'error': str(e)
        }


# ============================================================================
# PRINTING UTILITIES
# ============================================================================

def format_param_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_best_parameters(params: Dict, study: optuna.Study, window_idx: int):
    print(f"\nWindow {window_idx} - Best Parameters")
    print("-" * 60)
    print(params)
    param_groups = {}
    for param_name, value in sorted(params.items()):
        prefix = param_name.split('_')[0].upper()
        if prefix not in param_groups:
            param_groups[prefix] = []
        param_groups[prefix].append((param_name, value))
    
    for group_name, param_list in sorted(param_groups.items()):
        print(f"\n  {group_name}:")
        for param_name, value in param_list:
            formatted_value = format_param_value(value)
            print(f"    {param_name:25s} = {formatted_value:>10s}")
    
    print(f"\n  PERFORMANCE:")
    print(f"    {'objective_value':25s} = {study.best_value:>10.4f}")
    
    attrs = study.best_trial.user_attrs
    if 'win_rate' in attrs:
        print(f"    {'win_rate':25s} = {attrs['win_rate']:>9.2%}")
    if 'sharpe_ratio' in attrs:
        print(f"    {'sharpe_ratio':25s} = {attrs['sharpe_ratio']:>10.4f}")
    if 'total_trades' in attrs:
        print(f"    {'total_trades':25s} = {int(attrs['total_trades']):>10d}")
    if 'total_return' in attrs:
        print(f"    {'total_return':25s} = {attrs['total_return']:>9.2f}%")
    if 'max_drawdown' in attrs:
        print(f"    {'max_drawdown':25s} = {attrs['max_drawdown']:>9.2f}%")


def calculate_summary_stats(results_df: pd.DataFrame, prefix: str) -> Dict:
    if results_df.empty:
        return {}
    
    return {
        f'{prefix}_avg_win_rate': results_df[f'{prefix}_win_rate'].mean(),
        f'{prefix}_avg_sharpe': results_df[f'{prefix}_sharpe_ratio'].mean(),
        f'{prefix}_avg_return': results_df[f'{prefix}_total_return'].mean(),
        f'{prefix}_avg_drawdown': results_df[f'{prefix}_max_drawdown'].mean(),
        f'{prefix}_total_trades': results_df[f'{prefix}_total_trades'].sum(),
        f'{prefix}_positive_windows': (results_df[f'{prefix}_total_return'] > 0).sum(),
        f'{prefix}_positive_pct': (results_df[f'{prefix}_total_return'] > 0).mean()
    }


def print_summary(results_df: pd.DataFrame):
    print(f"\n{'='*80}")
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    test_stats = calculate_summary_stats(results_df, 'test')
    holdout_stats = calculate_summary_stats(results_df, 'holdout')
    
    print("\nOUT-OF-SAMPLE TESTING:")
    print(f"  Win Rate:         {test_stats['test_avg_win_rate']:.2%}")
    print(f"  Sharpe Ratio:     {test_stats['test_avg_sharpe']:.4f}")
    print(f"  Avg Return:       {test_stats['test_avg_return']:.2f}%")
    print(f"  Avg Drawdown:     {test_stats['test_avg_drawdown']:.2f}%")
    print(f"  Total Trades:     {test_stats['test_total_trades']:.0f}")
    print(f"  Positive Windows: {test_stats['test_positive_windows']}/{len(results_df)} ({test_stats['test_positive_pct']:.1%})")
    
    print("\nHOLDOUT VALIDATION:")
    print(f"  Win Rate:         {holdout_stats['holdout_avg_win_rate']:.2%}")
    print(f"  Sharpe Ratio:     {holdout_stats['holdout_avg_sharpe']:.4f}")
    print(f"  Avg Return:       {holdout_stats['holdout_avg_return']:.2f}%")
    print(f"  Avg Drawdown:     {holdout_stats['holdout_avg_drawdown']:.2f}%")
    print(f"  Total Trades:     {holdout_stats['holdout_total_trades']:.0f}")
    print(f"  Positive Windows: {holdout_stats['holdout_positive_windows']}/{len(results_df)} ({holdout_stats['holdout_positive_pct']:.1%})")


# ============================================================================
# OPTIMIZATION
# ============================================================================

def calculate_objective(win_rate: float,
                       sharpe_ratio: float,
                       win_rate_weight: float = 0.35,
                       sharpe_weight: float = 0.65) -> float:
    normalized_sharpe = max(0, min(1, (sharpe_ratio + 3) / 6))
    return win_rate_weight * win_rate + sharpe_weight * normalized_sharpe


def create_objective_function(data_dict: Dict[str, pd.DataFrame],
                             param_space: Dict[str, Tuple],
                             strategy_class: type,
                             strategy_name: str,
                             trade_config: TradeConfig,
                             min_trades: int,
                             objective_weights: Tuple[float, float]) -> Callable:
    
    def objective(trial: optuna.Trial) -> float:
        params = {}
        for param_name, (param_type, *args) in param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, args[0], args[1])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, args[0], args[1])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])
        
        try:
            results = run_backtest(data_dict, params, strategy_class, strategy_name, trade_config)
            
            if 'error' in results or results['total_trades'] < min_trades:
                return -999.0
            
            objective_value = calculate_objective(
                results['win_rate'],
                results['sharpe_ratio'],
                objective_weights[0],
                objective_weights[1]
            )
            
            trial.set_user_attr('win_rate', results['win_rate'])
            trial.set_user_attr('sharpe_ratio', results['sharpe_ratio'])
            trial.set_user_attr('total_trades', results['total_trades'])
            trial.set_user_attr('total_return', results['total_return'])
            trial.set_user_attr('max_drawdown', results['max_drawdown'])
            
            return objective_value
            
        except Exception:
            return -999.0
    
    return objective


def optimize_parameters(data_dict: Dict[str, pd.DataFrame],
                       param_space: Dict[str, Tuple],
                       strategy_class: type,
                       strategy_name: str,
                       trade_config: TradeConfig,
                       wf_config: Dict,
                       window_idx: int,
                       objective_weights: Tuple[float, float]) -> Tuple[Dict, optuna.Study]:
    
    seed = wf_config.get('random_seed', 42) + window_idx
    set_random_seed(seed)
    
    objective_fn = create_objective_function(
        data_dict, param_space, strategy_class, strategy_name,
        trade_config, wf_config['min_trades'], objective_weights
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=seed),
        study_name=f'{strategy_name}_window_{window_idx}'
    )
    
    study.optimize(
        objective_fn,
        n_trials=wf_config['n_trials'],
        timeout=wf_config['timeout'],
        n_jobs=wf_config['n_jobs'],
        show_progress_bar=False
    )
    
    return study.best_params, study


# ============================================================================
# WINDOW PROCESSING
# ============================================================================

def process_window(window: Dict,
                  window_idx: int,
                  data_dict: Dict[str, pd.DataFrame],
                  param_space: Dict[str, Tuple],
                  strategy_class: type,
                  strategy_name: str,
                  trade_config: TradeConfig,
                  wf_config: Dict,
                  objective_weights: Tuple[float, float],
                  verbose: bool) -> Optional[Dict]:
    
    seed = wf_config.get('random_seed', 42) + window_idx
    set_random_seed(seed)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"WINDOW {window_idx} (Seed: {seed})")
        print(f"{'='*80}")
        print(f"Training:  {window['train_start'].date()} to {window['train_end'].date()}")
        print(f"Testing:   {window['test_start'].date()} to {window['test_end'].date()}")
        print(f"Holdout:   {window['holdout_start'].date()} to {window['holdout_end'].date()}")
    
    train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
    if not train_data:
        if verbose:
            print("Skipping: no training data")
        return None
    
    if verbose:
        print(f"\nOptimizing parameters...")
    
    best_params, study = optimize_parameters(
        train_data, param_space, strategy_class, strategy_name,
        trade_config, wf_config, window_idx, objective_weights
    )
    
    if not best_params:
        if verbose:
            print("Skipping: optimization failed")
        return None
    
    if verbose:
        print_best_parameters(best_params, study, window_idx)
    
    test_data = filter_data_by_date(data_dict, window['test_start'], window['test_end'])
    if not test_data:
        if verbose:
            print("\nSkipping: no test data")
        return None
    
    if verbose:
        print(f"\nTesting out-of-sample...")
    
    test_results = run_backtest(test_data, best_params, strategy_class, strategy_name, trade_config)
    
    if verbose:
        print(f"  Trades: {test_results['total_trades']}, "
              f"Win Rate: {test_results['win_rate']:.2%}, "
              f"Sharpe: {test_results['sharpe_ratio']:.4f}, "
              f"Return: {test_results['total_return']:.2f}%")
    
    holdout_data = filter_data_by_date(data_dict, window['holdout_start'], window['holdout_end'])
    if not holdout_data:
        if verbose:
            print("\nSkipping: no holdout data")
        return None
    
    if verbose:
        print(f"Holdout validation...")
    
    holdout_results = run_backtest(holdout_data, best_params, strategy_class, strategy_name, trade_config)
    
    if verbose:
        print(f"  Trades: {holdout_results['total_trades']}, "
              f"Win Rate: {holdout_results['win_rate']:.2%}, "
              f"Sharpe: {holdout_results['sharpe_ratio']:.4f}, "
              f"Return: {holdout_results['total_return']:.2f}%")
    
    result = {
        'window': window_idx,
        'seed': seed,
        'train_start': window['train_start'],
        'train_end': window['train_end'],
        'test_start': window['test_start'],
        'test_end': window['test_end'],
        'holdout_start': window['holdout_start'],
        'holdout_end': window['holdout_end'],
        'n_trials': len(study.trials),
        'best_objective': study.best_value
    }
    
    result.update({f'best_{k}': v for k, v in best_params.items()})
    result.update({f'test_{k}': v for k, v in test_results.items() if k != 'error'})
    result.update({f'holdout_{k}': v for k, v in holdout_results.items() if k != 'error'})
    
    return result


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def walk_forward_analysis(data_dict: Dict[str, pd.DataFrame],
                         param_space: Dict[str, Tuple],
                         strategy_class: type,
                         strategy_name: str,
                         trade_config: TradeConfig,
                         wf_config: Dict,
                         objective_weights: Tuple[float, float] = (0.35, 0.65),
                         verbose: bool = False) -> Tuple[pd.DataFrame, List[optuna.Study]]:
    
    set_random_seed(wf_config.get('random_seed', 42))
    
    start_date, end_date = get_date_range(data_dict)
    
    windows = generate_windows(
        start_date, end_date,
        wf_config['training_days'],
        wf_config['testing_days'],
        wf_config['holdout_days']
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD ANALYSIS")
        print(f"{'='*80}")
        print(f"Strategy:      {strategy_name}")
        print(f"Random Seed:   {wf_config.get('random_seed', 42)}")
        print(f"Date Range:    {start_date.date()} to {end_date.date()}")
        print(f"Total Windows: {len(windows)}")
        print(f"Trials/Window: {wf_config['n_trials']}")
    
    results = []
    studies = []
    
    for i, window in enumerate(windows, 1):
        result = process_window(
            window, i, data_dict, param_space,
            strategy_class, strategy_name, trade_config,
            wf_config, objective_weights, verbose
        )
        
        if result:
            results.append(result)
            train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
            if train_data:
                _, study = optimize_parameters(
                    train_data, param_space, strategy_class, strategy_name,
                    trade_config, wf_config, i, objective_weights
                )
                studies.append(study)
    
    results_df = pd.DataFrame(results)
    
    if verbose and not results_df.empty:
        print_summary(results_df)
    
    return results_df, studies


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def calculate_param_importance(studies: List[optuna.Study]) -> pd.DataFrame:
    sensitivity_results = []
    
    for idx, study in enumerate(studies, 1):
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, score in importance.items():
                sensitivity_results.append({
                    'window': idx,
                    'parameter': param,
                    'importance_score': score
                })
        except Exception:
            continue
    
    return pd.DataFrame(sensitivity_results) if sensitivity_results else pd.DataFrame()


def aggregate_param_importance(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    if sensitivity_df.empty:
        return pd.DataFrame()
    
    return sensitivity_df.groupby('parameter')['importance_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).sort_values('mean', ascending=False)


def get_param_distributions(studies: List[optuna.Study], top_n: int = 5) -> Dict[str, Dict]:
    sensitivity_df = calculate_param_importance(studies)
    if sensitivity_df.empty:
        return {}
    
    importance = aggregate_param_importance(sensitivity_df)
    top_params = importance.head(top_n).index.tolist()
    
    distributions = {}
    for param in top_params:
        values = []
        for study in studies:
            if param in study.best_params:
                values.append(study.best_params[param])
        
        if values:
            distributions[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return distributions


def analyze_param_stability(results_df: pd.DataFrame, param_name: str) -> Dict:
    param_col = f'best_{param_name}'
    if param_col not in results_df.columns:
        return {}
    
    values = results_df[param_col].dropna()
    
    if values.empty:
        return {}
    
    mean_val = values.mean()
    std_val = values.std()
    cv = (std_val / mean_val) if mean_val != 0 else 0.0
    
    from scipy import stats
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    
    return {
        'parameter': param_name,
        'mean': mean_val,
        'median': values.median(),
        'std': std_val,
        'min': values.min(),
        'max': values.max(),
        'cv': cv,
        'trend_slope': slope,
        'trend_r2': r_value**2,
        'is_stable': cv < 0.3
    }


def analyze_all_params_stability(results_df: pd.DataFrame) -> pd.DataFrame:
    param_cols = [col for col in results_df.columns if col.startswith('best_')]
    
    stability_results = []
    for col in param_cols:
        param_name = col.replace('best_', '')
        stability = analyze_param_stability(results_df, param_name)
        if stability:
            stability_results.append(stability)
    
    return pd.DataFrame(stability_results).sort_values('cv') if stability_results else pd.DataFrame()


# ============================================================================
# EXPORT RESULTS
# ============================================================================

def export_results(results_df: pd.DataFrame,
                  sensitivity_df: pd.DataFrame,
                  studies: List[optuna.Study],
                  output_dir: str = "walk_forward_results"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if not results_df.empty:
        results_df.to_csv(f"{output_dir}/walk_forward_results.csv", index=False)
        print(f"\nResults exported to {output_dir}/walk_forward_results.csv")
    
    if not sensitivity_df.empty:
        importance = aggregate_param_importance(sensitivity_df)
        importance.to_csv(f"{output_dir}/parameter_sensitivity.csv")
        print(f"Sensitivity analysis exported to {output_dir}/parameter_sensitivity.csv")
    
    history_data = []
    for idx, study in enumerate(studies, 1):
        for trial in study.trials:
            history_data.append({
                'window': idx,
                'trial': trial.number,
                'value': trial.value,
                **trial.params,
                **trial.user_attrs
            })
    
    if history_data:
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(f"{output_dir}/optimization_history.csv", index=False)
        print(f"Optimization history exported to {output_dir}/optimization_history.csv")