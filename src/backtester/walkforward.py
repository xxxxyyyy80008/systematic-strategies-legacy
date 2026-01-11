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
warnings.filterwarnings('ignore')

from backtester.backtest_engine import BacktestEngine
from backtester.types_core import StrategyConfig, TradeConfig


# CONFIGURATION
def create_wf_config(training_days: int = 252,
                     testing_days: int = 63,
                     holdout_days: int = 21,
                     min_trades: int = 10,
                     n_trials: int = 100,
                     n_jobs: int = -1,
                     timeout: int = 3600) -> Dict:
    return {
        'training_days': training_days,
        'testing_days': testing_days,
        'holdout_days': holdout_days,
        'min_trades': min_trades,
        'n_trials': n_trials,
        'n_jobs': n_jobs,
        'timeout': timeout
    }


# DATA FILTERING
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


# WINDOW GENERATION
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


# BACKTEST EXECUTION
def run_backtest(data_dict: Dict[str, pd.DataFrame],
                params: Dict,
                strategy_class: type,
                strategy_name: str,
                trade_config: TradeConfig) -> Dict:
    strategy_config = StrategyConfig(name=strategy_name, parameters=params)
    strategy = strategy_class(strategy_config)
    engine = BacktestEngine(strategy, trade_config)
    results = engine.run(data_dict)
    
    metrics = results.metrics
    trades_df = results.trades
    
    total_trades = len(trades_df) if not trades_df.empty else 0
    winning_trades = len(trades_df[trades_df['net_pnl'] > 0]) if not trades_df.empty else 0
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': metrics.get('total_return_pct', 0.0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': metrics.get('max_drawdown_pct', 0.0),
        'profit_factor': metrics.get('profit_factor', 0.0),
        'avg_win': metrics.get('avg_win', 0.0),
        'avg_loss': metrics.get('avg_loss', 0.0),
        'expectancy': metrics.get('expectancy', 0.0),
        'sortino_ratio': metrics.get('sortino_ratio', 0.0),
        'calmar_ratio': metrics.get('calmar_ratio', 0.0)
    }


# OBJECTIVE FUNCTION
def calculate_objective(win_rate: float,
                       sharpe_ratio: float,
                       win_rate_weight: float = 0.35,
                       sharpe_weight: float = 0.65) -> float:
    normalized_sharpe = max(0, min(1, (sharpe_ratio + 3) / 6))
    return win_rate_weight * win_rate + sharpe_weight * normalized_sharpe


# PARAMETER OPTIMIZATION
def create_objective_function(data_dict: Dict[str, pd.DataFrame],
                             param_space: Dict[str, Tuple],
                             strategy_class: type,
                             strategy_name: str,
                             trade_config: TradeConfig,
                             min_trades: int,
                             objective_weights: Tuple[float, float] = (0.35, 0.65)) -> Callable:
    
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
            
            if results['total_trades'] < min_trades:
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
                       objective_weights: Tuple[float, float] = (0.35, 0.65)) -> Tuple[Dict, optuna.Study]:
    
    objective_fn = create_objective_function(
        data_dict, param_space, strategy_class, strategy_name,
        trade_config, wf_config['min_trades'], objective_weights
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
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


# WINDOW PROCESSING
def process_window(window: Dict,
                  window_idx: int,
                  data_dict: Dict[str, pd.DataFrame],
                  param_space: Dict[str, Tuple],
                  strategy_class: type,
                  strategy_name: str,
                  trade_config: TradeConfig,
                  wf_config: Dict,
                  objective_weights: Tuple[float, float] = (0.35, 0.65),
                  verbose: bool = False) -> Optional[Dict]:
    
    if verbose:
        print(f"\nWindow {window_idx}")
        print(f"Training:  {window['train_start'].date()} to {window['train_end'].date()}")
        print(f"Testing:   {window['test_start'].date()} to {window['test_end'].date()}")
        print(f"Holdout:   {window['holdout_start'].date()} to {window['holdout_end'].date()}")
    
    train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
    
    best_params, study = optimize_parameters(
        train_data, param_space, strategy_class, strategy_name,
        trade_config, wf_config, window_idx, objective_weights
    )
    
    if best_params is None:
        return None
    
    if verbose:
        print(f"Best objective: {study.best_value:.4f}")
    
    test_data = filter_data_by_date(data_dict, window['test_start'], window['test_end'])
    test_results = run_backtest(test_data, best_params, strategy_class, strategy_name, trade_config)
    
    holdout_data = filter_data_by_date(data_dict, window['holdout_start'], window['holdout_end'])
    holdout_results = run_backtest(holdout_data, best_params, strategy_class, strategy_name, trade_config)
    
    result = {
        'window': window_idx,
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
    result.update({f'test_{k}': v for k, v in test_results.items()})
    result.update({f'holdout_{k}': v for k, v in holdout_results.items()})
    
    return result


# MAIN WALK-FORWARD ANALYSIS
def walk_forward_analysis(data_dict: Dict[str, pd.DataFrame],
                         param_space: Dict[str, Tuple],
                         strategy_class: type,
                         strategy_name: str,
                         trade_config: TradeConfig,
                         wf_config: Dict,
                         objective_weights: Tuple[float, float] = (0.35, 0.65),
                         verbose: bool = False) -> Tuple[pd.DataFrame, List[optuna.Study]]:
    
    start_date, end_date = get_date_range(data_dict)
    
    windows = generate_windows(
        start_date, end_date,
        wf_config['training_days'],
        wf_config['testing_days'],
        wf_config['holdout_days']
    )
    
    if verbose:
        print(f"Walk-Forward Analysis: {len(windows)} windows")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
    
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
            _, study = optimize_parameters(
                train_data, param_space, strategy_class, strategy_name,
                trade_config, wf_config, i, objective_weights
            )
            studies.append(study)
    
    results_df = pd.DataFrame(results)
    
    if verbose and not results_df.empty:
        print_summary(results_df)
    
    return results_df, studies


# PARAMETER SENSITIVITY
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
    
    if not sensitivity_results:
        return pd.DataFrame()
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    return sensitivity_df


def aggregate_param_importance(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    if sensitivity_df.empty:
        return pd.DataFrame()
    
    return sensitivity_df.groupby('parameter')['importance_score'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).sort_values('mean', ascending=False)


def get_param_distributions(studies: List[optuna.Study],
                           top_n: int = 5) -> Dict[str, Dict]:
    
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
                'values': values
            }
    
    return distributions


# RESULTS SUMMARY
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
    print("\n" + "="*80)
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print("="*80)
    
    test_stats = calculate_summary_stats(results_df, 'test')
    holdout_stats = calculate_summary_stats(results_df, 'holdout')
    
    print("\nOUT-OF-SAMPLE TESTING:")
    print(f"  Win Rate:     {test_stats['test_avg_win_rate']:.2%}")
    print(f"  Sharpe Ratio: {test_stats['test_avg_sharpe']:.4f}")
    print(f"  Avg Return:   {test_stats['test_avg_return']:.2f}%")
    print(f"  Avg Drawdown: {test_stats['test_avg_drawdown']:.2f}%")
    print(f"  Total Trades: {test_stats['test_total_trades']:.0f}")
    print(f"  Positive Windows: {test_stats['test_positive_windows']}/{len(results_df)} ({test_stats['test_positive_pct']:.1%})")
    
    print("\nHOLDOUT VALIDATION:")
    print(f"  Win Rate:     {holdout_stats['holdout_avg_win_rate']:.2%}")
    print(f"  Sharpe Ratio: {holdout_stats['holdout_avg_sharpe']:.4f}")
    print(f"  Avg Return:   {holdout_stats['holdout_avg_return']:.2f}%")
    print(f"  Avg Drawdown: {holdout_stats['holdout_avg_drawdown']:.2f}%")
    print(f"  Total Trades: {holdout_stats['holdout_total_trades']:.0f}")
    print(f"  Positive Windows: {holdout_stats['holdout_positive_windows']}/{len(results_df)} ({holdout_stats['holdout_positive_pct']:.1%})")


# EXPORT FUNCTIONS
def export_results(results_df: pd.DataFrame,
                  sensitivity_df: pd.DataFrame,
                  studies: List[optuna.Study],
                  output_dir: str = './walkforward_results'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(f"{output_dir}/walkforward_results.csv", index=False)
    
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(f"{output_dir}/parameter_sensitivity.csv", index=False)
        
        importance = aggregate_param_importance(sensitivity_df)
        importance.to_csv(f"{output_dir}/parameter_importance.csv")
    
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

