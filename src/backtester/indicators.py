"""Technical indicator functions"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List


def sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return prices.rolling(window=period, min_periods=period).mean()


def ema(prices: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return prices.ewm(span=period, min_periods=period).mean()


def wma(prices: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return prices.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()



def bollinger_bands(price: pd.Series, 
                   period: int = 20, 
                   std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands 
    
    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    that are standard deviations away from the middle band.
    
    Args:
        price: Price series
        period: SMA period (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        
    Returns:
        Dictionary with 'upper', 'middle', 'lower', 'width', 'width_pct'
        
    References:
        Bollinger, J. (2001). Bollinger on Bollinger Bands
    """
    middle = price.rolling(window=period, min_periods=period).mean()
    std = price.rolling(window=period, min_periods=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    # Calculate band width
    width = upper - lower
    
    # Avoid division by zero for bandwidth percentage
    middle_safe = middle.replace(0, np.nan)
    width_pct = width / middle_safe * 100
    
    return {
        'BB_UPPER': upper.rename('BB_UPPER'),
        'BB_MIDDLE': middle.rename('BB_MIDDLE'),
        'BB_LOWER': lower.rename('BB_LOWER'),
        'BB_WIDTH': width.rename('BB_WIDTH'),
        'BB_WIDTH_PCT': width_pct.rename('BB_WIDTH_PCT')
    }


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    Returns RSI values between 0-100.
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values

def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
        signal: int = 9) -> Dict[str, pd.Series]:
    """
    MACD indicator .
    
    Returns dict with 'macd', 'signal', 'histogram' keys.
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Momentum indicator ."""
    return prices - prices.shift(period)


def roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change ."""
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100

def stddev(prices: pd.Series, period: int) -> pd.Series:
    """Standard Deviation ."""
    return prices.rolling(window=period, min_periods=period).std()

def mabw(prices: pd.Series, fast_period: int = 10, slow_period: int = 50,
        multiplier: float = 1.0) -> Dict[str, pd.Series]:
    """
    Moving Average Band Width indicator .
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        multiplier: Band width multiplier
        
    Returns:
        Dict with 'upper', 'middle', 'lower', 'width', 'llv' keys
    """
    ma_slow = ema(prices, slow_period)
    ma_fast = ema(prices, fast_period)
    
    dst = ma_slow - ma_fast
    dv = dst.pow(2).rolling(fast_period, min_periods=fast_period).mean().apply(
        lambda x: x ** 0.5 if not pd.isna(x) else np.nan
    )
    dev = dv * multiplier
    
    upper = ma_slow + dev
    lower = ma_slow - dev
    width = (upper - lower) / ma_slow * 100
    
    # Lowest Low Value of width over slow_period
    llv = width.rolling(slow_period, min_periods=slow_period).min()
    
    return {
        'upper': upper,
        'middle': ma_slow,
        'lower': lower,
        'width': width,
        'llv': llv
    }


def typical_price(df: pd.DataFrame) -> pd.Series:
    """
    Typical Price (HLC/3) .
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
    """
    return (df['High'] + df['Low'] + df['Close']) / 3


def vpn(df: pd.DataFrame, period: int = 30, smooth: int = 3, 
        ma_period: int = 30) -> Dict[str, pd.Series]:
    """
    Volume Pressure Number
    
    Args:
        df: DataFrame with OHLCV data
        period: Calculation period
        smooth: EMA smoothing period
        ma_period: Moving average period for signal line
        
    Returns:
        Dict with 'vpn', 'vpn_smoothed', 'vpn_ma' keys.
    """
    # Calculate typical price and changes
    tp = typical_price(df)
    tp_prev = tp.shift(1)
    
    # Calculate ATR threshold
    atr_val = atr(df['High'], df['Low'], df['Close'], period)
    threshold = 0.1 * atr_val
    
    # Volume calculations
    volume = df['Volume']
    mav = volume.rolling(window=period, min_periods=period).mean()
    
    # Price change conditions
    price_change = tp - tp_prev
    positive_cond = price_change > threshold
    negative_cond = price_change < -threshold
    
    # Volume pressure sums
    vp = np.where(positive_cond, volume, 0)
    vn = np.where(negative_cond, volume, 0)
    
    vp_sum = pd.Series(vp, index=df.index).rolling(window=period, min_periods=period).sum()
    vn_sum = pd.Series(vn, index=df.index).rolling(window=period, min_periods=period).sum()
    
    # Calculate VPN
    mav_safe = mav.replace(0, 1)
    vpn_raw = (vp_sum - vn_sum) / mav_safe / period * 100
    
    # Smooth and create signal line
    vpn_ema = vpn_raw.ewm(span=smooth, min_periods=smooth).mean()
    vpn_ma = vpn_ema.rolling(window=ma_period, min_periods=ma_period).mean()
    
    return {
        'vpn': vpn_raw,
        'vpn_smoothed': vpn_ema,
        'vpn_ma': vpn_ma
    }


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume .
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
    """
    direction = np.where(df['Close'] > df['Close'].shift(1), 1,
                        np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    return (direction * df['Volume']).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price .
    
    Args:
        df: DataFrame with OHLCV data
    """
    tp = typical_price(df)
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()



def adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    """
    Average Directional Index .
    
    Returns dict with 'adx', 'plus_di', 'minus_di' keys.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate directional movements
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # Calculate ATR and smooth DM
    atr_val = atr(high, low, close, period)
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()
    
    # Calculate DI
    plus_di = 100 * (plus_dm_smooth / atr_val)
    minus_di = 100 * (minus_dm_smooth / atr_val)
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()
    
    return {
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di
    }


def pivot_points(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Standard Pivot Points .
    
    Returns dict with 'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3' keys.
    """
    high = df['High'].shift(1)
    low = df['Low'].shift(1)
    close = df['Close'].shift(1)
    
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_indicators_to_df(df: pd.DataFrame, indicators_config: Dict) -> pd.DataFrame:
    """
    Add multiple indicators to DataFrame .
    
    Args:
        df: OHLCV DataFrame
        indicators_config: Dict of {indicator_name: params}
        
    Returns:
        New DataFrame with indicators added
    """
    result = df.copy()
    
    for indicator_name, params in indicators_config.items():
        if indicator_name == 'sma':
            for period in params.get('periods', []):
                result[f'SMA_{period}'] = sma(df['Close'], period)
        
        elif indicator_name == 'ema':
            for period in params.get('periods', []):
                result[f'EMA_{period}'] = ema(df['Close'], period)
        
        elif indicator_name == 'atr':
            period = params.get('period', 14)
            result['ATR'] = atr(df['High'], df['Low'], df['Close'], period)
        
        elif indicator_name == 'rsi':
            period = params.get('period', 14)
            result['RSI'] = rsi(df['Close'], period)
        
        elif indicator_name == 'bollinger':
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            bb = bollinger_bands(df['Close'], period, std_dev)
            result['BB_UPPER'] = bb['upper']
            result['BB_MIDDLE'] = bb['middle']
            result['BB_LOWER'] = bb['lower']
        
        elif indicator_name == 'mabw':
            fast = params.get('fast_period', 10)
            slow = params.get('slow_period', 50)
            mult = params.get('multiplier', 1.0)
            mabw_result = mabw(df['Close'], fast, slow, mult)
            result['MAB_UPPER'] = mabw_result['upper']
            result['MAB_MIDDLE'] = mabw_result['middle']
            result['MAB_LOWER'] = mabw_result['lower']
            result['MAB_WIDTH'] = mabw_result['width']
            result['MAB_LLV'] = mabw_result['llv']
        
        elif indicator_name == 'macd':
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            macd_result = macd(df['Close'], fast, slow, signal)
            result['MACD'] = macd_result['macd']
            result['MACD_SIGNAL'] = macd_result['signal']
            result['MACD_HIST'] = macd_result['histogram']
    
    return result


def ama(ohlc: pd.DataFrame, 
        period: int = 10, 
        fast_period: int = 2, 
        slow_period: int = 30) -> pd.Series:
    """
    Calculate Adaptive Moving Average (AMA) - Perry Kaufman's algorithm.
    
    The AMA adjusts its smoothing constant based on market efficiency,
    using the ratio of price change to price volatility.
    
    Args:
        ohlc: DataFrame with OHLC data (must have 'High', 'Low', 'Close')
        period: Period for highest high/lowest low calculation
        fast_period: Fast smoothing constant period
        slow_period: Slow smoothing constant period
        
    Returns:
        AMA series
        
    References:
        Kaufman, P. J. (2013). Trading Systems and Methods (5th ed.)
    """
    # Create a copy and standardize column names
    data = ohlc.copy()
    data.columns = [c.lower() for c in data.columns]
    
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate highest high and lowest low over period
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate efficiency ratio component
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.nan)  # Avoid division by zero
    
    numerator = abs((close - lowest_low) - (highest_high - close))
    efficiency_ratio = numerator / price_range
    
    # Calculate smoothing constant
    ssc = efficiency_ratio * (fast_sc - slow_sc) + slow_sc
    cst = ssc ** 2
    
    # Initialize AMA array
    ama_values = np.zeros(len(data))
    
    # Calculate AMA iteratively
    for i in range(len(data)):
        if i == 0:
            ama_values[i] = close.iloc[i]
        elif i < period or pd.isna(cst.iloc[i]):
            # Use simple average for early periods
            ama_values[i] = close.iloc[i]
        else:
            ama_values[i] = ama_values[i-1] + cst.iloc[i] * (close.iloc[i] - ama_values[i-1])
    
    return pd.Series(ama_values, index=data.index, name='AMA')


def efficiency_ratio(price: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Efficiency Ratio (ER).
    
    ER measures the efficiency of price movement:
    - ER = 1: Perfect trending market
    - ER = 0: Pure random walk
    
    Formula: ER = |Price Change| / Sum(|Daily Price Changes|)
    
    Args:
        price: Price series
        period: Lookback period
        
    Returns:
        Efficiency Ratio series
    """
    change = price.diff(period).abs()
    volatility = price.diff().abs().rolling(window=period).sum()
    
    # Avoid division by zero
    volatility = volatility.replace(0, np.nan)
    er = change / volatility
    er = er.replace([np.inf, -np.inf], np.nan)
    
    return er.rename(f'ER{period}')


def kama(price: pd.Series, 
         er_period: int = 10, 
         ema_fast: int = 2, 
         ema_slow: int = 30, 
         period: int = 20) -> pd.Series:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA is similar to AMA but uses a different efficiency ratio calculation
    and smoothing methodology. It adapts to market conditions:
    - Fast EMA in trending markets
    - Slow EMA in ranging markets
    
    Args:
        price: Price series
        er_period: Period for efficiency ratio calculation
        ema_fast: Fast EMA period for trending markets
        ema_slow: Slow EMA period for ranging markets
        period: Initial SMA period
        
    Returns:
        KAMA series
        
    References:
        Kaufman, P. J. (1995). "Smarter Trading"
    """
    # Calculate efficiency ratio
    er = efficiency_ratio(price, period=er_period)
    
    # Calculate alpha values
    fast_alpha = 2.0 / (ema_fast + 1)
    slow_alpha = 2.0 / (ema_slow + 1)
    
    # Calculate smoothing constant
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    
    # Initialize with SMA
    sma_values = price.rolling(window=period).mean()
    kama_values = sma_values.copy()
    
    # Calculate KAMA recursively
    for i in range(period, len(price)):
        if pd.notna(kama_values.iloc[i-1]) and pd.notna(sc.iloc[i]):
            kama_values.iloc[i] = (
                kama_values.iloc[i-1] + 
                sc.iloc[i] * (price.iloc[i] - kama_values.iloc[i-1])
            )
    
    return kama_values.rename('KAMA')
# ===================================================================================


# UTILITY FUNCTIONS

def rolling_window(series: pd.Series, window: int) -> List[pd.Series]:
    """
    Create rolling windows .
    
    Returns list of Series, each of length 'window'.
    """
    return [series.iloc[i:i+window] for i in range(len(series) - window + 1)]


def normalize(series: pd.Series, method: str = 'zscore') -> pd.Series:
    """
    Normalize series .
    
    Methods: 'zscore', 'minmax', 'percentrank'
    """
    if method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'percentrank':
        return series.rank(pct=True)
    return series


###################################################################################################

# ============================================================================
# CANDLESTICK PATTERNS
# ============================================================================

def bullish_engulfing(open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    """
    Detect Bullish Engulfing candlestick pattern
    
    Pattern Requirements:
    1. Previous candle is bearish (close < open)
    2. Current candle is bullish (close > open)
    3. Current candle body completely engulfs previous candle body
    4. Current open < previous close
    5. Current close > previous open
    
    Args:
        open_price: Open price series
        close_price: Close price series
        
    Returns:
        Boolean series indicating Bullish Engulfing pattern
        
    References:
        Nison, S. (1991). Japanese Candlestick Charting Techniques
    """
    # Current candle is bullish (close > open)
    current_bullish = close_price > open_price
    
    # Previous candle is bearish (close < open)
    prev_bearish = close_price.shift(1) < open_price.shift(1)
    
    # Current close is greater than previous open
    close_above_prev_open = close_price > open_price.shift(1)
    
    # Current open is less than previous close
    open_below_prev_close = open_price < close_price.shift(1)
    
    # Current candle body engulfs previous candle body
    current_body = abs(close_price - open_price)
    prev_body = abs(close_price.shift(1) - open_price.shift(1))
    body_engulfing = current_body > prev_body
    
    # Combine all conditions
    bull_engulfing = (
        current_bullish & 
        prev_bearish & 
        close_above_prev_open & 
        open_below_prev_close & 
        body_engulfing
    )
    
    return bull_engulfing.rename('BULL_ENGULF')


def bearish_engulfing(open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    """
    Detect Bearish Engulfing candlestick pattern 
    
    Pattern Requirements:
    1. Previous candle is bullish (close > open)
    2. Current candle is bearish (close < open)
    3. Current candle body completely engulfs previous candle body
    
    Args:
        open_price: Open price series
        close_price: Close price series
        
    Returns:
        Boolean series indicating Bearish Engulfing pattern
    """
    # Current candle is bearish (close < open)
    current_bearish = close_price < open_price
    
    # Previous candle is bullish (close > open)
    prev_bullish = close_price.shift(1) > open_price.shift(1)
    
    # Current close is less than previous open
    close_below_prev_open = close_price < open_price.shift(1)
    
    # Current open is greater than previous close
    open_above_prev_close = open_price > close_price.shift(1)
    
    # Current candle body engulfs previous candle body
    current_body = abs(close_price - open_price)
    prev_body = abs(close_price.shift(1) - open_price.shift(1))
    body_engulfing = current_body > prev_body
    
    # Combine all conditions
    bear_engulfing = (
        current_bearish & 
        prev_bullish & 
        close_below_prev_open & 
        open_above_prev_close & 
        body_engulfing
    )
    
    return bear_engulfing.rename('BEAR_ENGULF')


def hammer(open_price: pd.Series, 
           high_price: pd.Series,
           low_price: pd.Series, 
           close_price: pd.Series) -> pd.Series:
    """
    Detect Hammer candlestick pattern
    
    Pattern Requirements:
    1. Small real body at the upper end of range
    2. Long lower shadow (at least 2x body size)
    3. Little to no upper shadow
    
    Args:
        open_price: Open price series
        high_price: High price series
        low_price: Low price series
        close_price: Close price series
        
    Returns:
        Boolean series indicating Hammer pattern
    """
    body_size = abs(close_price - open_price)
    upper_shadow = high_price - pd.concat([open_price, close_price], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_price, close_price], axis=1).min(axis=1) - low_price
    
    # Long lower shadow (at least 2x body)
    long_lower_shadow = lower_shadow >= (2 * body_size)
    
    # Small upper shadow
    small_upper_shadow = upper_shadow <= (0.3 * body_size)
    
    # Body in upper part of range
    total_range = high_price - low_price
    body_in_upper = (pd.concat([open_price, close_price], axis=1).min(axis=1) - low_price) >= (0.6 * total_range)
    
    return (long_lower_shadow & small_upper_shadow & body_in_upper).rename('HAMMER')


# ============================================================================
# BOLLINGER BANDS SIGNALS
# ============================================================================

def bbands_squeeze(bb_width: pd.Series, period: int = 125) -> pd.Series:
    """
    Detect Bollinger Bands squeeze
    
    Squeeze occurs when band width reaches its lowest level over lookback period.
    
    Args:
        bb_width: Bollinger Bands width series
        period: Lookback period for minimum width
        
    Returns:
        Boolean series indicating squeeze condition
    """
    lowest_width = bb_width.rolling(window=period, min_periods=period).min()
    squeeze = bb_width <= lowest_width
    
    return squeeze.rename('BB_SQUEEZE')


def bbands_bounce(close: pd.Series, 
                 low: pd.Series,
                 bb_lower: pd.Series) -> pd.Series:
    """
    Detect Bollinger Bands bounce signal 
    
    Bounce occurs when:
    1. Previous close was below lower band
    2. Current close is above lower band
    3. Current low touched or penetrated lower band
    
    Args:
        close: Close price series
        low: Low price series
        bb_lower: Bollinger Bands lower band
        
    Returns:
        Boolean series indicating bounce signal
    """
    # Previous close below lower band
    prev_close_below = close.shift(1) < bb_lower.shift(1)
    
    # Current close above lower band
    current_close_above = close > bb_lower
    
    # Current low touched or penetrated lower band
    current_low_below = low <= bb_lower
    
    bounce = prev_close_below & current_close_above & current_low_below
    
    return bounce.rename('BBANDS_BOUNCE')


def bbands_breakout(high: pd.Series, bb_upper: pd.Series) -> pd.Series:
    """
    Detect Bollinger Bands upper breakout
    
    Breakout occurs when price crosses above upper band.
    
    Args:
        high: High price series
        bb_upper: Bollinger Bands upper band
        
    Returns:
        Boolean series indicating breakout signal
    """
    # Previous high below upper band
    prev_below = high.shift(1) < bb_upper.shift(1)
    
    # Current high above upper band
    current_above = high > bb_upper
    
    breakout = prev_below & current_above
    
    return breakout.rename('BBANDS_BREAKOUT')


def stochastic_macd(ohlc: pd.DataFrame,
                   period: int = 45,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9,
                   adjust: bool = True) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic MACD Oscillator
    
    The Stochastic MACD combines the stochastic oscillator concept with MACD,
    normalizing the MACD values to the recent price range for better overbought/
    oversold signals.
    
    Algorithm:
        1. Find highest high and lowest low over period
        2. Calculate fast and slow EMAs
        3. Normalize EMAs to stochastic values:
           FastStoch = (FastEMA - LowLow) / (HighHigh - LowLow)
           SlowStoch = (SlowEMA - LowLow) / (HighHigh - LowLow)
        4. STMACD = (FastStoch - SlowStoch) * 100
        5. Signal = EMA(STMACD, signal_period)
    
    Args:
        ohlc: DataFrame with OHLC data (must have 'High', 'Low', 'Close')
        period: Period for highest high/lowest low calculation (default: 45)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        adjust: Use adjusted EMA calculation (default: True)
        
    Returns:
        Dictionary with 'stmacd', 'signal', 'histogram'
        
    References:
        Apirine, V. (2019). "Stochastic MACD Oscillator"
        Technical Analysis of Stocks & Commodities, November 2019
        https://traders.com/Documentation/FEEDbk_docs/2019/11/TradersTips.html
    """
    # Create a copy and standardize column names
    data = ohlc.copy()
    data.columns = [c.lower() for c in data.columns]
    
    # Calculate highest high and lowest low over period
    highest_high = data['high'].rolling(window=period, min_periods=period).max()
    lowest_low = data['low'].rolling(window=period, min_periods=period).min()
    
    # Calculate fast and slow EMAs
    ema_fast = data['close'].ewm(span=fast_period, adjust=adjust, min_periods=fast_period).mean()
    ema_slow = data['close'].ewm(span=slow_period, adjust=adjust, min_periods=slow_period).mean()
    
    # Calculate price range (avoid division by zero)
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.nan)
    
    # Calculate stochastic values for fast and slow EMAs
    stoch_fast = (ema_fast - lowest_low) / price_range
    stoch_slow = (ema_slow - lowest_low) / price_range
    
    # Calculate STMACD (scale by 100)
    stmacd_line = (stoch_fast - stoch_slow) * 100
    
    # Calculate signal line (EMA of STMACD)
    signal_line = stmacd_line.ewm(span=signal_period, adjust=adjust, min_periods=signal_period).mean()
    
    # Calculate histogram
    histogram = stmacd_line - signal_line
    
    return {
        'stmacd': stmacd_line.rename('STMACD'),
        'signal': signal_line.rename('STMACD_SIGNAL'),
        'histogram': histogram.rename('STMACD_HISTOGRAM')
    }


def rsema(price: pd.Series,
         ema_period: int = 50,
         rs_period: int = 50,
         multiplier: float = 10.0,
         adjust: bool = True) -> pd.Series:
    """
    Calculate Relative Strength EMA (RS-EMA)
    
    RS-EMA is a volatility-adjusted exponential moving average that uses
    relative strength to dynamically adjust the smoothing rate. It becomes
    more responsive during trending periods and smoother during consolidation.
    
    Algorithm:
        1. Calculate price changes (up and down moves)
        2. Smooth gains and losses with EMA
        3. Calculate RS = |EMA(gains) - EMA(losses)| / (EMA(gains) + EMA(losses))
        4. Adjust smoothing rate: Rate = (2/(period+1)) * (1 + RS*multiplier)
        5. Apply adaptive EMA with adjusted rate
    
    Args:
        price: Price series
        ema_period: Base EMA period for smoothing rate (default: 50)
        rs_period: Period for relative strength calculation (default: 50)
        multiplier: RS multiplier for rate adjustment (default: 10.0)
        adjust: Use adjusted EMA calculation (default: True)
        
    Returns:
        RS-EMA series
        
    References:
        "Relative Strength EMA" by John Ehlers
        Technical Analysis of Stocks & Commodities, May 2022
        https://traders.com/Documentation/FEEDbk_docs/2022/05/TradersTips.html
    """
    # Base smoothing multiplier
    mltp1 = 2.0 / (ema_period + 1.0)
    
    # Calculate price changes
    delta = price.diff()
    delta.iloc[0] = 0  # Set first value to 0
    
    # Separate up and down moves
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = down.abs()
    
    # Calculate EMA of gains and losses
    gain_ema = up.ewm(span=rs_period, adjust=adjust, min_periods=rs_period).mean()
    loss_ema = down.ewm(span=rs_period, adjust=adjust, min_periods=rs_period).mean()
    
    # Calculate relative strength
    # RS = |gain - loss| / (gain + loss + epsilon)
    rs = (gain_ema - loss_ema).abs() / (gain_ema + loss_ema + 0.00001)
    rs = rs * multiplier
    
    # Calculate adaptive rate
    rate = mltp1 * (1.0 + rs)
    
    # Calculate RS-EMA iteratively
    rsema_values = np.zeros(len(price))
    rsema_values[0] = price.iloc[0]
    
    for i in range(1, len(price)):
        # RSEMA[i] = rate[i] * price[i] + (1 - rate[i]) * RSEMA[i-1]
        rsema_values[i] = rate.iloc[i] * price.iloc[i] + (1.0 - rate.iloc[i]) * rsema_values[i-1]
    
    return pd.Series(rsema_values, index=price.index, name='RSEMA')
