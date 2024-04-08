import numpy as np


###################################################################################################
# 计算技术面指标
###################################################################################################

# 计算移动平均指数
def calculate_moving_average(data, window):
    ma = []
    for i in range(window - 1, len(data)):
        ma.append(sum(data[i - window + 1:i + 1]) / window)
    return ma

# 计算EMA
def calculate_ema(close_prices, window):
    ema = [close_prices[0]]
    multiplier = 2 / (window + 1)
    for i in range (1, len(close_prices)):
        ema.append((close_prices[i] - ema[-1]) * multiplier + ema[-1])
    return np.array(ema)

# 计算MACD
def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    # 计算短期和长期的指数移动平均
    short_ema = calculate_ema(close_prices, fast_period)
    long_ema = calculate_ema(close_prices, slow_period)
    # 计算 MACD 线
    macd_line = short_ema - long_ema
    # 计算信号线
    signal_line = calculate_ema(macd_line, signal_period)
    # 计算 MACD 柱状线
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

# 计算Elder force index
def calculate_elder_force_index(close_prices, volumes, window):
    # 计算价格变化
    price_change = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices))]
    # 计算 Force Index
    force_index = [price_change[i] * volumes[i] for i in range(len(price_change))]
    # 对 Force Index 进行移动平均（可选）
    force_index_smoothed = calculate_moving_average(force_index, window)
    return force_index_smoothed

# 计算ATR（平均真实范围）
def calculate_atr(higher_prices, lower_prices, close_prices, n):
    # 计算TR
    hl = higher_prices - lower_prices
    hc = np.abs(higher_prices - np.roll(close_prices, 1))
    lc = np.abs(lower_prices - np.roll(close_prices, 1))
    tr = np.max(np.column_stack((hl, hc, lc)), axis=1)
    # 计算ATR
    atr = np.convolve(tr, np.ones(n)/n, mode='valid')
    return np.mean(atr[-5:])

# 计算RSI (相对强弱指标)
def calculate_rsi(prices, period):
    changes = np.diff(prices)
    gains = changes[changes > 0]
    losses = -changes[changes < 0]

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 计算SAR指标
def calculate_sar(high_prices, low_prices, acceleration_factor=0.02, max_acceleration_factor=0.2):
    sar = np.full_like(high_prices, np.nan)
    trend = np.full_like(high_prices, None)
    af = acceleration_factor
    ep = high_prices[0]

    for i in range(2, len(high_prices)):
        if trend[i-1] == "up":
            if low_prices[i] < sar[i-1]:
                trend[i] = "down"
                sar[i] = ep
                ep = high_prices[i]
                af = acceleration_factor
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = min(sar[i], low_prices[i-1], low_prices[i-2])
                if high_prices[i] > ep:
                    ep = high_prices[i]
                    af = min(af + acceleration_factor, max_acceleration_factor)
        else:
            if high_prices[i] > sar[i-1]:
                trend[i] = "up"
                sar[i] = ep
                ep = low_prices[i]
                af = acceleration_factor
            else:
                sar[i] = sar[i-1] - af * (sar[i-1] - ep)
                sar[i] = max(sar[i], high_prices[i-1], high_prices[i-2])
                if low_prices[i] < ep:
                    ep = low_prices[i]
                    af = min(af + acceleration_factor, max_acceleration_factor)

    return sar, trend

# 计算唐奇安通道
def calculate_donchian_channel(high_prices, low_prices, window):
    high_channel = np.max(high_prices[-window:])
    low_channel = np.min(low_prices[-window:])
    return high_channel, low_channel