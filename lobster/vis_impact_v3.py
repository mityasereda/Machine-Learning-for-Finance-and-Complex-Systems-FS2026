# --- 全流程代码：LOBSTER 主动买单/卖单市场冲击复现（修正版）---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: 读入数据
orderbook_path = 'AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
message_path = 'AMZN_2012-06-21_34200000_57600000_message_10.csv'

orderbook = pd.read_csv(orderbook_path, header=None)
messages = pd.read_csv(message_path, header=None)
messages.columns = ['time', 'type', 'order_id', 'size', 'price', 'direction']

# 筛选真实的交易事件
# type 4: 显性交易 (市价单或激进限价单导致的成交)
# type 5: 隐藏订单导致的交易
# 同时考虑交易方向和规模
min_trade_size = 100  # 最小交易规模
impact_events = messages[
    (messages['type'].isin([4, 5])) &  # 只选择真实交易
    (messages['size'] >= min_trade_size)  # 交易规模足够大
].copy()

# 分别筛选买单和卖单事件
# direction = -1 表示买单（吃掉ask），direction = 1 表示卖单（吃掉bid）
buy_events = impact_events[impact_events['direction'] == -1]  # 买单（吃掉ask）
sell_events = impact_events[impact_events['direction'] == 1]  # 卖单（吃掉bid）

# Step 2: 参数设定
num_impacts = 30  # 每个方向选择30个样本
pre_window = 500 / 1000     # 0.5 秒前
post_window = 5000 / 1000   # 5 秒后 (缩短观察窗口)
time_grid = np.arange(-int(pre_window*1000), int(post_window*1000) + 1)

# 从每个类别中随机选择样本
np.random.seed(42)  # 设置随机种子以保证结果可重复
buy_indices = np.random.choice(buy_events.index, size=min(num_impacts, len(buy_events)), replace=False)
sell_indices = np.random.choice(sell_events.index, size=min(num_impacts, len(sell_events)), replace=False)

# Step 3: 买单冲击模拟
def simulate_buy_side_impact(indices, messages, orderbook, time_grid):
    buy_paths = []
    for impact_index in indices:
        t0 = messages.loc[impact_index, 'time']
        order_row = orderbook.iloc[impact_index]

        # 读取卖单数据（Ask side）
        ask_prices = [order_row[i] / 10000 for i in [0, 4, 8]]  # Ask Price 1,2,3
        ask_sizes = [order_row[i] for i in [1, 5, 9]]  # Ask Size 1,2,3

        window_mask = (messages['time'] >= t0 - pre_window) & (messages['time'] <= t0 + post_window)
        window_indices = messages[window_mask].index
        if len(window_indices) < 20:
            continue

        mid_prices = []
        rel_times = []

        for idx in window_indices:
            row = orderbook.iloc[idx]
            ask = row[0] / 10000  # Ask Price 1
            bid = row[2] / 10000  # Bid Price 1
            mid = (ask + bid) / 2
            rel_time = int((messages.loc[idx, 'time'] - t0) * 1000)
            mid_prices.append(mid)
            rel_times.append(rel_time)

        # 使用事件前100ms的平均价格作为基准
        pre_event_mask = (messages['time'] >= t0 - 0.1) & (messages['time'] < t0)
        pre_event_prices = messages[pre_event_mask]['price'].values / 10000
        # baseline_price = np.mean(pre_event_prices) if len(pre_event_prices) > 0 else mid_prices[0]
        # Use the mid-quote series you already collected
        pre_idx = [i for i, t in enumerate(rel_times) if t < 0]
        baseline_price = np.mean([mid_prices[i] for i in pre_idx])  # always defined because we required ≥20 points

        # 买单冲击导致价格上涨（正向影响）
        relative_mid_prices = [(p - baseline_price) / baseline_price for p in mid_prices]

        price_series = pd.Series(data=relative_mid_prices, index=rel_times)
        price_series = price_series.groupby(level=0).mean()
        interpolated = price_series.reindex(time_grid).interpolate(limit_direction='both')
        buy_paths.append(interpolated.values)

    return np.vstack(buy_paths)

# Step 4: 卖单冲击模拟
def simulate_sell_side_impact(indices, messages, orderbook, time_grid):
    sell_paths = []
    for impact_index in indices:
        t0 = messages.loc[impact_index, 'time']
        order_row = orderbook.iloc[impact_index]

        # 读取买单数据（Bid side）
        bid_prices = [order_row[i] / 10000 for i in [2, 6, 10]]  # Bid Price 1,2,3
        bid_sizes = [order_row[i] for i in [3, 7, 11]]  # Bid Size 1,2,3

        window_mask = (messages['time'] >= t0 - pre_window) & (messages['time'] <= t0 + post_window)
        window_indices = messages[window_mask].index
        if len(window_indices) < 20:
            continue

        mid_prices = []
        rel_times = []

        for idx in window_indices:
            row = orderbook.iloc[idx]
            ask = row[0] / 10000  # Ask Price 1
            bid = row[2] / 10000  # Bid Price 1
            mid = (ask + bid) / 2
            rel_time = int((messages.loc[idx, 'time'] - t0) * 1000)
            mid_prices.append(mid)
            rel_times.append(rel_time)

        # 使用事件前100ms的平均价格作为基准
        pre_event_mask = (messages['time'] >= t0 - 0.1) & (messages['time'] < t0)
        pre_event_prices = messages[pre_event_mask]['price'].values / 10000
        # baseline_price = np.mean(pre_event_prices) if len(pre_event_prices) > 0 else mid_prices[0]# Use the mid-quote series you already collected
        pre_idx = [i for i, t in enumerate(rel_times) if t < 0]
        baseline_price = np.mean([mid_prices[i] for i in pre_idx])  # always defined because we required ≥20 points

        
        # 卖单冲击导致价格下跌（负向影响）
        relative_mid_prices = [(p - baseline_price) / baseline_price for p in mid_prices]  # 移除负号

        price_series = pd.Series(data=relative_mid_prices, index=rel_times)
        price_series = price_series.groupby(level=0).mean()
        interpolated = price_series.reindex(time_grid).interpolate(limit_direction='both')
        sell_paths.append(interpolated.values)

    return np.vstack(sell_paths)

# 运行模拟
buy_price_matrix = simulate_buy_side_impact(buy_indices, messages, orderbook, time_grid)
mean_buy_path = np.nanmean(buy_price_matrix, axis=0)
std_buy_path = np.nanstd(buy_price_matrix, axis=0)

sell_price_matrix = simulate_sell_side_impact(sell_indices, messages, orderbook, time_grid)
sell_mean_path = np.nanmean(sell_price_matrix, axis=0)
sell_std_path = np.nanstd(sell_price_matrix, axis=0)

# Step 5: 绘制 side-by-side 买单和卖单市场冲击图
plt.style.use('seaborn-v0_8-deep')  # 使用现代的seaborn风格
sns.set_context("paper", font_scale=1.5)  # 增加字体大小
sns.set_palette("husl")  # 使用现代配色方案

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# 买单
sns.lineplot(ax=axes[0], x=time_grid, y=mean_buy_path * 10000, 
            color='forestgreen', linewidth=2.5)
axes[0].fill_between(time_grid, 
                    (mean_buy_path - std_buy_path) * 10000, 
                    (mean_buy_path + std_buy_path) * 10000, 
                    color='forestgreen', alpha=0.2)
axes[0].axvline(x=0, linestyle='--', color='gray', alpha=0.5, linewidth=2)
axes[0].set_title('Market Impact of Large Buy Trades', fontsize=14, pad=15)
axes[0].set_xlabel('Relative Time (ms)', fontsize=12, labelpad=10)
axes[0].set_ylabel('Baseline-Relative Price', fontsize=12, labelpad=10)
axes[0].tick_params(axis='both', which='major', labelsize=11)
axes[0].grid(True, alpha=0.3)

# 卖单
sns.lineplot(ax=axes[1], x=time_grid, y=sell_mean_path * 10000, 
            color='crimson', linewidth=2.5)
axes[1].fill_between(time_grid, 
                    (sell_mean_path - sell_std_path) * 10000, 
                    (sell_mean_path + sell_std_path) * 10000, 
                    color='crimson', alpha=0.2)
axes[1].axvline(x=0, linestyle='--', color='gray', alpha=0.5, linewidth=2)
axes[1].set_title('Market Impact of Large Sell Trades', fontsize=14, pad=15)
axes[1].set_xlabel('Relative Time (ms)', fontsize=12, labelpad=10)
axes[1].tick_params(axis='both', which='major', labelsize=11)
axes[1].grid(True, alpha=0.3)

# 整体标题
# plt.suptitle('Average Market Impact: Buy vs. Sell (30 Samples Each)', 
#             fontsize=16, y=1.05)

# 调整布局
plt.tight_layout()

# 保存图片（高质量）
# plt.savefig('market_impact_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('market_impact_analysis.png', format='png', dpi=300, bbox_inches='tight')

plt.show()
