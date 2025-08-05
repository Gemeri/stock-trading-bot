import pandas as pd
from data import BacktestItem
import matplotlib.pyplot as plt
from dataclasses import asdict
from datetime import datetime


def plot_and_save(stock_ticker:str, backtest_list:list[BacktestItem], threshold:int)->str:

    # Step 3: Convert list of dataclasses to list of dicts
    backtest_dicts = [asdict(item) for item in backtest_list]

    # Step 4: Create DataFrame
    df = pd.DataFrame(backtest_dicts)

    # ⏱ Prepare timestamp index
    timestamps = df['timestamp'].to_numpy()
    portfolio_values = df['portfolio_value'].to_numpy()
    actions = df['action'].to_numpy()
    stock_prices = df['stock_price'].to_numpy()
    position_values = df['position'].to_numpy()
    balances = df['cash'].to_numpy()

    # 📈 Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # dates formatting
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()

    # 📉 Plot Net Worth (left Y-axis)
    ax1.plot(timestamps, portfolio_values, color='blue', label="Net Worth ($)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net Worth ($)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # 📊 Plot action bars (on same axis)
    bar_colors = ['green' if a > 0 else 'red' if a < 0 else 'gray' for a in actions]
    ax1.bar(timestamps, actions, color=bar_colors, alpha=0.5, label='Trades')

    # 📈 Plot Stock Price (right Y-axis)
    ax2 = ax1.twinx()
    ax2.plot(timestamps, stock_prices, color='orange', label="Stock Price ($)")
    ax2.set_ylabel("Stock Price ($)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # 💰 Plot Cash (third Y-axis)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset third axis
    ax3.plot(timestamps, balances, color='red', label='Cash ($)', linestyle=':')
    ax3.set_ylabel("Cash ($)", color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    # 🏷️ Format X-axis as dates
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()

    # 🧾 Title and combined legend
    plt.title(f"XGboost prediction {stock_ticker}: Net Worth, Stock Price, Balance & Trade Actions")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left")

    fig.tight_layout()    

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig(f"pred/backtest-graph-{stock_ticker}-th{threshold}-{timestamp}.png")

    # Save as CSV
    df.to_csv(f"pred/backtest-data-{stock_ticker}-th{threshold}-{timestamp}.csv", index=True)