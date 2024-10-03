import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates


# Function to plot indicators and signals (without unused parameters)
def plot_indicators_and_signals(symbol, indicators_1m, indicators_5m, signals_1m, signals_5m):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 12), sharex=True)

    # Plot candlestick chart for 5-minute interval
    axs[0].set_title(f'{symbol} Candlestick Chart (5 minutes)')
    candlestick_ohlc(axs[0], indicators_5m['ohlc'], width=0.6, colorup='g', colordown='r')
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[0].legend(['5 min Candlestick'])

    # Plot Moving Averages (MA) on 5-minute interval
    axs[1].plot(indicators_5m['ma_5'], label='MA 5', color='blue')
    axs[1].plot(indicators_5m['ma_10'], label='MA 10', color='orange')
    axs[1].set_title('Moving Averages (5 minutes)')
    axs[1].legend()

    # Plot signals on 5-minute interval
    axs[2].set_title(f'Signals for {symbol} (5 minutes)')
    axs[2].scatter(signals_5m['BUY'].index, signals_5m['BUY']['close'],
                   label='BUY Signal', marker='^', color='green', alpha=0.7)
    axs[2].scatter(signals_5m['SELL'].index, signals_5m['SELL']['close'],
                   label='SELL Signal', marker='v', color='red', alpha=0.7)
    axs[2].legend()

    # Plot Moving Averages (MA) on 1-minute interval
    axs[3].plot(indicators_1m['ma_5'], label='MA 5', color='blue')
    axs[3].plot(indicators_1m['ma_10'], label='MA 10', color='orange')
    axs[3].set_title('Moving Averages (1 minute)')
    axs[3].legend()

    # Plot signals on 1-minute interval
    axs[3].set_title(f'Signals for {symbol} (1 minute)')
    axs[3].scatter(signals_1m['BUY'].index, signals_1m['BUY']['close'],
                   label='BUY Signal', marker='^', color='green', alpha=0.7)
    axs[3].scatter(signals_1m['SELL'].index, signals_1m['SELL']['close'],
                   label='SELL Signal', marker='v', color='red', alpha=0.7)
    axs[3].legend()

    plt.tight_layout