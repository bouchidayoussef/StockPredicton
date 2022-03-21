import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
import datetime
import math

fig = plt.figure()
fig.patch.set_facecolor('#121416')
gs = fig.add_gridspec(6, 6)
ax1 = fig.add_subplot(gs[0:4, 0:4])
ax2 = fig.add_subplot(gs[0, 4:6])
ax3 = fig.add_subplot(gs[1, 4:6])
ax4 = fig.add_subplot(gs[2, 4:6])
ax5 = fig.add_subplot(gs[3, 4:6])
ax6 = fig.add_subplot(gs[4, 4:6])
ax7 = fig.add_subplot(gs[5, 4:6])
ax8 = fig.add_subplot(gs[4, 0:4])
ax9 = fig.add_subplot(gs[5, 0:4])

Stock = ['UBER', 'PYPL', 'AAPL', 'AMZN', 'MSFT', 'NFLX', 'GOOG']


def figure_design(ax):
    ax.set_facecolor('#091217')
    ax.tick_params(axis='both', labelsize=14, colors='white')
    ax.ticklabel_format(useOffset=False)
    ax.spines['bottom'].set_color('#808080')
    ax.spines['top'].set_color('#808080')
    ax.spines['left'].set_color('#808080')
    ax.spines['right'].set_color('#808080')


def subplot_plot(ax, stock_code, data, latest_price, latest_change, pattern, target):
    ax.clear()
    # print(len(data['close']))
    ax.plot(data['close'], color='white', linewidth=2)

    ymin = data['close'].min()
    ymax = data['close'].max()
    ystd = data['close'].std()

    if not math.isnan(ymax) and ymax != 0:
        ax.set_ylim([ymin - ystd * 0.5, ymax + ystd * 3])

    ax.text(0.02, 0.95, stock_code, transform=ax.transAxes, color='#FFBF00', fontsize=11,
            fontweight='bold', horizontalalignment='left', verticalalignment='top')

    ax.text(0.2, 0.95, latest_price, transform=ax.transAxes, color='white', fontsize=11,
            fontweight='bold', horizontalalignment='left', verticalalignment='top')

    if latest_change[0] == '+':
        colorcode = '#18b800'
    else:
        colorcode = '#ff3503'

    ax.text(0.4, 0.95, latest_change, transform=ax.transAxes, color=colorcode, fontsize=11,
            fontweight='bold', horizontalalignment='left', verticalalignment='top')

    if pattern == "Bullish":
        colorcode = '#18b800'
    elif pattern == "Bearish":
        colorcode = '#ff3503'
    else:
        colorcode = 'white'

    ax.text(0.98, 0.95, pattern, transform=ax.transAxes, color=colorcode, fontsize=11,
            fontweight='bold', horizontalalignment='right', verticalalignment='top')

    ax.text(0.98, 0.75, target, transform=ax.transAxes, color='#08a0e9', fontsize=11,
            fontweight='bold', horizontalalignment='right', verticalalignment='top')

    figure_design(ax)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def string_to_number(d2f, column):
    if isinstance(d2f.iloc[0, d2f.columns.get_loc(column)], str):
        d2f[column] = d2f[column].str.replace(',', '').astype(float)
    return d2f


def computeRSI(data, time_window):
    diff = data.diff(1).dropna()

    up_chg = 0 * diff
    down_chg = 0 * diff

    up_chg[diff > 0] = diff[diff > 0]

    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)

    return rsi


def read_data_ohlc(filename, stock_code, usecols):
    d2f = pd.read_csv(filename, header=None, usecols=usecols,
                      names=['time', stock_code, 'change', 'volume', 'pattern', 'target'], index_col='time',
                      parse_dates=['time'])

    # index_with_nan = df.index[df.isnull().any(axis=1)]
    # df.drop(index_with_nan, 0, inplace=True)

    d2f.index = pd.DatetimeIndex(d2f.index)
    # print(d2f.dtypes)
    d2f = string_to_number(d2f, stock_code)
    d2f = string_to_number(d2f, 'volume')
    d2f = string_to_number(d2f, 'target')

    # print(d2f.tail())

    latest_info = d2f.iloc[-1, :]
    latest_price = str(latest_info.iloc[0])
    latest_change = str(latest_info.iloc[1])

    df_vol = d2f['volume'].resample('1min').mean()
    data = d2f[stock_code].resample('1min').ohlc()
    data['time'] = data.index
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    data['MA5'] = data['close'].rolling(5).mean()
    data['MA10'] = data['close'].rolling(10).mean()
    data['MA20'] = data['close'].rolling(20).mean()
    data['RSI'] = computeRSI(data['close'], 14)

    data['volume_diff'] = df_vol.diff()
    data[data['volume_diff'] < 0] = None

    index_with_nan = data.index[data.isnull().any(axis=1)]
    data.drop(index_with_nan, 0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    # print(data)
    return data, latest_price, latest_change, d2f['pattern'][-1], d2f['target'][-1], d2f['volume'][-1]


def animate(i):
    # time_stamp = datetime.datetime.now() - datetime.timedelta(hours=6)
    # time_stamp = time_stamp.strftime('%Y-%m-%d %H:%M:%S')
    # filename = str(time_stamp) + 'stock data.csv'
    filename = '2022-03-09 stock data.csv'
    data, latest_price, latest_change, pattern, target, volume = \
        read_data_ohlc(filename, Stock[0], [1, 2, 3, 4, 5, 6])

    # print(data)

    candle_counter = range(len(data['open']) - 1)
    ohlc = []
    for candle in candle_counter:
        append_me = candle_counter[candle], data['open'][candle], \
                    data['high'][candle], data['low'][candle], \
                    data['close'][candle]
        ohlc.append(append_me)

    ax1.clear()
    candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#18b800', colordown='#ff3503')

    ax1.plot(data['MA5'], color='pink', linestyle='-', linewidth='1', label='5 minutes SMA')
    ax1.plot(data['MA10'], color='orange', linestyle='-', linewidth='1', label='10 minutes SMA')
    ax1.plot(data['MA20'], color='#08a0e9', linestyle='-', linewidth='1', label='20 minutes SMA')

    leg = ax1.legend(loc='upper left', facecolor='#121416', fontsize=10)
    for text in leg.get_texts():
        plt.setp(text, color='w')

    figure_design(ax1)

    ax1.text(0.005, 1.05, Stock[0], transform=ax1.transAxes, color='black', fontsize=18,
             fontweight='bold', horizontalalignment='left', verticalalignment='center',
             bbox=dict(facecolor='#FFBF00'))

    ax1.text(0.2, 1.05, latest_price, transform=ax1.transAxes, color='white', fontsize=18,
             fontweight='bold', horizontalalignment='center', verticalalignment='center')

    if latest_change[0] == '+':
        colorcode = '#18b800'
    else:
        colorcode = '#ff3503'

    ax1.text(0.4, 1.05, latest_change, transform=ax1.transAxes, color=colorcode, fontsize=18,
             fontweight='bold', horizontalalignment='center', verticalalignment='center')

    ax1.text(0.6, 1.05, target, transform=ax1.transAxes, color='#08a0e9', fontsize=18,
             fontweight='bold', horizontalalignment='center', verticalalignment='center')

    time_stamp = datetime.datetime.now()
    time_stamp = time_stamp.strftime('%Y-%m-%d %H:%M:%S')

    ax1.text(1.4, 1.05, time_stamp, transform=ax1.transAxes, color='white', fontsize=12,
             fontweight='bold', horizontalalignment='center', verticalalignment='center')

    ax1.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    ax1.set_xticklabels([])

    data_ax2, latest_price, latest_change, pattern, target, volume = read_data_ohlc(filename, Stock[1], [1, 7, 8, 9, 10, 11])
    subplot_plot(ax2, Stock[1], data_ax2, latest_price, latest_change, pattern, target)

    data_ax3, latest_price, latest_change, pattern, target, volume = read_data_ohlc(filename, Stock[2], [1, 12, 13, 14, 15, 16])
    subplot_plot(ax3, Stock[2], data_ax3, latest_price, latest_change, pattern, target)

    data_ax4, latest_price, latest_change, pattern, target, volume = read_data_ohlc(filename, Stock[3], [1, 17, 18, 19, 20, 21])
    subplot_plot(ax4, Stock[3], data_ax4, latest_price, latest_change, pattern, target)

    data_ax5, latest_price, latest_change, pattern, target, volume = read_data_ohlc(filename, Stock[4], [1, 22, 23, 24, 25, 26])
    subplot_plot(ax5, Stock[4], data_ax5, latest_price, latest_change, pattern, target)

    data_ax6, latest_price, latest_change, pattern, target, volume = read_data_ohlc(filename, Stock[5], [1, 27, 28, 29, 30, 31])
    subplot_plot(ax6, Stock[5], data_ax6, latest_price, latest_change, pattern, target)

    data_ax7, latest_price, latest_change, pattern, target, volume = read_data_ohlc(filename, Stock[6], [1, 32, 33, 34, 35, 36])
    subplot_plot(ax7, Stock[6], data_ax7, latest_price, latest_change, pattern, target)

    ax8.clear()
    figure_design(ax8)
    ax8.axes.yaxis.set_visible(False)

    pos = data['open'] - data['close'] < 0
    neg = data['open'] - data['close'] > 0
    data['x_axis'] = list(range(1, len(data['volume_diff']) + 1))
    ax8.bar(data['x_axis'][pos], data['volume_diff'][pos], color='#18b800', width=0.8, align='center')
    ax8.bar(data['x_axis'][neg], data['volume_diff'][neg], color='#ff3503', width=0.8, align='center')

    ymax = data['volume_diff'].max()
    ystd = data['volume_diff'].std()

    if not math.isnan(ymax):
        ax8.set_ylim([0, ymax + ystd * 3])

    ax8.text(0.01, 0.95, 'Volume:' + "{:,}".format(int(volume)), transform=ax8.transAxes, color='white',
             fontsize=10, fontweight='bold', horizontalalignment='left', verticalalignment='top')

    ax8.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    ax8.set_xticklabels([])

    ax9.clear()
    figure_design(ax9)
    ax9.axes.yaxis.set_visible(False)
    ax9.set_ylim([-5, 105])

    ax9.axhline(30, linestyle='-', color='green', linewidth=0.5)
    ax9.axhline(50, linestyle='-', color='white', linewidth=0.5)
    ax9.axhline(70, linestyle='-', color='red', linewidth=0.5)
    ax9.plot(data['x_axis'], data['RSI'], color='#08a0e9', linewidth=1.5)

    # 'RSI(14): ' + str(round(data['RSI'].iloc[-1], 2))
    ax9.text(0.01, 0.95, 'RSI(14): ', transform=ax9.transAxes, color='white',
             fontsize=10, fontweight='bold', horizontalalignment='left', verticalalignment='top')

    xdate = [i for i in data['time']]

    def mydate(x, pos=None):
        try:
            t = xdate[int(x)].strftime('%H:%M')
            return xdate[int(x)].strftime('%H:%M')
        except IndexError:
            return ''

    ax9.xaxis.set_major_formatter(mticker.FuncFormatter(mydate))
    ax8.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    ax8.tick_params(axis='x', which='major', labelsize=10)


ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()
