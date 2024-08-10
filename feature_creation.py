import pandas as pd
import talib as tb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import datetime
import polars as pl


def Heikin_Ashi_data(index, open, high, low, close):
    data1 = {
        'DATETIME': index.tolist(),
        'Open': open.tolist(),
        'High': high.tolist(),
        'Low': low.tolist(),
        'Close': close.tolist()
    }

    data = pd.DataFrame(data1)
    h_data = {
        'DateTime': [],
        'Open': [],
        'High': [],
        'Low': [],
        'Close': []
    }
    signal_data = {
        'DateTime': [],
        'Signal_value': []
    }

    ha_close = [(o + h + l + c) / 4 for o, h, l, c in zip(data['Open'], data['High'], data['Low'], data['Close'])]
    ha_dt = (data['DATETIME']).tolist()

    for i in range(len(data['Open'])):
        ha_open = 0
        if i == 0:
            ha_open = data['Open'][i]

        else:
            ha_open = (data['Open'][i - 1] + data['Close'][i - 1]) / 2

        ha_high = max(data['High'][i], ha_open, ha_close[i])
        ha_low = min(data['Low'][i], ha_open, ha_close[i])

        h_data['DateTime'].append(ha_dt[i])
        h_data['Open'].append(ha_open)
        h_data['High'].append(ha_high)
        h_data['Low'].append(ha_low)
        h_data['Close'].append(ha_close[i])
        signal_data['DateTime'].append(ha_dt[i])

        # check for strong trend
        if ha_open > ha_close[i] and ha_high == ha_open:
            signal_data['Signal_value'].append(1)

        elif ha_open < ha_close[i] and ha_low == ha_open:
            signal_data['Signal_value'].append(-1)

        else:
            signal_data['Signal_value'].append(0)

    df = pd.DataFrame(signal_data)
    df.set_index('DateTime', inplace=True)

    return df.shift(periods=1)


def shift_column_up(column, shift_len, column_name):
    if not isinstance(column, pd.Series):
        raise ValueError("Input not pandas Series")
    df = (pd.DataFrame(column))
    df.columns = [column_name]

    return df.shift(periods=shift_len)


def column_expand(indicator_data, indicator_type):
    col_2, col_3, col_4 = shift_column_up(indicator_data, 1, f'{indicator_type}_back1'), shift_column_up(indicator_data,2,f'{indicator_type}_back2'), shift_column_up(
        indicator_data, 3, f'{indicator_type}_back3')

    return pd.concat([col_2, col_3, col_4], axis=1)

def price_columns(prices, tp, sl, look_forward):
    'removal of the trades that are almost the same? removal or stay -- check the results to determine ( on the validation set )'
    'i.e if trade a is taken at 3:15 and another is taken at 3:30 and both are successful, one should be removed'#after processing is done
    prices = price_action.copy()
    prices.reset_index(inplace=True)

    result ={
        'DateTime': [],
        'TRD_Final': []
    }

    def append_func(dt, trd_result):
        result['DateTime'].append(dt)
        result['TRD_Final'].append(trd_result)

    for index, (date, Open,high,low,close,vol) in enumerate(prices.itertuples(index=False)):
        entry_price = close

        max_high = max(prices[index:look_forward + index].High)
        min_low = min(prices[index:look_forward + index].Low)

        sl_buy, tp_buy =  entry_price - sl, entry_price + tp
        sl_sell, tp_sell = entry_price + sl, entry_price - tp

        if max_high > tp_buy and sl_buy < min_low:
            append_func(date,1)
        elif max_high < sl_sell and tp_sell > min_low:
            append_func(date, -1)
        else:
            append_func(date, 0)
    return pd.DataFrame(result)


def rate_of_change_columns(data_frame, period):
    data = data_frame.copy()
    new_roc = {}

    try:
        categorical_columns = data.select_dtypes(include=['int']).columns.tolist()
        numerical_columns = data.select_dtypes(include=['float']).columns.tolist()

    except Exception as e:
        print(f"{e} : Remove all nan values present in the DataFrame")

    for column in numerical_columns:
        col_name = f'{column}_ROC_{period}'
        if col_name == 'TRD_Final_ROC_4': continue
        new_roc[col_name] = data[column].pct_change(periods=period)

    return pd.DataFrame(new_roc, index=data.index)

def ma_categorical(ma, close_prices):
    ma_df, categories_df = pd.concat(ma, axis=1), {}
    for column in ma_df.columns.tolist():
        col_name = f'{column}_categorical'
        categories_df[col_name] = ma_df[column] > (close_prices)

    return pd.DataFrame(categories_df).astype(int)

# Data Preparation
pd.options.display.width = 0
df = pd.read_csv("C:/Users/25473/Downloads/eurusd-15m/eurusd-15m.csv", sep= ';')
df = df[:5000]

df.columns = ['DATE','TIME','Open', 'High', 'Low', 'Close', 'Volume']
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
# df.drop(['DATE', 'TIME'], axis=1, inplace=True)
order = ['DATETIME', 'Open', 'High', 'Low', 'Close', 'Volume']


#For 15M data
time_from = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
time_to = datetime.datetime.strptime('06:00:00', '%H:%M:%S').time()

df['TIME'] = [datetime.datetime.strptime(time, '%H:%M:%S').time() for time in df['TIME']]
df['DATETIME'] = [np.nan if x > time_from and x < time_to else x for x in df['TIME'] ]
df.dropna(inplace=True, axis=0)
price_action = (df[order])
df.drop(['DATE', 'TIME'], axis=1, inplace=True)
price_action.set_index("DATETIME", inplace=True)

# price_action = price_action[:1000]
# Price Transform indicators
avg_price = pd.DataFrame(tb.AVGPRICE(price_action.Open, price_action.High, price_action.Low, price_action.Close),
                         columns=['AVGPRICE'])
med_price = pd.DataFrame(tb.MEDPRICE(price_action.High, price_action.Low), columns=['MEDPRICE'])
typical_price = pd.DataFrame(tb.TYPPRICE(price_action.High, price_action.Low, price_action.Close), columns=['TYPPRICE'])
wcl_price = pd.DataFrame(tb.WCLPRICE(price_action.High, price_action.Low, price_action.Close), columns=['WCLPRICE'])


def chunk_split(df):
    no_of_chunks = round(len(df)/multiprocessing.cpu_count())
    temp,a = [],0

    for i in range(1,round(len(df)/no_of_chunks)+1):
        a += no_of_chunks
        if i == 1:
            temp.append(df[:a])
            continue
        elif i == no_of_chunks:
            temp.append(df[prev:])
            break

        prev = a - no_of_chunks
        temp.append(df[prev:a])

    return temp


def indicator_calc(periods, indicator_name, data_to_be_used):
    '''For the data_to_be_used parameter pass in a list of strings '''
    tb_indicator = getattr(tb,indicator_name)
    temp = {}
    if data_to_be_used.lower() == 'close':
        for period in periods:
            temp[f'{indicator_name}_{period}'] = (tb_indicator(price_action.Close, timeperiod=period))

    elif data_to_be_used.lower() == 'avgprice':
        for periods in periods:
            temp[f'{indicator_name}_{period}_avg_price'] = tb_indicator(price_action.Close, timeperiod=period)

    else:
        for period in periods:
            temp[f'{indicator_name}_{period}'] = (tb_indicator(price_action.High, price_action.Low,price_action.Close, timeperiod=period))

    return pd.DataFrame(temp)
parameters_list = [
    ([7, 14, 21, 28], 'RSI', 'close'),
    ([7, 14, 21, 28], 'MOM', 'close'),
    ([7, 14, 21, 28], 'PLUS_DI', 'HLC'),
    ([7, 14, 21, 28], 'MINUS_DI', 'HLC'),
    ([7, 14, 21, 28], 'CMO', 'close'),
    ([7, 14, 21, 28], 'WILLR', 'HLC'),
    ([15, 30, 45, 60], 'KAMA', 'close'),
    ([9, 21, 50, 100], 'EMA', 'close'),
    ([15, 30, 45, 60], 'TEMA', 'close'),
    ([5, 12, 20, 25], 'T3', 'close'),
    ([7, 14, 21, 28], 'ATR', 'HLC'),
    ([7, 14, 21, 28], 'ADX', 'HLC')
]


rsi_periods,mom_periods,plus_di_periods, minus_di_periods,cmo_periods,willr_periods,kama_periods,ema_periods,tema_periods,t3_periods,atr_periods,adx_period =(
        Parallel(n_jobs=-1, verbose=1)(delayed(indicator_calc)(period,indicator,data_use) for period,indicator,data_use in parameters_list )
)

# Statistical Methods
std_dev = pd.concat((pd.DataFrame(tb.STDDEV(price_action.Close, timeperiod=3), columns=['Std_dev_3']),
                     pd.DataFrame(tb.STDDEV(price_action.Close, timeperiod=5), columns=['Std_dev_5']),
                     pd.DataFrame(tb.STDDEV(price_action.Close, timeperiod=7), columns=['Std_dev_7']),
                     pd.DataFrame(tb.STDDEV(price_action.Close, timeperiod=10), columns=['Std_dev_10'])), axis=1)

correl = pd.concat((pd.DataFrame(tb.CORREL(price_action.High, price_action.Low, timeperiod=15), columns=['CORREL_15']),
                    pd.DataFrame(tb.CORREL(price_action.High, price_action.Low, timeperiod=30), columns=['CORREL_30']),
                    pd.DataFrame(tb.CORREL(price_action.High, price_action.Low, timeperiod=45), columns=['CORREL_45']),
                    pd.DataFrame(tb.CORREL(price_action.High, price_action.Low, timeperiod=60),
                                 columns=['CORREL_60'])), axis=1)

linear_reg = pd.concat((pd.DataFrame(tb.LINEARREG(price_action.Close, timeperiod=14), columns=['LinearReg_14']),
                        pd.DataFrame(tb.LINEARREG(price_action.Close, timeperiod=21), columns=['LinearReg_21']),
                        pd.DataFrame(tb.LINEARREG(price_action.Close, timeperiod=28), columns=['LinearReg_28']),
                        pd.DataFrame(tb.LINEARREG(price_action.Close, timeperiod=7), columns=['LinearReg_7'])), axis=1)

linearReg_slope = pd.concat(
    [pd.DataFrame(tb.LINEARREG_SLOPE(price_action.Close, timeperiod=14), columns=['LinearSlope_14']),
     pd.DataFrame(tb.LINEARREG_SLOPE(price_action.Close, timeperiod=7), columns=['LinearSlope_7']),
     pd.DataFrame(tb.LINEARREG_SLOPE(price_action.Close, timeperiod=21), columns=['LinearSlope_21']),
     pd.DataFrame(tb.LINEARREG_SLOPE(price_action.Close, timeperiod=28), columns=['LinearSlope_28'])], axis=1)

# Cycle Indicators
ht_dcperiod = pd.concat((pd.DataFrame(tb.HT_DCPERIOD(price_action.Close), columns=['HT_DCPERIOD']),
                         pd.DataFrame(tb.HT_DCPERIOD(avg_price.AVGPRICE), columns=['HT_DCPERIOD_avg_price'])
                         ), axis=1)

ht_dcphase = pd.concat((pd.DataFrame(tb.HT_DCPHASE(price_action.Close), columns=['HT_DCPHASE']),
                        pd.DataFrame(tb.HT_DCPHASE(avg_price.AVGPRICE), columns=['HT_DCPHASE_avg_price'])
                        ), axis=1)

ht_trendmode = pd.concat((pd.DataFrame(tb.HT_TRENDMODE(price_action.Close), columns=['HT_TRENDMODE']),
                          pd.DataFrame(tb.HT_TRENDMODE(avg_price.AVGPRICE), columns=['HT_TRENDMODE_avg_price'])
                          ), axis=1)

# Math transforms
ln = pd.concat((pd.DataFrame(tb.LN(price_action.Close), columns=['LN']),
                pd.DataFrame(tb.LN(avg_price.AVGPRICE), columns=['LN_avg_price'])
                ), axis=1)

LOG10 = pd.concat((pd.DataFrame(tb.LOG10(price_action.Close), columns=['LOG_10']),
                   pd.DataFrame(tb.LOG10(avg_price.AVGPRICE), columns=['LOG_10_avg_price'])
                   ), axis=1)

sqrt = pd.concat((pd.DataFrame(tb.SQRT(price_action.Close), columns=['SQRT']),
                  pd.DataFrame(tb.SQRT(avg_price.AVGPRICE), columns=['SQRT_avg_price'])
                  ), axis=1)

exp = pd.concat((pd.DataFrame(tb.EXP(price_action.Close), columns=['EXP']),
                 pd.DataFrame(tb.EXP(avg_price.AVGPRICE), columns=['EXP_avg_price'])
                 ), axis=1)

cos = pd.concat((pd.DataFrame(tb.COS(price_action.Close), columns=['COS']),
                 pd.DataFrame(tb.COS(avg_price.AVGPRICE), columns=['COS_avg_price'])
                 ), axis=1)

Heikin_ashi = Heikin_Ashi_data(price_action.index, price_action.Open, price_action.High, price_action.Low,price_action.Close)
ma_categories = ma_categorical([kama_periods, t3_periods, ema_periods, tema_periods], price_action.Close)
import time as tm

start = tm.perf_counter()
print(price_columns(price_action,0.0010,0.0010,10)['TRD_Final'].value_counts())
end = tm.perf_counter()
print(f'Time Taken: {end - start} seconds')

def file_creation(tp,sl,look_forward):
    trade_columns = price_columns(price_action,tp=tp,sl=sl,look_forward=look_forward)
    trade_columns.set_index(price_action.index, inplace=True)
    trade_columns.drop('DateTime', axis=1, inplace=True)

    df = pd.concat(
        [price_action.Open, price_action.High, price_action.Low, price_action.Close, Heikin_ashi, cos, exp, sqrt, LOG10, ln,
         ht_trendmode, ht_dcphase, ht_dcperiod, linear_reg, linearReg_slope, correl, std_dev, adx_period, atr_periods,
         t3_periods, tema_periods,ema_periods, kama_periods, willr_periods, cmo_periods, wcl_price, plus_di_periods, minus_di_periods, mom_periods,
         rsi_periods, avg_price, typical_price, med_price, ma_categories, trade_columns], axis=1)

    df.dropna(axis=0, inplace=True)
    df['Signal_value'] = df['Signal_value'].astype(int)

    ROC_col = rate_of_change_columns(df, 4)
    df = pd.concat([df, ROC_col], axis=1)

    df.dropna(axis=0, inplace=True)  # to remove the nan values from ROC

    #creating a new file every
    file_path = f'C:/Users/25473/Documents/DataFrame2/15M_tp_{tp}_sl_{sl}.csv'
    # with open(file_path, 'w+') as file:
    #     pass

    df.to_csv(file_path, index=False)

loop_params_m15 = [[0.0010,0.0005,15],[0.0020,0.0010,15],[0.0015,0.0010,15],[0.0014,0.0007,15],[0.0016,0.0008,15],[0.0012,0.0008,15],[0.0009,0.0006,15] ]
loop_params_h4 = [[0.0020,0.0010,10],[0.0020,0.0015,10],[0.0030,0.0010,10],[0.0030,0.0015,10]]

for tp,sl,lp in loop_params_m15:
    file_creation(tp,sl,lp)



#             After file creation ( try and see which one macimizes the scores)
# Try mixing the data ( for equal distribution ) -- after processing
# Try and remove if i.e 3 are following each other drop 2 ? try if better
# Find out what is your move ( incase all of this fails ) -- switch to dsa? internships? quant? learn what ?
#
# Addition of financial data and removal of entries based off of the news
# Then removal of the nan values ( keep training with the 4h then later do the 15 minute)
# find the bset combination use a for loop ( all combinations of sl, tp and look period)
# and use of the art method for sl and tp -- find the most profiatvble ( ask gpt if approach is good )
#
#             considerations
# removal of big moves i.e like 30 pips in a candle ( depending on the timeframe )
# Tp abd Sl values are to be change considering the timeframe
# Another method would be not to used fixed tp and sl zones
# reduce timeperiods (to 5 ish ) and addition of the continur=e keyword (reduce irrelevant iterations )
