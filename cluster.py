import pandas as pd
import talib as tb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed



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
    col_2, col_3, col_4 = shift_column_up(indicator_data, 1, f'{indicator_type}_back1'), shift_column_up(indicator_data,
                                                                                                         2,
                                                                                                         f'{indicator_type}_back2'), shift_column_up(
        indicator_data, 3, f'{indicator_type}_back3')

    return pd.concat([col_2, col_3, col_4], axis=1)


def price_columns(price, take_profit, stop_loss, look_period):
    '''Row data in format of (DateTime,Open, High, Low, Close)'''
    result = {
        'DateTime': [],
        'TRD_Final': [],

    }
    for index, df_row_main in enumerate(price.iterrows()):
        df_row, dt_main_loop = list(df_row_main[1]), df_row_main[0]
        entry_price, temp_a = df_row[3], []

        foward_data = {
            'High': [],
            'Low': [],
        }

        for date, df_row_a in (price.iterrows()):
            val_return = {
                'DateTime': [],
                'Result': []
            }

            if date > dt_main_loop and len(foward_data['High']) < look_period:  # error encountered( first condition)

                foward_data['High'].append(df_row_a.iloc[1])
                foward_data['Low'].append(df_row_a.iloc[2])

                if len(foward_data['High']) == look_period:

                    for high, low in zip(foward_data["High"], foward_data['Low']):
                        if entry_price + take_profit <= high and entry_price - stop_loss <= low:
                            val_return['DateTime'].append(df_row_main[0])
                            val_return['Result'].append(1)

                        elif entry_price - take_profit <= low and entry_price + stop_loss > high:
                            val_return['DateTime'].append(df_row_main[0])
                            val_return['Result'].append(-1)

                        elif entry_price - stop_loss <= low or entry_price + stop_loss >= high:  # if any of the sl have been hit
                            val_return['DateTime'].append(df_row_main[0])
                            val_return['Result'].append(0)

                    result['DateTime'].append(val_return['DateTime'][0])
                    result['TRD_Final'].append(val_return['Result'][0])

    end = pd.DataFrame(result).set_index('DateTime')

    missing_data = (price_action.tail(look_period).copy()).drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

    missing_data['TRD_Final'] = np.nan

    return pd.concat([end, missing_data])


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
        new_roc[col_name] = data[column].pct_change(periods=period)  # print for rsi and ensure other data is correct

    return pd.DataFrame(new_roc, index=data.index)


def ma_categorical(ma, close_prices):
    ma_df, categories_df = pd.concat(ma, axis=1), {}

    for column in ma_df.columns.tolist():
        col_name = f'{column}_categorical'
        categories_df[col_name] = ma_df[column] > (close_prices)
    return pd.DataFrame(categories_df).astype(int)


# Data Preparation
pd.options.display.width = 0
df = pd.read_csv("C:/Users/samue/OneDrive/Documents/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv")
df.columns = ['DATETIME', 'Open', 'High', 'Low', 'Close', 'Volume']
print(df.columns)
# df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
# df.drop(['DATE', 'TIME'], axis=1, inplace=True)
order = ['DATETIME', 'Open', 'High', 'Low', 'Close', 'Volume']
price_action = (df[order])

price_action.set_index("DATETIME", inplace=True)
print(price_action)

# Feature engineering
# Price Transform indicators
avg_price = pd.DataFrame(tb.AVGPRICE(price_action.Open, price_action.High, price_action.Low, price_action.Close),
                         columns=['AVGPRICE'])
med_price = pd.DataFrame(tb.MEDPRICE(price_action.High, price_action.Low), columns=['MEDPRICE'])
typical_price = pd.DataFrame(tb.TYPPRICE(price_action.High, price_action.Low, price_action.Close), columns=['TYPPRICE'])
wcl_price = pd.DataFrame(tb.WCLPRICE(price_action.High, price_action.Low, price_action.Close), columns=['WCLPRICE'])


# function for momentum indicators
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
                                 columns=['CORREL_60   '])), axis=1)

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

Heikin_ashi = Heikin_Ashi_data(price_action.index, price_action.Open, price_action.High, price_action.Low,
                               price_action.Close)
trade_columns = price_columns(price_action, take_profit=0.0026, stop_loss=0.0013,
                              look_period=10)  # find the bset combination use a for loop ( all combinations )
ma_categories = ma_categorical([kama_periods, t3_periods, ema_periods, tema_periods], price_action.Close)

df = pd.concat(
    [price_action.Open, price_action.High, price_action.Low, price_action.Close, Heikin_ashi, cos, exp, sqrt, LOG10, ln,
     ht_trendmode, ht_dcphase, ht_dcperiod, linear_reg, linearReg_slope, correl, std_dev, adx_period, atr_periods,
     t3_periods, tema_periods,
     ema_periods, kama_periods, willr_periods, cmo_periods, wcl_price, plus_di_periods, minus_di_periods, mom_periods,
     rsi_periods, avg_price, typical_price, med_price, trade_columns, ma_categories], axis=1)

df.dropna(axis=0, inplace=True)
df['Signal_value'] = df['Signal_value'].astype(int)

ROC_col = rate_of_change_columns(df, 4)
df = pd.concat([df, ROC_col], axis=1)

df.dropna(axis=0, inplace=True)  # to remove the nan values from ROC

file_path = 'C:/Users/25473/Documents/DataFrame/price.csv'  # Replace with your desired file path
df.to_csv(file_path, index=False)
# use of atr for sl etc


# Addition of financial data and removal of entries based off of the news
# Then removal of the nan values ( keep training with the 4h then later do the 15 minute)
# find the bset combination use a for loop ( all combinations of sl, tp and look period)
# and use of the art method for sl and tp -- find the most profiatvble ( ask gpt if approach is good )


# considerations
# removal of big moves i.e like 30 pips in a candle ( depending on the timeframe )
# Tp abd Sl values are to be change considering the timeframe
# Another method would be not to used fixed tp and sl zones


#       Note
# after the mddel has done predictions check if it is profitable and drawdown has not taken place
# When done with fundumentals concact the data and store as csv( with all of the features of different time periods)--- should or not rsi and mom and stoch
