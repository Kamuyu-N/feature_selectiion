import pandas as pd
import talib as tb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    col_2,col_3,col_4 = shift_column_up(indicator_data,1,f'{indicator_type}_back1'),shift_column_up(indicator_data,2,f'{indicator_type}_back2'),shift_column_up(indicator_data,3,f'{indicator_type}_back3')

    return pd.concat([col_2,col_3,col_4],axis=1)

def price_columns(price,take_profit,stop_loss, look_period):
    '''Row data in format of (DateTime,Open, High, Low, Close)'''
    result = {
        'DateTime':[],
        'TRD_Final':[],

    }
    for index, df_row_main in enumerate(price.iterrows()):
        df_row,dt_main_loop = list(df_row_main[1]), df_row_main[0]
        entry_price, temp_a = df_row[3],[]

        foward_data = {
            'High': [],
            'Low': [],
        }

        for date, df_row_a in (price.iterrows()):
            val_return = {
                'DateTime': [],
                'Result': []
            }

            if date > dt_main_loop and len(foward_data['High']) < look_period:# error encountered( first condition)

                foward_data['High'].append(df_row_a.iloc[1])
                foward_data['Low'].append(df_row_a.iloc[2])

                if len(foward_data['High']) == look_period:

                    for high, low in zip(foward_data["High"],foward_data['Low']):
                        if entry_price + take_profit <= high and entry_price - stop_loss <= low:
                            val_return['DateTime'].append(df_row_main[0])
                            val_return['Result'].append(1)

                        elif entry_price - take_profit <= low and entry_price + stop_loss > high:
                            val_return['DateTime'].append(df_row_main[0])
                            val_return['Result'].append(-1)

                        elif entry_price - stop_loss <= low  or entry_price + stop_loss >= high: # if any of the sl have been hit
                            val_return['DateTime'].append(df_row_main[0])
                            val_return['Result'].append(0)

                    result['DateTime'].append(val_return['DateTime'][0])
                    result['TRD_Final'].append(val_return['Result'][0])

    return pd.DataFrame(result)

def rate_of_change(indicator_data):
    pass

# Data Preparation
pd.options.display.width = 0
df = pd.read_csv("C:/Users/25473/Downloads/EURUSD_4H.csv", sep='\t')
df.columns = ['DATE', 'TIME', 'Open', 'High', 'Low', 'Close', 'Volume', 'VOL', 'SPREAD']
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
df.drop(['DATE', 'TIME'], axis = 1, inplace=True)
order = ['DATETIME', 'Open', 'High', 'Low', 'Close','Volume']
price_actionn = (df[order])
price_action = price_actionn.tail(100)
price_action.set_index("DATETIME", inplace=True)

                                # Feature engineering
#rsi columns
'''Check the impoertance values of the current then compare them with the lagged (shifted) -- chose the one with a higher importance reading '''
rsi_periods  = pd.concat((pd.DataFrame(tb.RSI(price_action.Close, timeperiod = 7),columns=['Rsi_7']),pd.DataFrame(tb.RSI(price_action.Close, timeperiod = 21),
                columns=['Rsi_21']),pd.DataFrame(tb.RSI(price_action.Close, timeperiod = 14),columns=['Rsi_14'])),axis=1)

#Moving averages
kama_periods = pd.concat((pd.DataFrame(tb.KAMA(price_action.Close, timeperiod = 30), columns=['Kama_30']), pd.DataFrame(tb.KAMA(price_action.Close, timeperiod = 15), columns=['Kama_15']),
                pd.DataFrame(tb.KAMA(price_action.Close, timeperiod = 60), columns=['Kama_60'])),axis=1)

ema_periods = pd.concat((pd.DataFrame(tb.EMA(price_action.Close, timeperiod = 9), columns=['EMA_9']), pd.DataFrame(tb.EMA(price_action.Close, timeperiod = 21), columns=['EMA_21']),
                pd.DataFrame(tb.EMA(price_action.Close, timeperiod = 50), columns=['EMA_50']),pd.DataFrame(tb.EMA(price_action.Close, timeperiod = 100), columns=['EMA_100'])), axis=1)

tema_periods = pd.concat((pd.DataFrame(tb.TEMA(price_action.Close, timeperiod = 30), columns=['TEMA_30']), pd.DataFrame(tb.TEMA(price_action.Close, timeperiod = 15), columns=['TEMA_15']),
                pd.DataFrame(tb.TEMA(price_action.Close, timeperiod = 60), columns=['TEMA_60'])),axis=1)

t3_periods = pd.concat(((pd.DataFrame(tb.T3(price_action.Close, timeperiod=5),columns=['T3_5'])), (pd.DataFrame(tb.T3(price_action.Close, timeperiod=12),columns=['T3_12'])),
                (pd.DataFrame(tb.T3(price_action.Close, timeperiod=20),columns=['T3_20']))),axis=1)

#Other
atr_periods = pd.concat(((pd.DataFrame(tb.ATR(price_action.High,price_action.Low,price_action.Close, timeperiod = 14),columns=['ATR_14'])),(pd.DataFrame(tb.ATR(price_action.High,price_action.Low,price_action.Close, timeperiod = 21),columns=['ATR_21'])),
                (pd.DataFrame(tb.ATR(price_action.High,price_action.Low,price_action.Close, timeperiod = 28),columns=['ATR_28'])),(pd.DataFrame(tb.ATR(price_action.High,price_action.Low,price_action.Close, timeperiod = 7),columns=['ATR_7']))),axis = 1)

adx_period =  pd.concat(((pd.DataFrame(tb.ADX(price_action.High,price_action.Low,price_action.Close, timeperiod = 14),columns=['ADX_14'])),(pd.DataFrame(tb.ADX(price_action.High,price_action.Low,price_action.Close, timeperiod = 21),columns=['ADX_21'])),
                (pd.DataFrame(tb.ADX(price_action.High,price_action.Low,price_action.Close, timeperiod = 28),columns=['ADX_28'])),(pd.DataFrame(tb.ADX(price_action.High,price_action.Low,price_action.Close, timeperiod = 7),columns=['ADX_7']))),axis = 1)

#Price Transform indicators
avg_price = pd.DataFrame(tb.AVGPRICE(price_action.Open, price_action.High,price_action.Low, price_action.Close), columns=['AVGPRICE'])
med_price = pd.DataFrame(tb.MEDPRICE(price_action.High, price_action.Low),columns=['MEDPRICE'])
typical_price = pd.DataFrame(tb.TYPPRICE(price_action.High,price_action.Low, price_action.Close), columns=['TYPPRICE'])
wcl_price = pd.DataFrame(tb.WCLPRICE(price_action.High,price_action.Low, price_action.Close),columns=['WCLPRICE'])

#Statistical Methods



Heikin_ashi = Heikin_Ashi_data(price_action.index, price_action.Open, price_action.High, price_action.Low, price_action.Close)# should be shifted upwards once
trade_columns = price_columns(price_action,take_profit=0.0010,stop_loss=0.0010,look_period=3) # for the fixed prices

#rsi perv












print(rsi_columns)
# print(type(X))
# create all ( inclusive of the math functions ) and different period of the technical indicators --- then feature selection methods
# look for a way to see previous movement



# look at all of the coeficints from variance thresholds for the technical indicators
# fundumentals( if there is high data for the day or not )


        # considerations
# removal of big moves i.e like 30 pips in a candle ( depending on the timeframe )
# Tp abd Sl values are to be change considering the timeframe
# Another method would be not to used fixed tp and sl zones

#       Note
# carry out eda to find out how many trades are taken in total(Before and after )
# after the mddel has done predictions check if it is profitable and drawdown has not taken place
# remember for data to be separated into 3 sets
# use 4h data then scale upto 15 min data
# addition of economic indicators
# When done with fundumentals concact the data and store as csv( with all of the features of different time periods)--- should or not rsi and mom and stoch
# addition of the math ones( we will see if anything arises in the model )
# file kra returns and do the assignment
