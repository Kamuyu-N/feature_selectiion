import pandas as pd
import talib as tb
import seaborn as sns
import matplotlib.pyplot as plt


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

    return pd.DataFrame(signal_data).set_index('DateTime')


def shift_column_up(column, shift_len):
    if not isinstance(column, pd.Series):
        raise ValueError("Input not pandas Series")

    shifted_column = column.copy()

    # Shift the values up by one position
    shifted_column[1:] = shifted_column[:-1]

    return shifted_column


# Example usage
data = {'A': [10, 20, 30, 40]}
df = pd.DataFrame(data)

# Apply the function to the column 'A'
df['A_shifted'] = shift_column_up(df['A'])

print(df)

# nan values will all be removed
def prev_data(indicator_data, indicator_type,TimeStamp):
    dt = TimeStamp.tolist()

    #Column names will be changed during output
    # must create a feasible schedule that can be followed( for sucess' pfailing to planis planning to falil)
    data ={
        'DateTime':[],
        'c1':[],
        'c2': [],
        'c3': [],
        'c4': [],
        'c5':[]
    }

    for index, (date, value) in enumerate(indicator_data.items()):
        data['DateTime'].append(dt.iloc[index])
        if index == 0:
            data['c1'].append(value)
            data['c2'].append(value)
            data['c3'].append(value)
            data['c4'].append(value)
            data['c5'].append(value)

        else:
            pass


# Data Preparation
pd.options.display.width = 0
df = pd.read_csv("C:/Users/25473/Downloads/EURUSD_4H.csv", sep='\t')
df.columns = ['DATE', 'TIME', 'Open', 'High', 'Low', 'Close', 'Volume', 'VOL', 'SPREAD']
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
df.drop(['DATE', 'TIME'], axis = 1, inplace=True)
order = ['DATETIME', 'Open', 'High', 'Low', 'Close','Volume']
price_action = (df[order])
price_action.set_index("DATETIME", inplace=True)

#Feature engineering
rsi_columns = prev_data(tb.RSI(price_action.Close, timeperiod = 14),'RSI', price_action.index)
kama = tb.KAMA(price_action.Close, timeperiod = 30)
adx = tb.ADX(price_action.High, price_action.Low, price_action.Close)
heikin_ashi = Heikin_Ashi_data(price_action.index, price_action.Open, price_action.High, price_action.Low, price_action.Close)



# for index,(date,value) in enumerate(rsi.items()):
#     print(rsi[index])





def indicator_addition(price):
    '''Columns of indicator data will be input contsining the last three values in each column for each indicator
    'for the price action ( columnss for the subtraction ( if a sucessful trade) remove the ones that were stoped out ( look at their low/high)
     we can also look for a way to intoroduce the use of the hieken data (0 if confusing,1 if bbull and -1 if bear
      We will play around with the timeperiod values'''



