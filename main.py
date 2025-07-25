# Python code to automate a stock trading strategy
# Based on an idea from _Stocks on the Move_ by Andreas F. Clenow
#
# Looks for stocks that have a nearly constant rate of growth.
#
###
# David Guilbeau
# Version 0.0.5

import csv
import datetime
from datetime import timedelta
import operator
import sys
import pandas as pd
import yfinance as yf
## ? import yfinance-cache as yfc
import robin_stocks.robinhood as rs
import sqlite3
import math
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import os

database_filename = r'.\stock_data.sqlite3'
symbols_filename = r'.\constituents.csv'
pickle_filename = r'.\stock_group_df_0.0.1.pkl'
download = True
# download = False

# backtest = True
backtest = False
# maximum_trading_days_needed = 600
maximum_trading_days_needed = 100
window_step_size = 5

# this would vary with the indicator calculation
window_trading_days_needed = 90

maximum_calendar_days_needed = maximum_trading_days_needed * 365.25 / 253
# start = finish - timedelta(days=130)
# 253 trading days in a year
# 365.25 days in a year
# 90 trading days to look at
# How many calendar days?
# 90 * 365.25 / 253 = 130 calendar days

# Set requested date range
finish_date = datetime.date.today()
# finish_date = datetime.datetime(2021, 7, 6)
start_date = finish_date - timedelta(days=maximum_calendar_days_needed)
print("Requested start:", start_date, "finish:", finish_date)

extra_days = 5  # extra days to look at in case the start date is not a trading day


# From Google AI Overview for "python library average true range"
def calculate_atr(df, window=14):

    df['H-L'] = df['high'] - df['low']
    df['H-C'] = abs(df['high'] - df['close'].shift(1))
    df['L-C'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window).mean()
    
    return df['ATR']



def create_database_if_needed():
    global con

    cur = con.cursor()

    # If table does not exist, create it
    sql = '''
    CREATE TABLE  IF NOT EXISTS stock_data
    (date timestamp NOT NULL,
    ticker text NOT NULL,
    open real,
    high real,
    low real,
    close real,
    volume real,
    primary key(date, ticker)
    )
    '''
    cur.execute(sql)


def find_download_start_date(requested_start_date):
    global con
    # print("In find_download_start_date:", requested_start_date, type(requested_start_date))
    cur = con.cursor()

    # Find the last date in the database:
    sql = '''
    Select date From stock_data
    Order By date Desc
    Limit 1
    '''
    cur.execute(sql)
    rows = cur.fetchall()

    # if no date
    if len(rows) < 1:
        print('No rows found in database table.')
        download_start_date = requested_start_date
    else:
        print('Last date found in database:', rows[0][0])
        # Download the day after the one in the database
        download_start_date = rows[0][0].date() + timedelta(days=1)

    return download_start_date


def download_stock_data(download_start_date, download_finish_date):
    global con
    global stock_list

    if download:
        data = yf.download(stock_list,
                           start=(download_start_date - timedelta(days=extra_days)),
                           end=(download_finish_date + timedelta(days=1)),
                           group_by='ticker')

        data.to_pickle(pickle_filename)
    else:
        pickle_file = open(pickle_filename, 'rb')
        data = pickle.load(pickle_file)

    # https://stackoverflow.com/questions/63107594/how-to-deal-with-multi-level-column-names-downloaded-with-yfinance/63107801#63107801
    t_df = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    t_df = t_df.reset_index()

    cur = con.cursor()

    # This would insert dataframe data into database, but it fails if a date and ticker already exist
    # t_df.to_sql('stock_data', con, if_exists='append', index=False)

    print('Inserting data into database...')
    for i in range(len(t_df)):
        sql = 'insert into stock_data (date, ticker, close, high, low, open, volume) ' \
              'values (?,?,?,?,?,?,?)'
        try:
            cur.execute(sql, (t_df.iloc[i].get('Date').to_pydatetime(),
                              t_df.iloc[i].get('Ticker'),
                              t_df.iloc[i].get('Adj Close'),
                              t_df.iloc[i].get('High'),
                              t_df.iloc[i].get('Low'),
                              t_df.iloc[i].get('Open'),
                              t_df.iloc[i].get('Volume')))
        except sqlite3.IntegrityError:
            # print("\r", "Failed inserting:", str(t_df.iloc[i][0]), t_df.iloc[i][1], end='')
            print("Failed inserting:", str(t_df.iloc[i][0]), t_df.iloc[i][1])

    con.commit()
    print("told database to commit")
    # print("\r                                                    ")
#


def download_stock_data_robinhood(download_start_date, download_finish_date):
    global con
    global stock_list

    if download:
        # use Robinhood to get stock data
        # https://robin-stocks.readthedocs.io/en/latest/index.html
        rs.login()
        stock_price_list = rs.stocks.get_stock_historicals(
            stock_list[:10], interval='day', span='year', bounds='regular', info=None)
        data = pd.DataFrame(stock_price_list)

        print(data)

        data = data.drop(columns=['session', 'interpolated'])

        print(data)

        data = data.astype({'open_price': float,
                            'close_price': float,
                            'high_price': float,
                            'low_price': float,
                            'volume': int,
                            'symbol': str
                            })

        print('types changing')
        print(data)

        data = data.rename(columns={'begins_at': 'Date',
                                    'open_price': 'Open',
                                    'close_price': 'Close',
                                    'high_price': 'High',
                                    'low_price': 'Low',
                                    'symbol': 'Ticker',
                                    'volume': 'Volume'})

        data['Date'] = pd.to_datetime(data['Date'])

        print('columns renaming')
        print(data)

        # rs.logout()

        data.to_pickle(pickle_filename)
    else:
        pickle_file = open(pickle_filename, 'rb')
        data = pickle.load(pickle_file)

    # https://stackoverflow.com/questions/63107594/how-to-deal-with-multi-level-column-names-downloaded-with-yfinance/63107801#63107801
    t_df = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    t_df = t_df.reset_index()

    print('t_df')
    print(t_df)

    cur = con.cursor()

    # This would insert dataframe data into database, but it fails if a date and ticker already exist
    #t_df.to_sql('stock_data', con, if_exists='append', index=False)

    print('Inserting data into database...')
    for i in range(len(t_df)):

        sql = 'insert into stock_data (date, ticker, close, high, low, open, volume) ' \
              'values (?,?,?,?,?,?,?)'
        try:
            cur.execute(sql, (t_df.iloc[i].get('Date'),
                              t_df.iloc[i].get('Ticker'),
                              t_df.iloc[i].get('Close'),
                              t_df.iloc[i].get('High'),
                              t_df.iloc[i].get('Low'),
                              t_df.iloc[i].get('Open'),
                              t_df.iloc[i].get('Volume')))
        except sqlite3.IntegrityError:
            print("\r", "Failed inserting:", str(t_df.iloc[i][0]), t_df.iloc[i][1], end='')

    con.commit()
    print("\r                                                    ")
#


# Could return a list of 10 stocks with the number of dollars to invest in each
def find_list(stock_group_df):

    # Calculate indicators
    account_value = 200
    slope = {}
    r_sq = {}  # R squared
    annualized_return = {}
    adjusted_slope = {}

    x = {}
    y = {}
    plotly_x = {}
    predicted_y = {}
    jumped = {}
    above_predicted = {}
    below_25_percent_growth = {}
    atr_20 = {}
    shares_to_own = {}
    last_price = {}

    # print(stock_group_df)

    window_start_date = stock_group_df.iloc[0].name[1]
    window_finish_date = stock_group_df.iloc[-1].name[1]
    print("Window range:", window_start_date, window_finish_date)

    tickers = stock_group_df.index.unique(level=0)

    print('Calculating indicators')
    count = 0
    for stock in tickers:
        print("\r", stock, end='')

        # Get last 90 trading days of this stock
        # maybe make a copy, in case adding the 'ln' column changes the original stock_group_df
        stock_df = stock_group_df.loc[stock].iloc[-90:]

        last_price[stock] = stock_df['close'][-1]

        # Calculate adjusted slope
        # Regression of the natural logarithm of price
        stock_df.insert(0, 'ln', 0)
        stock_df['ln'] = np.log(stock_df['close'])
        # Pandas.datetime to numpy.datetime64 to numpy.datetime64 days to a floating point number
        x[stock] = stock_df.index.values.astype("datetime64[D]").astype("float").reshape(-1, 1)
        plotly_x[stock] = stock_df.index.values.astype("datetime64[D]")
        y[stock] = stock_df.get('ln').values

        model = LinearRegression().fit(x[stock], y[stock])
        predicted_y[stock] = model.predict(x[stock])
        slope[stock] = model.coef_[0]
        r_sq[stock] = model.score(x[stock], y[stock])
        annualized_return[stock] = pow(math.exp(slope[stock]), 250) - 1.0
        adjusted_slope[stock] = r_sq[stock] * annualized_return[stock]

        if y[stock][-1] < predicted_y[stock][-1]:
            # print('< predicted price')
            above_predicted[stock] = False
        else:
            # print('> predicted price')
            above_predicted[stock] = True

        # count = count + 1
        # if count > 5:
        #     break


        # Find if the stock moved +-15% in across 2 trading days
        jumped[stock] = False
        first_time = True
        previous_price = 0
        for price in stock_df.get('close').values:
            if first_time:
                previous_price = price
                first_time = False
                continue
            else:
                difference = previous_price - price
                percent = (difference / previous_price) * 100
                if abs(percent) >= 15:
                    jumped[stock] = True

                previous_price = price


        # Is the annualized growth rate below 25%?
        below_25_percent_growth[stock] = False
        if annualized_return[stock] * 100 < 25:
            below_25_percent_growth[stock] = True


        # Calculate Average True Range (20 days)
        stock_df = stock_group_df.loc[stock].iloc[-21:]

        atr = calculate_atr(stock_df['high'], stock_df['low'], stock_df['close'], length=20)
        if atr is None:
            print("* " + stock + " is None")
            atr_20[stock] = None
            shares_to_own[stock] = 0
        else:
            atr_20[stock] = atr[-1]
            shares_to_own[stock] = (account_value * 0.1) / atr_20[stock]


    print("\rR squared, price, shares, annualized rate of return :")

    # output = sorted(adjusted_slope.items(), key=operator.itemgetter(1), reverse=True)
    output = sorted(r_sq.items(), key=operator.itemgetter(1), reverse=True)

    count = 1
    for t in output:

        if count > 15:
            break

        stock = t[0]
        explanation_string = ''

        if below_25_percent_growth[stock]:
            ranking_string = 'X'
            explanation_string = explanation_string + 'Below 25% annual growth rate'
            continue
        elif jumped[stock]:
            ranking_string = 'X'
            explanation_string = 'Jumped >15% '
        elif above_predicted[stock]:
            ranking_string = 'X'
            explanation_string = explanation_string + 'Above predicted price'
            count = count + 1
        else:
            ranking_string = str(count)
            explanation_string = ''
            count = count + 1

        print(ranking_string, stock,
              round(t[1], 2),
              round(last_price[stock], 2) ,
              round(shares_to_own[stock], 1),
              str(round(annualized_return[stock] * 100)) + '%',
              explanation_string,
              )
        line1 = px.line(x=plotly_x[stock], y=y[stock], title=stock)
        line2 = px.line(x=plotly_x[stock], y=predicted_y[stock], title=stock)
        figure = go.Figure(data=line1.data + line2.data)
        figure.update_layout(title=stock)

        #figure.show()

        #if len(explanation_string) == 0:
        #    figure.show()
        #    continue
#


# Yahoo Finance has been unreliable with inserting a few days at a time,
# so as a workaround, delete the database
print("Deleting database file.")
try:
    os.remove(database_filename)
except:
    print("Had trouble deleting.")

stock_list = []

# con = sqlite3.connect(':memory:')
# detect_types is for timestamp support
con = sqlite3.connect(database_filename,
                      detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)


def main():
    global stock_list
    global con
    global start_date
    global finish_date

    csvfile = open(symbols_filename, "r", newline='')
    reader = csv.DictReader(csvfile)

    for row in reader:
        #Yahoo Finance uses a dash in stock symbols like BRK-B instead of a period like BRK.B
        stock_list.append(row["Symbol"].replace(".", "-"))
        # Debug
        # print(row["Symbol"], end= ' ')
        
    cur = con.cursor()

    create_database_if_needed()

    # print("in main:", start_date, type(start_date))
    download_start_date = find_download_start_date(start_date)

    download_finish_date = finish_date

    if download_start_date <= download_finish_date:
        download_stock_data(download_start_date, download_finish_date)
        # download_stock_data_robinhood(download_start_date, download_finish_date)
    else:
        print("Not downloading.")

    # Load requested date range from the database
    sql = '''
    Select * From stock_data
    Where Date >= ? and Date <= ?
    '''
    cur.execute(sql, [start_date - timedelta(days=extra_days),
                      finish_date + timedelta(days=1)])

    print('Database request start date, finish date:',
          start_date - timedelta(days=extra_days),
          finish_date + timedelta(days=1))

    stock_group_df = pd.DataFrame(cur.fetchall(),
                                  columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])

    con.close()

    # store the list of stocks
    stocks = stock_group_df["ticker"].unique().tolist()
    print('Length of a stocks:', len(stocks))

    # stock_group_df = stock_group_df.set_index(['ticker', 'date']).sort_index()
    stock_group_df = stock_group_df.set_index(['date', 'ticker']).sort_index()

    # print('stock_group_df:', stock_group_df)
    valid_stock_symbol = stocks[0]
    # Skipping for now
    # below is broken since the indexes were switched
    # print('Length of a stock in stock_group_df:', len(stock_group_df.loc[valid_stock_symbol]))

    # Need to drop any extra rows

    # Find actual start date
    # The second index "name" is the date
    database_start_date = stock_group_df.iloc[0].name[0]

    # Find actual finish date
    database_finish_date = stock_group_df.iloc[-1].name[0]
    print("Dataframe range:", database_start_date, database_finish_date)

    if backtest:
        # back test: step through time, sending a window of data to find_list
        # level 0 of the index has the dates
        trading_dates = stock_group_df.index.unique(level=0)
        starts = range(0, maximum_trading_days_needed - window_trading_days_needed, window_step_size)
        for s in starts:
            start_index = s
            finish_index = start_index + window_trading_days_needed
            window_df = stock_group_df.loc[ trading_dates[ start_index:finish_index ] ]
            window_df = window_df.reset_index()
            window_df = window_df.set_index(['ticker', 'date']).sort_index()

            find_list(window_df)
            print("")
    else:
        # current indicators
        trading_dates = stock_group_df.index.unique(level=0)
        window_df = stock_group_df.loc[trading_dates[-window_trading_days_needed:]]
        window_df = window_df.reset_index()
        window_df = window_df.set_index(['ticker', 'date']).sort_index()
        find_list(window_df)


if __name__ == '__main__':
    main()
