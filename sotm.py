# Python code to automate a stock trading strategy
# Based on an idea from _Stocks on the Move_ by Andreas F. Clenow
#
###
# David Guilbeau
# Version 0.0.1

import csv
import datetime
from datetime import timedelta
import operator
import sys
import pandas as pd
import yfinance as yf
import sqlite3
import math
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas_ta as ta
# import traceback
import plotly.express as px
import plotly.graph_objects as go

database_filename = r'.\stock_data.sqlite3'
symbols_filename = r'.\sp500symbols.csv'
pickle_filename = r'.\stock_group_df_0.0.1.pkl'
download = False
maximum_trading_days_needed = 100  # for 100 day moving average

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
            print("\r", "Failed inserting:", str(t_df.iloc[i][0]), t_df.iloc[i][1], end='')

    con.commit()
    print("\r                                                    ")


#


stock_list = []
csvfile = open(symbols_filename, newline='')
reader = csv.reader(csvfile)

for row in reader:
    stock_list.append(row[0])

# detect_types is for timestamp support
con = sqlite3.connect(database_filename,
                      detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
cur = con.cursor()

create_database_if_needed()

# print("in main:", start_date, type(start_date))
download_start_date = find_download_start_date(start_date)

download_finish_date = finish_date

if download_start_date <= download_finish_date:
    download_stock_data(download_start_date, download_finish_date)
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
      finish_date + timedelta(days=1) )

stock_group_df = pd.DataFrame(cur.fetchall(),
                              columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
stock_group_df = stock_group_df.set_index(['ticker', 'date']).sort_index()

# print('stock_group_df:', stock_group_df)
valid_stock_symbol = stock_group_df.iloc[0].name[0]
print('Length of a stock in stock_group_df:', len(stock_group_df.loc[valid_stock_symbol]))

# Need to drop any extra rows

# Find actual start date
query = '''
select date from stock_data
order by date
limit 1
'''
cur.execute(query)
t = cur.fetchone()
print("Database start date:", t[0])

# Find actual finish date
query = '''
Select date From stock_data
Order By date Desc
limit 1
'''
cur.execute(query)
t = cur.fetchone()
print("Database finish date:", t[0])

# Make a list of the stocks in the database
query = '''
SELECT DISTINCT ticker
FROM stock_data
'''
cur.execute(query)
t = cur.fetchall()

stocks = []
for stock in t:
    stocks.append(stock[0])

# con.close()

# Correct start and finish dates so they are trading days
trading_days = stock_group_df.loc[stocks[0]].index  # "Date" is part of the MultiIndex
# print('Trading days from db:', trading_days)

start_day_range = pd.date_range(start_date - timedelta(days=extra_days),
                                start_date).tolist()
start_day_range.reverse()
print('start_day_range:', start_day_range)
found_start_day = False

for d in start_day_range:
    if d in trading_days:
        start_date = d
        found_start_day = True
        break

if found_start_day == False:
    print('Could not find a trading day for the start day.')
    sys.exit(1)

finish_day_range = pd.date_range(finish_date - timedelta(days=extra_days),
                                 finish_date).tolist()
finish_day_range.reverse()
found_finish_day = False

for d in finish_day_range:
    if d in trading_days:
        finish_date = d
        found_finish_day = True
        break

if found_finish_day == False:
    print('Could not find a trading day for the finish day.')
    sys.exit(1)

print("Corrected start:", start_date, " finish: ", finish_date)

####
# Calculate indicators

slope = {}
r_sq = {}  # R squared
annualized_return = {}
adjusted_slope = {}

x = {}
y = {}
plotly_x = {}
predicted_y = {}

# natural logarithm of price
stock_group_df.insert(0, 'ln', 0)
stock_group_df['ln'] = np.log(stock_group_df['close'])

print('Calculating indicators')
count = 0
for stock in stock_list:
    print("\r", stock, end='')
    # print(stock_group_df.loc[stock])

    # Get last 90 trading days of this stock
    sql = '''
    Select * From stock_data
    Where ticker = ?
    Order By date Desc
    Limit 90
    '''
    cur.execute(sql, [stock])
    # t = cur.fetchall()
    # print('t:', t)
    stock_df = pd.DataFrame(cur.fetchall(),
                            columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])

    # calculate adjusted slope

    # find if the stock moved +-15% in across 2 trading days

    # get last 100 trading days of this stock
    # calculate the 100 day moving average

    # Regression of the natural logarithm of price
    # Pandas.datetime to numpy.datetime64 to numpy.datetime64 days to a floating point number
    x[stock] = stock_group_df.loc[stock].index.values.astype("datetime64[D]").astype("float").reshape(-1, 1)
    plotly_x[stock] = stock_group_df.loc[stock].index.values.astype("datetime64[D]")
    y[stock] = stock_group_df.loc[stock].get('ln').values

    model = LinearRegression().fit(x[stock], y[stock])

    predicted_y[stock] = model.predict(x[stock])

    slope[stock] = model.coef_[0]

    r_sq[stock] = model.score(x[stock], y[stock])

    annualized_return[stock] = pow(math.exp(slope[stock]), 250) - 1.0

    adjusted_slope[stock] = r_sq[stock] * annualized_return[stock]

    # count = count + 1
    # if count > 5:
    #     break

con.close()

print("\rAnnualized rate of return, adjusted slope:")

output = sorted(adjusted_slope.items(), key=operator.itemgetter(1), reverse=True)
for t in output[0:10]:
    stock = t[0]
    print(stock, round(annualized_return[stock] * 100), '% ', round(t[1], 2))
    line1 = px.line(x=plotly_x[stock], y=y[stock], title=stock)
    line2 = px.line(x=plotly_x[stock], y=predicted_y[stock], title=stock)
    figure = go.Figure(data=line1.data + line2.data)
    figure.update_layout(title=stock)
    figure.show()
