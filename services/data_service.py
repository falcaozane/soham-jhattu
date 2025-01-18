import pandas as pd
import yfinance as yf
from datetime import datetime

def get_data(assets):
    df = pd.DataFrame()
    #yf.pdr_override()
    end_date = datetime.now()
    for stock in assets:
        try:
            df[stock] = yf.download(stock, start='2014-01-01', end=end_date.strftime('%Y-%m-%d'), interval='1d')['Adj Close']
        except Exception as e:
            print(f"Failed to download data for {stock}: {e}")
            df[stock] = pd.Series(dtype=float)  # Create an empty series for the failed ticker
    return df
