import logging
import pandas as pd
from binance.client import Client
api_key  = '1ogI8fNLApPLGseEXtbOHMK3RaYGyOyx8P4mQjvWyq2Ek9uNP0OUQo224TshdRy3'
api_secret  = 'DzpvMIjo2ULeVSnKuQGi1ygU8QGVSkp73deVPiY1H4xWWOGJOGrilI0RaIDQiPDO' 
client = Client(api_key, api_secret)

def fetch_historical_data(symbol, interval='1m', limit=200):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data = data[['open', 'high', 'low', 'close', 'volume']]
        data['close'] = data['close'].astype(float)
        return data
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()
    