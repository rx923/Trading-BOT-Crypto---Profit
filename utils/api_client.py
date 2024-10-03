# import time
# from binance.client import Client
# from binance.exceptions import BinanceAPIException
# import logging


# api_key  = '1ogI8fNLApPLGseEXtbOHMK3RaYGyOyx8P4mQjvWyq2Ek9uNP0OUQo224TshdRy3'
# api_secret  = 'DzpvMIjo2ULeVSnKuQGi1ygU8QGVSkp73deVPiY1H4xWWOGJOGrilI0RaIDQiPDO' 


# client = Client(api_key, api_secret)



# # Fetch current price of the symbol
# def fetch_current_price(symbol):
#     try:
#         ticker = client.futures_symbol_ticker(symbol=symbol)
#         return float(ticker['price'])
#     except Exception as e:
#         logging.error(f"Error fetching current price for {symbol}: {str(e)}")
#         return None
    
# def order_exists_for_level(symbol, percentage):
#     try:
#         # Example: Retrieve existing orders for the symbol from Binance
#         orders = client.futures_get_open_orders(symbol=symbol)
#         for order in orders:
#             if order['type'] == 'TAKE_PROFIT' and float(order['price']) == percentage:
#                 return True
#         return False
#     except Exception as e:
#         logging.error(f"Error checking order existence for {symbol} at percentage {percentage}: {e}")
#         return False

