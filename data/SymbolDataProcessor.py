# from binance.client import AsyncClient, Client
# import os
# import logging
# import asyncio
# import math
# from typing import Optional, Tuple
# from binance.exceptions import BinanceAPIException
# import dotenv
# dotenv.load_dotenv()  # Load environment variables from a .env file

# # Load API credentials from environment variables
# api_key = os.getenv('BINANCE_API_KEY')
# api_secret = os.getenv('BINANCE_API_SECRET')
# client = AsyncClient.create(api_key=api_key, api_secret=api_secret)


# class SymbolDataProcessor:
#     def __init__(self, client):
#         self.client = client
#         self.api_key = os.getenv('BINANCE_API_KEY')
#         self.api_secret = os.getenv('BINANCE_API_SECRET')

#     async def fetch_symbol_info(self, symbol: str) -> Optional[dict]:
#         try:
#             exchange_info = await self.client.futures_exchange_info()
#             if not exchange_info or 'symbols' not in exchange_info:
#                 logging.warning("Exchange info does not contain 'symbols'.")
#                 return None

#             symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
#             if symbol_info:
#                 logging.info(f"Found symbol {symbol} in exchange info.")
#                 price_precision = symbol_info['pricePrecision']
#                 quantity_precision = symbol_info['quantityPrecision']
#                 return {'symbol': symbol, 'pricePrecision': price_precision, 'quantityPrecision': quantity_precision}

#             logging.warning(f"Symbol {symbol} not found in exchange info.")
#             return None

#         except BinanceAPIException as e:
#             logging.error(f"Binance API Exception while fetching symbol info for {symbol}: {e}")
#             return None

#         except Exception as e:
#             logging.error(f"Error fetching symbol info for {symbol}: {e}")
#             return None

#     async def process_symbol(self, symbol: str) -> Tuple[Optional[int], Optional[int]]:
#         try:
#             logging.info(f"Processing symbol {symbol}")
#             symbol_info = await self.fetch_symbol_info(symbol)
#             if not symbol_info:
#                 logging.warning(f"Symbol {symbol} not found or fetch failed. Skipping.")
#                 return None, None

#             price_precision = symbol_info.get('pricePrecision')
#             quantity_precision = symbol_info.get('quantityPrecision')
#             logging.info(f"{symbol} - pricePrecision: {price_precision}, quantityPrecision: {quantity_precision}")

#             return price_precision, quantity_precision

#         except BinanceAPIException as e:
#             logging.error(f"Binance API Exception while processing symbol {symbol}: {e}")
#             return None, None

#         except KeyError as e:
#             logging.error(f"KeyError while processing symbol {symbol}: {e}")
#             return None, None

#         except Exception as e:
#             logging.error(f"Unexpected error processing symbol {symbol}: {e}")
#             return None, None



#     async def fetch_symbol_precisions(self, symbols: list[str]) -> None:
#         tasks = [self.process_symbol(symbol) for symbol in symbols]
#         results = await asyncio.gather(*tasks, return_exceptions=True)

#         for symbol, result in zip(symbols, results):
#             if isinstance(result, tuple) and all(isinstance(prec, (int, type(None))) for prec in result):
#                 price_precision, quantity_precision = result
#                 if price_precision is not None and quantity_precision is not None:
#                     logging.info(f"Symbol: {symbol}, pricePrecision: {price_precision}, quantityPrecision: {quantity_precision}")
#                 else:
#                     logging.error(f"Precisions for {symbol} returned as None")
#             else:
#                 logging.error(f"Failed to fetch precisions for {symbol}")


#     async def get_price_precision(self, symbol: str, precision_levels: dict) -> Optional[int]:
#         try:
#             if precision_levels and symbol in precision_levels:
#                 return precision_levels[symbol]['price_precision']

#             symbol_info = await self.client.get_symbol_info(symbol)
#             if symbol_info:
#                 filters = symbol_info.get('filters', [])
#                 for f in filters:
#                     if f['filterType'] == 'PRICE_FILTER':
#                         tick_size = float(f['tickSize'])
#                         return int(-math.log10(tick_size))

#                 logging.error(f"Price filter not found for symbol {symbol}. Skipping order.")
#                 return None
#             else:
#                 logging.error(f"Symbol information not found for {symbol}. Skipping order.")
#                 return None

#         except Exception as e:
#             logging.error(f"Error fetching price precision for {symbol}: {e}")
#             return None
        
        
#     async def close(self):
#             """Close the Binance client session."""
#             if self.client:
#                 await self.client.close_connection()
#                 logging.info("Binance client connection closed.")