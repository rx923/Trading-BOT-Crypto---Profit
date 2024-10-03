# import asyncio
# import logging
# # import ccxt.async_support as ccxt
# import pandas as pd
# import datetime
# from colorama import Fore, Style, Back, init
# from tabulate import tabulate
# import aiohttp
# import os
# from datetime import datetime
# from dotenv import load_dotenv
# # from binance.enums import KLINE_INTERVAL_1DAY
# from binance.client import Client, AsyncClient
# from binance.exceptions import BinanceAPIException
# import random
# import json
# import smtplib
# import math
# from time import time
# import aiofiles
# import uuid
# from typing import Optional, Tuple
# from functools import partial
# from decimal import Decimal, ROUND_DOWN

# base_url = 'https://fapi.binance.com/fapi/v2/'
# # Load environment variables from .env file
# load_dotenv()

# api_key = os.getenv('BINANCE_API_KEY')
# api_secret = os.getenv('BINANCE_API_SECRET')
# # client = Client(api_key, api_secret)



# class BinanceClient:
#     def __init__(self, api_key=None, api_secret=None):
#         self.api_key = api_key or os.getenv('BINANCE_API_KEY')
#         self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')

#         if not self.api_key or not self.api_secret:
#             raise ValueError("API key or secret not found. Please check your .env file.")

#         self.client = AsyncClient.create(api_key, api_secret)

#     async def initialize(self):
#         logging.info("Initializing Binance client...")
#         self.client = await AsyncClient.create(self.api_key, self.api_secret)
#         logging.info("Binance client initialized.")

#     async def close(self):
#         if self.client:
#             logging.info("Closing Binance client connection...")
#             await self.client.close_connection()
#             logging.info("Binance client connection closed.")

#     async def fetch_usdt_balance(self):
#         try:
#             logging.info("Fetching USDT balance...")
#             account_info = await self.client.futures_account_balance()

#             balances = account_info.get('assets', [])

#             for asset in balances:
#                 if asset['asset'] == 'USDT':
#                     balance = float(asset.get('walletBalance', 0))
#                     logging.info(f"Current USDT balance: {balance:.4f}")
#                     return balance

#             logging.warning("No USDT balance found.")
#             return None

#         except BinanceAPIException as e:
#             logging.error(f"Binance API error fetching USDT balance: {e}")
#             return None

#         except Exception as e:
#             logging.error(f"Error fetching USDT balance: {e}")
#             return None

# async def main():
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#     client = BinanceClient()

#     try:
#         await client.initialize()

#         balance = await client.fetch_usdt_balance()

#         if balance is not None:
#             logging.info(f"Fetched USDT balance: {balance:.4f}")
#         else:
#             logging.warning("USDT balance is None.")

#     except ValueError as e:
#         logging.error(str(e))
    
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")

#     finally:
#         await client.close()

# if __name__ == "__main__":
#     asyncio.run(main())
