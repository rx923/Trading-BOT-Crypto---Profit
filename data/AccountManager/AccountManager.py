import logging
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException

import dotenv
from time import time
import os
from decimal import Decimal, ROUND_DOWN
from typing import Optional
import random
import aiohttp, asyncio


dotenv.load_dotenv()  # Load environment variables from a .env file




# Set the event loop policy to SelectorEventLoop for compatibility with aiodns
if os.name == 'nt':  # Check if the OS is Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# Initialize the Binance client with appropriate API keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

client = AsyncClient.create(api_key, api_secret)


class AccountManager:
    def __init__(self, client, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        if not self.api_key or not self.api_secret:
            raise ValueError("API Key and Secret must be provided.")
        # Initialize the client as an instance variable
        self.client = client  # Initialize client as None


    async def initialize_client(self):
        if self.client is None:
            try:
                self.client = await AsyncClient.create(self.api_key, self.api_secret)
                logging.info("Binance async client initialized")
            except BinanceAPIException as e:
                logging.error(f"Binance API error initializing client: {e}", exc_info=True)
            except aiohttp.ClientError as e:
                logging.error(f"HTTP error initializing client: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Unexpected error initializing client: {e}", exc_info=True)


    async def fetch_futures_balance(self, asset: str) -> Optional[Decimal]:
        if not isinstance(self.client, AsyncClient):
            logging.error("Binance client is not properly initialized.")
            return None

        retries = 5
        for attempt in range(retries):
            try:
                # Fetch the balance information for futures account
                balance_info = await asyncio.wait_for(self.client.futures_account_balance(), timeout=10)

                # Filter balance information for the specific asset
                balance_data = next((item for item in balance_info if item['asset'] == asset), None)
                if balance_data is not None:
                    return Decimal(balance_data['balance'])
                else:
                    logging.error(f"Balance for {asset} not found in the fetched data.")
                    return None

            except BinanceAPIException as e:
                logging.error(f"Binance API Exception: {e.code} - {e.message}", exc_info=True)
                if e.code == -2015:
                    logging.error("Invalid API key or permissions. Please check your API key settings.")
                    return None
            except asyncio.TimeoutError:
                logging.error("Timeout error fetching balance information.")
            except Exception as e:
                logging.error(f"Unexpected error fetching balance for {asset}: {e}", exc_info=True)

            wait_time = 2 ** attempt + random.uniform(0, 1)
            logging.info(f"Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)

        logging.error("Max retries reached. Unable to fetch balance.")
        return None



    async def update_balance_periodically(self, asset='BNFCR', interval=120):
        if isinstance(self.client, AsyncClient):
            logging.info("Binance client is properly initialized. Starting periodic balance update.")
        else:
            logging.error("Binance client is not properly initialized.")
            return

        try:
            while True:
                balance = await self.fetch_futures_balance(asset)
                if balance is not None:
                    logging.info(f"Updated balance for {asset}: {balance:.4f}")
                else:
                    logging.warning(f"Failed to update balance for {asset}.")
                await asyncio.sleep(interval)
        except Exception as e:
            logging.error(f"An error occurred during the periodic balance update: {e}", exc_info=True)
        finally:
            logging.info("Periodic balance update has stopped.")
