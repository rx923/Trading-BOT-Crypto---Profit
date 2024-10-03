import aiohttp
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()


api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# Example class for managing exchange instance
class ExchangeInstance:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    async def fetch_account_balance(self):
        try:
            balance_endpoint = "/fapi/v2/balance"
            url = self.base_url + balance_endpoint

            headers = {
                "Content-Type": "application/json",
                "X-MBX-APIKEY": self.api_key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {item['asset']: float(item['balance']) for item in data}
                    else:
                        logging.error(f"Failed to fetch account balance. Status code: {response.status}")
                        return None

        except Exception as e:
            logging.error(f"Error fetching account balance: {e}")
            return None

# Example usage
async def main():
    base_url = 'https://api.binance.com'
    exchange_instance = ExchangeInstance(api_key, api_secret, base_url)

    # Fetch account balance
    account_balance = await exchange_instance.fetch_account_balance()
    if not account_balance:
        logging.error("Failed to fetch account balance. Exiting.")
        return

    # Use account balance as needed
    print(f"Account balance: {account_balance}")

# Run the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())