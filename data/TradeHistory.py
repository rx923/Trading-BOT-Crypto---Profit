from tabulate import tabulate 
from binance.client import AsyncClient
import logging
import datetime
from colorama import Fore, Style, Back

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Set colors based on log level
        if record.levelno == logging.INFO:
            color = Fore.GREEN
        elif record.levelno == logging.WARNING:
            color = Fore.YELLOW
        elif record.levelno == logging.ERROR:
            color = Fore.RED
        else:
            color = Style.RESET_ALL

        # Format the message with color
        formatted_message = f"{color}{record.levelname} - {record.getMessage()}{Style.RESET_ALL}"
        return formatted_message

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = ColoredFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)

class TradeHistory:
    def __init__(self, trade_history_file='trade_history.txt'):
        self.trade_history_file = trade_history_file
        self.trade_history = []

    async def initialize_client(self, api_key, api_secret):
        """ Initialize Binance async client """
        try:
            self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
            logging.info("Binance async client initialized")
        except Exception as e:
            logging.error(f"Error initializing Binance async client: {e}")

    async def update_trade_history(self, profitable, size, price, symbol, side, realized_pnl):
        trade_entry = {
            'date_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol,
            'side': side,
            'initial_position': size,
            'entry_price': price,
            'profitable': profitable,
            'realized_pnl': realized_pnl
        }
        self.trade_history.append(trade_entry)

        if len(self.trade_history) > 100:
            self.trade_history.pop(0)

        try:
            with open(self.trade_history_file, 'a') as f:
                # Convert dict_keys to a list
                f.write(tabulate(
                    [trade_entry.values()],
                    headers=list(trade_entry.keys()),  # Convert to list
                    tablefmt='plain'
                ) + '\n')
                logging.info(f"Trade logged: {trade_entry}")
        except Exception as e:
            logging.error(f"Failed to log trade history to file: {e}")

    async def close(self):
            """Close the Binance client session."""
            if self.client:
                await self.client.close_connection()
                logging.info("Binance client connection closed.")