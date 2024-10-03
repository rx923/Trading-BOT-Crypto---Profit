import logging
from binance.client import AsyncClient
import os
import json
import datetime


class OrderFileManager:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')

    async def initialize_client(self, api_key, api_secret):
        """ Initialize Binance async client """
        try:
            self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
            logging.info("Binance async client initialized")
        except Exception as e:
            logging.error(f"Error initializing Binance async client: {e}")

    def save_order_details(self, symbol, details):
        try:
            order_file = f"{symbol}_orders.json"
            with open(order_file, 'a') as f:
                json.dump(details, f)
                f.write('\n')  # Separate entries by new lines
            logging.info(f"Order details saved for {symbol}")
        except Exception as e:
            logging.error(f"Error saving order details for {symbol}: {e}")

    def update_order_details(self, symbol, order_id, updated_details):
        try:
            order_file = f"{symbol}_orders.json"
            if os.path.exists(order_file):
                with open(order_file, 'r') as f:
                    all_orders = f.readlines()

                # Find the relevant order by order_id and update its details
                found = False
                updated_orders = []
                for order in all_orders:
                    order_data = json.loads(order)
                    if order_data.get('orderId') == order_id:
                        order_data.update(updated_details)
                        updated_orders.append(order_data)
                        found = True
                    else:
                        updated_orders.append(order_data)

                if found:
                    with open(order_file, 'w') as f:
                        for order in updated_orders:
                            json.dump(order, f)
                            f.write('\n')
                    logging.info(f"Order details updated for order ID {order_id} in {symbol}")
                else:
                    logging.warning(f"Order ID {order_id} not found in {symbol}'s order file. No update performed.")
            else:
                logging.warning(f"Order details file not found for {symbol}. Cannot update.")
        except Exception as e:
            logging.error(f"Error updating order details for {symbol}: {e}")

    def get_order_details(self, symbol):
        try:
            order_file = f"{symbol}_orders.json"
            if os.path.exists(order_file):
                with open(order_file, 'r') as f:
                    orders = f.readlines()
                return [json.loads(order.strip()) for order in orders]
            else:
                logging.warning(f"Order details file not found for {symbol}.")
                return []
        except Exception as e:
            logging.error(f"Error getting order details for {symbol}: {e}")
            return []

    def remove_order_file(self, symbol):
        order_file = f"{symbol}_orders.json"
        if os.path.exists(order_file):
            os.remove(order_file)
            logging.info(f"Order file removed for {symbol}")

    def log_position_opened(self, symbol, order_id, entry_price):
        try:
            order_file = f"{symbol}_orders.json"
            timestamp = datetime.datetime.utcnow().isoformat()
            details = {
                'timestamp': timestamp,
                'symbol': symbol,
                'order_id': order_id,
                'entry_price': entry_price,
                'closure_price': None,  # Placeholder for closure price
                'realized_pnl': None,   # Placeholder for realized PNL
                'total_value': None     # Placeholder for total value
            }
            self.save_order_details(symbol, details)
        except Exception as e:
            logging.error(f"Error logging opened position for {symbol}: {e}")

    def log_position_closed(self, symbol, order_id, closure_price, realized_pnl):
        try:
            updated_details = {
                'closure_price': closure_price,
                'realized_pnl': realized_pnl
            }
            self.update_order_details(symbol, order_id, updated_details)
        except Exception as e:
            logging.error(f"Error logging closed position for {symbol}: {e}")

    async def close(self):
        await self.client.close_connection()