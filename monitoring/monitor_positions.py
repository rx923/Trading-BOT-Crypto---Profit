# from binance.enums import *
# from binance.client import Client
# import asyncio
# import os


# api_key = os.getenv('BINANCE_API_KEY')
# api_secret = os.getenv('BINANCE_API_SECRET')

# client = Client(api_key, api_secret)



# BINANCE_BASE_URL = 'https://api4.binance.com'

# class PositionMonitor:
#     def __init__(self):
#         self.client = Client(api_key, api_secret)

#     async def monitor_position(self, symbol, position_size, leverage, entry_price, profit_target1, profit_target2, position_type, stop_loss_threshold, stop_loss_fluctuation_threshold, position_id):
#         while True:
#             current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            
#             if position_type == "long":
#                 if current_price >= entry_price * (1 + profit_target1):
#                     # Close 50% of the position
#                     quantity = (position_size * leverage) / entry_price
#                     order = self.client.futures_create_order(
#                         symbol=symbol,
#                         side=SIDE_SELL,
#                         type=FUTURE_ORDER_TYPE_MARKET,
#                         quantity=quantity / 2,
#                         newClientOrderId=f"{position_id}_TP1"  # Unique ID for take profit order
#                     )
#                     print(f"Closed 50% of long position: {order}")

#                     # Continue to monitor the remaining position for the second profit target
#                     while True:
#                         current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
#                         if current_price >= entry_price * (1 + profit_target2):
#                             order = self.client.futures_create_order(
#                                 symbol=symbol,
#                                 side=SIDE_SELL,
#                                 type=FUTURE_ORDER_TYPE_MARKET,
#                                 quantity=quantity / 2,
#                                 newClientOrderId=f"{position_id}_TP2"  # Unique ID for take profit order
#                             )
#                             print(f"Closed remaining 50% of long position: {order}")
#                             return
#                         await asyncio.sleep(60)

#                 elif current_price <= entry_price * (1 - stop_loss_threshold):
#                     # Check for stop loss fluctuation
#                     await asyncio.sleep(60)  # Wait for 1 minute to confirm the fluctuation
#                     new_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
#                     if new_price <= entry_price * (1 - stop_loss_fluctuation_threshold):
#                         stop_loss_order = self.client.futures_create_order(
#                             symbol=symbol,
#                             side=SIDE_SELL,
#                             type=FUTURE_ORDER_TYPE_STOP_MARKET,
#                             quantity=quantity,
#                             stopPrice=entry_price * (1 - stop_loss_threshold),  # Adjust stop loss based on threshold
#                             newClientOrderId=f"{position_id}_SL"  # Unique ID for stop loss order
#                         )
#                         print(f"Closed long position due to stop loss fluctuation: {stop_loss_order}")
#                         return

#             elif position_type == "short":
#                 if current_price <= entry_price * (1 - profit_target1):
#                     # Close 50% of the position
#                     quantity = (position_size * leverage) / entry_price
#                     order = self.client.futures_create_order(
#                         symbol=symbol,
#                         side=SIDE_BUY,
#                         type=FUTURE_ORDER_TYPE_MARKET,
#                         quantity=quantity / 2,
#                         newClientOrderId=f"{position_id}_TP1"  # Unique ID for take profit order
#                     )
#                     print(f"Closed 50% of short position: {order}")

#                     # Continue to monitor the remaining position for the second profit target
#                     while True:
#                         current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
#                         if current_price <= entry_price * (1 - profit_target2):
#                             order = self.client.futures_create_order(
#                                 symbol=symbol,
#                                 side=SIDE_BUY,
#                                 type=FUTURE_ORDER_TYPE_MARKET,
#                                 quantity=quantity / 2,
#                                 newClientOrderId=f"{position_id}_TP2"  # Unique ID for take profit order
#                             )
#                             print(f"Closed remaining 50% of short position: {order}")
#                             return
#                         await asyncio.sleep(60)

#                 elif current_price >= entry_price * (1 + stop_loss_threshold):
#                     # Check for stop loss fluctuation
#                     await asyncio.sleep(60)  # Wait for 1 minute to confirm the fluctuation
#                     new_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
#                     if new_price >= entry_price * (1 + stop_loss_fluctuation_threshold):
#                         stop_loss_order = self.client.futures_create_order(
#                             symbol=symbol,
#                             side=SIDE_BUY,
#                             type=FUTURE_ORDER_TYPE_STOP_MARKET,
#                             quantity=quantity,
#                             stopPrice=entry_price * (1 + stop_loss_threshold),  # Adjust stop loss based on threshold
#                             newClientOrderId=f"{position_id}_SL"  # Unique ID for stop loss order
#                         )
#                         print(f"Closed short position due to stop loss fluctuation: {stop_loss_order}")
#                         return

#             await asyncio.sleep(60)