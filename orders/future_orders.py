# from binance.client import Client
# from binance.exceptions import BinanceAPIException
# import logging
# import os
# import time
# # from data.PositionSizer import PositionSizer


# # Initialize the Binance client with appropriate API keys
# api_key = os.getenv('BINANCE_API_KEY')
# api_secret = os.getenv('BINANCE_API_SECRET')
# client = Client(api_key, api_secret)


# def get_account_balance_with_retry(retries=3, delay=5):
#     attempt = 0
#     while attempt < retries:
#         try:
#             balance = client.futures_account_balance()
#             logging.info(f"Account balance: {balance}")
#             return balance
#         except BinanceAPIException as e:
#             logging.error(f"API error fetching account balance: {str(e)}")
#         except Exception as e:
#             logging.error(f"General error: {str(e)}")
#         attempt += 1
#         time.sleep(delay)
#     return None





# def get_order_id_for_level(symbol, percentage):
#     """
#     Retrieve the order ID for the take profit order at a specific percentage level.
#     """
#     try:
#         # Implement logic to fetch order ID from Binance Futures API
#         # Example: Assuming you have a way to fetch orders based on symbol and percentage
#         orders = client.futures_get_open_orders(symbol=symbol)

#         for order in orders:
#             # Check if the order matches the specific percentage level
#             # Assuming your order has some metadata or tags to identify the take profit order
#             if 'percentage' in order and order['percentage'] == percentage:
#                 return order['orderId']  # Return the order ID

#         # If no order is found for the specific percentage level, return None
#         return None

#     except Exception as e:
#         print(f"Error fetching order ID for {symbol} at {percentage * 100}%: {str(e)}")
#         return None
    


# def update_take_profit_order(symbol, entry_price, total_position_size):
#     """
#     Update or place multiple take profit orders at different percentage levels.
#     """
#     try:
#         # Define take profit levels and corresponding percentages
#         take_profit_levels = [
#             {'percentage': 0.20, 'triggered': False},  # 20% take profit level
#             {'percentage': 0.50, 'triggered': False},  # 50% take profit level
#             {'percentage': 0.75, 'triggered': False},  # 75% take profit level
#             {'percentage': 1.00, 'triggered': False}   # 100% take profit level
#         ]

#         for level in take_profit_levels:
#             if not level['triggered']:
#                 take_profit_price = calculate_take_profit_price(entry_price, level['percentage'])
#                 if take_profit_price is not None:
#                     try:
#                         # Check if take profit order already exists, if so, update it; otherwise, place a new order
#                         order_id = get_order_id_for_level(symbol, level['percentage'])
#                         if order_id:
#                             updated_order = client.futures_cancel_order(symbol=symbol, orderId=order_id)
#                             print(f"Cancelled existing take profit order for {symbol} at {level['percentage'] * 100}%: {order_id}")
#                             print(f"Updated order details: {updated_order}")
#                         else:
#                             quantity = PositionSizer.calculate_position_size(symbol, total_position_size)
#                             order = client.futures_create_order(
#                                 symbol=symbol,
#                                 side='BUY',  # Close a short position
#                                 type='TAKE_PROFIT_MARKET',  # Use 'TAKE_PROFIT_LIMIT' for limit order
#                                 quantity=quantity,  # Adjust quantity as needed
#                                 price=take_profit_price,  # Set the take profit price
#                                 reduceOnly=True  # Close only the position, not add to it
#                             )
#                             print(f"Placed take profit order for {symbol} at {level['percentage'] * 100}%: {take_profit_price}")
#                             print(f"Order details: {order}")
                        
#                         level['triggered'] = True  # Mark this level as triggered

#                     except Exception as e:
#                         print(f"Error managing take profit order at {level['percentage'] * 100}%: {str(e)}")
#                         continue

#         # Optionally, log completion or success message
#         print(f"All take profit orders updated successfully for {symbol}.")

#     except Exception as e:
#         raise RuntimeError(f"Error updating take profit orders for {symbol}: {str(e)}")




# def update_stop_loss_order(symbol, stop_loss_price):
#     """
#     Update or place a stop loss order for a specific symbol.
#     """
#     try:
#         # First, check if there's an existing stop loss order for the symbol
#         existing_orders = client.futures_get_open_orders(symbol=symbol)

#         for order in existing_orders:
#             if order['type'] == 'STOP_MARKET' and order['reduceOnly'] and order['side'] == 'SELL':
#                 # Update the existing stop loss order
#                 order_id = order['orderId']
#                 updated_order = client.futures_cancel_order(symbol=symbol, orderId=order_id)
#                 print(f"Cancelled existing stop loss order for {symbol} with order ID {order_id}")
#                 print(f"Updated order details: {updated_order}")  # Print for debugging

#         # Place a new stop loss order
#         stop_loss_order = client.futures_create_order(
#             symbol=symbol,
#             side='SELL',  # Since it's a stop loss for a short position, SELL to close position
#             type='STOP_MARKET',
#             stopPrice=stop_loss_price,
#             reduceOnly=True,  # Close the entire position when triggered
#             closePosition=True  # Adjust as per your strategy
#         )

#         print(f"Placed stop loss order for {symbol} at {stop_loss_price}")
#         print(f"Stop loss order details: {stop_loss_order}")  # Print for debugging

#     except Exception as e:
#         print(f"Error updating stop loss order for {symbol}: {str(e)}")


# def calculate_take_profit_price(symbol, percentage_decrease):
#     """
#     Calculate the take profit price based on the current price of the trading pair
#     and percentage decrease.
#     """
#     try:
#         # Fetch current price of the symbol from Binance
#         ticker = client.get_symbol_ticker(symbol=symbol)
#         current_price = float(ticker['price'])

#         # Calculate take profit price based on current price and percentage decrease
#         take_profit_price = current_price * (1 - percentage_decrease)
#         return take_profit_price

#     except Exception as e:
#         print(f"Error fetching price for {symbol}: {str(e)}")
#         return None


# def place_take_profit_order(symbol, take_profit_price, percentage, total_position_size):
#     """
#     Place a new take profit order for a specific percentage level.
#     """
#     try:
#         quantity = 0.25 * total_position_size  # Adjust position size calculation as needed
#         order = client.futures_create_order(
#             symbol=symbol,
#             side='BUY',  # Since we are short, BUY to close position
#             type='LIMIT',
#             quantity=quantity,
#             price=take_profit_price,
#             reduceOnly=True  # Adjust as per your strategy
#         )
    
#         print(f"Placed take profit order for {symbol} at {percentage * 100}%: {take_profit_price}")
#     except Exception as e:
#         raise RuntimeError(f"Error placing take profit order for {symbol} at {percentage * 100}%: {str(e)}")


# def manage_take_profit_orders(symbol, entry_price, total_position_size):
#     """
#     Manage take profit orders based on the entry price and position size.
#     This function will handle placing new orders and updating existing ones.
#     """
#     # Define take profit levels and corresponding percentages
#     take_profit_levels = [
#         {'percentage': 0.006, 'triggered': False},
#         {'percentage': 0.015, 'triggered': False},
#         {'percentage': 0.03, 'triggered': False},
#         {'percentage': 0.80, 'triggered': False}
#     ]

#     for level in take_profit_levels:
#         if not level['triggered']:
#             take_profit_price = calculate_take_profit_price(symbol, level['percentage'])
#             if take_profit_price is not None:
#                 try:
#                     # Place a new take profit order
#                     place_take_profit_order(symbol, take_profit_price, level['percentage'], total_position_size)

#                     level['triggered'] = True  # Mark this level as triggered
#                 except Exception as e:
#                     print(f"Error managing take profit order at {level['percentage'] * 100}%: {str(e)}")
#                     continue


# # Example function to close position
# def close_position(symbol):
#     try:
#         # Example: Implement logic to close position using Binance Futures API or other platform API
#         # client.futures_cancel_order()  # Example cancel order if necessary
#         # client.futures_create_order()  # Example create order to close position
#         print(f"Position closed successfully for {symbol}.")
#     except Exception as e:
#         print(f"Error closing position for {symbol}: {e}")









# async def monitor_position(symbol, entry_price, position_size, stop_loss_price, take_profit_price):
#     try:
#         while True:
#             # Fetching position information and monitoring PnL
#             position_info = client.futures_position_information(symbol=symbol.upper())
#             current_price = fetch_current_price(symbol)

#             # Calculating unrealized PnL and implementing stop loss/take profit logic
#             for position in position_info:
#                 if float(position['entryPrice']) == entry_price:
#                     pnl = float(position['unRealizedProfit'])
#                     logging.info(f"Unrealized PnL for {symbol}: {pnl}")

#                     if current_price >= stop_loss_price:
#                         # Execute stop loss
#                         logging.info(f"Stop loss hit for {symbol}. Closing position.")
#                         await close_position(symbol, position_size)
#                         return

#                     if current_price <= take_profit_price:
#                         # Execute take profit
#                         logging.info(f"Take profit reached for {symbol}. Closing position.")
#                         await close_position(symbol, position_size)
#                         return

#                     # Implement partial profit taking
#                     if pnl >= 0.5 * entry_price:
#                         quantity = position_size / 2
#                         side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
#                         try:
#                             order_id = await client.futures_create_order(
#                                 symbol=symbol,
#                                 side=side,
#                                 type='MARKET',
#                                 quantity=quantity,
#                                 reduceOnly=True,
#                                 closePosition=False
#                             )
#                             logging.info(f"Partial profit order placed for {symbol}: {side} {quantity} at market price. Order ID: {order_id}")
#                             # Log or handle order_id as needed
#                         except Exception as e:
#                             logging.error(f"Error placing partial profit order for {symbol}: {e}")

#             await asyncio.sleep(0.1)  # Adjust sleep interval as needed
    
#     except Exception as e:
#         logging.error(f"Error monitoring position for {symbol}: {e}")
