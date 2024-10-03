# async def check_confirmation_patterns(symbol):
#     try:
#         # Check confirmation criteria on 3m and 5m intervals
#         for higher_interval in ['1m', '3m', '5m']:
#             higher_df = fetch_historical_data(symbol, interval=higher_interval, limit=50)
#             higher_indicators = calculate_indicators(symbol, higher_interval, higher_df)
            
#             if higher_indicators is None or higher_indicators.empty:
#                 logging.warning(f"Failed to fetch or calculate indicators for {symbol} on {higher_interval}. Skipping.")
#                 continue

#             # Check if TEMA is above SMAs
#             if (higher_indicators['TEMA_9'].iloc[-1] > higher_indicators['SMA_7'].iloc[-1] and
#                 higher_indicators['TEMA_9'].iloc[-1] > higher_indicators['SMA_14'].iloc[-1] and
#                 higher_indicators['TEMA_9'].iloc[-1] > higher_indicators['SMA_30'].iloc[-1]):

#                 # Check for bearish hammer candlestick pattern
#                 if check_bearish_hammer(higher_df):
#                     logging.info(f"Bearish hammer at the top formation detected on {higher_interval} for {symbol}.")
#                     return True

#                 logging.info(f"Additional confirmation met on {higher_interval} for {symbol}. Waiting for further confirmation.")

#                 # Wait for the next candle update to confirm downward movement
#                 await asyncio.sleep(10)  # Wait 10 seconds for the next candle

#                 # Fetch updated data and confirm criteria
#                 df = fetch_historical_data(symbol, interval='1m', limit=50)
#                 indicators_new = calculate_indicators(symbol, '1m', df)

#                 # Additional confirmation checks
#                 if (indicators_new['TEMA_9'].iloc[-1] < indicators_new['SMA_7'].iloc[-1]):
#                     logging.info(f"All confirmation criteria met for {symbol} ({higher_interval}).")
#                     return True

#         return False  # Return False if additional confirmation criteria are not fully met

#     except Exception as e:
#         logging.error(f"Error checking confirmation patterns for {symbol}: {e}")
#         return False


# def send_alert(message):
#     print(message)  # Replace with actual alerting mechanism




# async def place_futures_order(symbol, quantity, side):
#     try:
#         order = await client.futures_create_order(
#             symbol=symbol,
#             side=side,
#             type='MARKET',
#             quantity=quantity,
#             leverage=30
#         )
#         print(f"Futures market order placed: {order}")
#         return order['orderId']  # Return order ID or other identifier
#     except Exception as e:
#         print(f"Error placing futures market order: {e}")
#         return None
