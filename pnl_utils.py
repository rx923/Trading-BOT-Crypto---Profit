import logging
# from data.utils import check_price_direction_change

async def should_activate_long_position(client, symbol, current_price, df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma):
    try:
        logging.info(f"Checking conditions to activate long position for {symbol}")

        if df_7d_ma is None or df_9d_tema is None or df_14d_ma is None or df_30d_ma is None:
            logging.warning(f"Data not available for all required moving averages for {symbol}")
            return False

        condition_met = (df_9d_tema['TEMA'].iloc[-1] < df_7d_ma['MA_7'].iloc[-1] and
                        df_7d_ma['MA_7'].iloc[-1] > df_14d_ma['MA_14'].iloc[-1])

        if condition_met:
            logging.info(f"Conditions met to activate long position for {symbol}")
            return True
        else:
            logging.info(f"Conditions not met to activate long position for {symbol}")
            return False

    except Exception as e:
        logging.error(f"Error checking conditions to activate long position for {symbol}: {e}")
        return False

async def activate_long_position(client, symbol, current_price):
    try:
        logging.info(f"Activating long position for {symbol} at current price {current_price}")

        order_response = await client.futures_create_order(
            symbol=symbol,
            side='BUY',
            type='MARKET',
            quantity=1,  # Adjust quantity calculation as per your strategy
        )

        if 'orderId' in order_response:
            logging.info(f"Long position opened successfully for {symbol}: {order_response}")
        else:
            logging.error(f"Failed to open long position for {symbol}: {order_response}")

    except Exception as e:
        logging.error(f"Error activating long position for {symbol}: {e}")