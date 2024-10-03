import logging
import asyncio
import pandas as pd
from strategy_long_activation_trades import TradingStrategy  # Adjust the import based on your actual module
from binance.client import Client, AsyncClient
from dotenv import load_dotenv
import dotenv
import os
dotenv.load_dotenv()  # Load environment variables from a .env file

# Load API credentials from environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = AsyncClient(api_key=api_key, api_secret=api_secret)

class TradingAnalyzer:
    def __init__(self, client):
        self.client = client

    async def combine_dataframes(self, symbol, df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, candles_df):
        """
        Combine and analyze data from multiple DataFrames using the TradingStrategy class.

        :param symbol: The trading pair symbol (e.g., BTCUSDT).
        :param df_9d_tema: DataFrame containing the 9-day TEMA.
        :param df_7d_ma: DataFrame containing the 7-day moving average.
        :param df_14d_ma: DataFrame containing the 14-day moving average.
        :param df_30d_ma: DataFrame containing the 30-day moving average.
        :param candles_df: DataFrame containing OHLC data for candlestick patterns.
        """
        strategy_for_long_position = TradingStrategy(symbol, client) 

        try:
            # Ensure all DataFrames are valid
            for df in [df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, candles_df]:
                if df is None or df.empty:
                    logging.error(f"One of the DataFrames is invalid or empty for symbol {symbol}.")
                    return  # Exit the function early if any DataFrame is invalid

            # Fetch the current price for the symbol
            current_price = await self.fetch_current_price(symbol)
            if current_price is None:
                logging.error(f"Failed to fetch current price for {symbol}.")
                return  # Exit the function early if price fetching fails

            # Check if conditions are met to activate a long position
            long_position_activated = await strategy_for_long_position.should_activate_long_position(
                current_price, df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, candles_df
            )

            if long_position_activated:
                logging.info(f"Long position activated for {symbol}.")
            else:
                logging.info(f"Long position not activated for {symbol}.")

            # Analyze data with the strategy_for_long_position
            tema_above_moving_averages = await strategy_for_long_position.verify_tema_above_moving_averages(
                df_9d_tema, df_7d_ma, df_14d_ma, df_30d_ma
            )
            logging.info(f"TEMA above moving averages: {tema_above_moving_averages}")

            historical_behavior_ok = await strategy_for_long_position.check_historical_behavior(
                df_9d_tema, df_7d_ma, df_14d_ma, df_30d_ma, df_30d_tema=df
            )
            logging.info(f"Historical behavior check: {historical_behavior_ok}")

            ma_cross_confirmed = await strategy_for_long_position.confirm_moving_average_cross(
                df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma
            )
            logging.info(f"Moving average cross confirmed: {ma_cross_confirmed}")

            # tema_cross_confirmed = await strategy_for_long_position.confirm_tema_cross(
                # df_9d_tema, df_7d_ma, df_14d_ma, short_period=None, long_period=None
            # )
            # logging.info(f"TEMA cross confirmed: {tema_cross_confirmed}")

        except Exception as e:
            logging.error(f"An error occurred while processing symbol {symbol}: {str(e)}")
            await asyncio.sleep(1)  # Pause before next attempt

    async def fetch_current_price(self, symbol):
        """
        Fetch the current price for the symbol using the client.

        :param symbol: The trading pair symbol (e.g., BTCUSDT).
        :return: The current price of the asset.
        """
        try:
            # Implement the logic to fetch the current price from the client
            price = await self.client.get_symbol_price(symbol)  # Placeholder for actual implementation
            return price
        except Exception as e:
            logging.error(f"Failed to fetch current price for {symbol}: {e}")
            return None

# Example usage:
# client = YourClient()  # Instantiate your API client
# analyzer = TradingAnalyzer(client)
# await analyzer.combine_dataframes(symbol, df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, candles_df)
    
    async def close(self):
        """Close the Binance client session."""
        if self.client:
            await self.client.close_connection()
            logging.info("Binance client connection closed.")