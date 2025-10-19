import yfinance as yf
import pandas as pd
import datetime as dt



def fetch_historical_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:

    ## Fetch historical adjusted closing prices for a list of tickers from Yahoo Finance

    # Args: 
        # tickers (list): A list of stock ticker symbols (e.g.,).
        # start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        # end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    #Returns:
        # pd.DataFrame: A pandas DataFrame containing the historical adjusted closing prices,
        # with dates as the index and tickers as columns. Returns an empty
        # DataFrame if data fetching fails.

    try:
        # Download the historical data for the given tickers
        # Keep only "Ajd Close" to get the Adjusted closing price

        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False, group_by='ticker')

        successful_tickers = []
        failed_tickers = []
        
        # Check if there are failed fetches for individual tickers
        if len (tickers) == 1: 
            if not data.empty:
                successful_tickers.append(tickers)
                data = data[['Adj Close']]
            else:
                failed_tickers.append(tickers)

        else:
            for ticker in tickers:
                if ticker not in data or data[ticker]['Adj Close'].isnull().all():
                    failed_tickers.append(ticker)
                else:
                    successful_tickers.append(ticker)

        if successful_tickers:
            # We need to stack the multi-level columns and select the successful tickers
            clean_data = data.stack(level=0).unstack(level=1)['Adj Close'][successful_tickers]
        else:
            clean_data = pd.DataFrame()

        if successful_tickers:
            print(f"Successfully fetched data for: {', '.join(successful_tickers)}")
        if failed_tickers:
            print(f"WARNING: Could not fetch data for: {', '.join(failed_tickers)}")

        return clean_data, failed_tickers
    
    except Exception as e: 
        print(f"An error occured while fetching data: {e}")
        # Return empty df in case of error

        return pd.DataFrame()


def calculate_returns(price_data: pd.DataFrame, nan_threshold = 0.3) -> tuple:
    """
    Calculates annualized mean returns and covariance matrix from price data.

    This function performs a two-step cleaning process:
    1. Filters out stocks with a percentage of missing data greater than the threshold.
    2. Truncates the dataset to the common date range where all remaining assets have price data.

    Args:
        price_data (pd.DataFrame): DataFrame of historical adjusted closing prices.
        nan_threshold (float): The maximum allowed percentage of NaN values for a stock. Defaults to 0.3 (30%).

    Returns:
        tuple: A tuple containing:
            - A pandas Series of annualized mean returns.
            - A pandas DataFrame of the annualized covariance matrix.
            - A list of tickers dropped due to poor data quality.
    """
    # --- NEW: Step 1 - Column Filtering based on NaN threshold ---
    nan_percentage = price_data.isnull().sum() / len(price_data)
    tickers_to_keep = nan_percentage[nan_percentage <= nan_threshold].index.tolist()
    tickers_dropped_quality = nan_percentage[nan_percentage > nan_threshold].index.tolist()

    filtered_price_data = price_data[tickers_to_keep]

    if filtered_price_data.empty:
        # No tickers met the data quality requirement
        return pd.Series(), pd.DataFrame(), tickers_dropped_quality

    # --- Step 2: Row Filtering to get common date range ---
    clean_price_data = filtered_price_data.dropna()

    if clean_price_data.empty:
        # No common date range found for the remaining tickers.
        return pd.Series(), pd.DataFrame(), tickers_dropped_quality

    # Calculate daily returns
    returns = clean_price_data.pct_change().dropna()

    # Annualize the mean of daily returns (assuming 252 trading days in a year)
    mean_returns = returns.mean() * 252

    # Annualize the covariance of daily returns
    cov_matrix = returns.cov() * 252

    return mean_returns, cov_matrix, tickers_dropped_quality