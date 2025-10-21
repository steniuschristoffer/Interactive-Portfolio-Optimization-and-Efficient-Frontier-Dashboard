import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import scipy.optimize as sco



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
            clean_data = data.stack(level=0, future_stack= True).unstack(level=1)['Adj Close'][successful_tickers]
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

def calculate_returns(price_data: pd.DataFrame, nan_threshold = 0.03) -> tuple:
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
    # --- Step 1 - Column Filtering based on NaN threshold ---
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

def run_monte_carlo_simulation(num_portfolios: int, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Runs a Monte Carlo simulation to generate a large number of random portfolios.

    Args:
        num_portfolios (int): The number of random portfolios to generate.
        mean_returns (pd.Series): A pandas Series of annualized mean returns for each asset.
        cov_matrix (pd.DataFrame): The annualized covariance matrix of returns for the assets.

    Returns:
        pd.DataFrame: A DataFrame containing the returns, volatility, and weights for each simulated portfolio.
    """
    num_assets = len(mean_returns)
    results = np.zeros((3 + num_assets, num_portfolios)) # 3 for return, volatility, sharpe + one for each weight

    for i in range(num_portfolios):
        # 1. Generate random weights that sum to 1
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # 2. Calculate portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Store the results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        # We will calculate Sharpe Ratio later, but reserve the space for it
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Store the weights for each asset
        for j in range(len(weights)):
            results[j + 3, i] = weights[j]

    # Convert results array to a DataFrame for easier handling
    column_names = ['return', 'volatility', 'sharpe'] + [ticker for ticker in mean_returns.index]
    results_df = pd.DataFrame(results.T, columns=column_names)
    
    print(f"Monte Carlo simulation completed with {num_portfolios} portfolios.")
    return results_df

def calculate_optimal_portfolios(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02) -> tuple:
    """
    Calculates the optimal portfolios for Maximum Sharpe Ratio and Minimum Volatility.

    Args:
        mean_returns (pd.Series): A pandas Series of annualized mean returns.
        cov_matrix (pd.DataFrame): The annualized covariance matrix of returns.
        risk_free_rate (float): The risk-free rate for the Sharpe Ratio calculation.

    Returns:
        tuple: A tuple containing:
            - A dictionary with the details of the max sharpe portfolio.
            - A dictionary with the details of the min volatility portfolio.
    """
    num_assets = len(mean_returns)

    # Helper function to calculate portfolio performance
    def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
        returns = np.sum(mean_returns * weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (returns - risk_free_rate) / volatility
        return returns, volatility, sharpe

    # Objective function to minimize (negative Sharpe Ratio)
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

    # Objective function to minimize (volatility)
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return portfolio_performance(weights, mean_returns, cov_matrix, 0)[1]

    # --- Optimization for Maximum Sharpe Ratio ---

    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds: each weight must be between 0 and 1
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial guess: equal weights
    initial_guess = num_assets * [1. / num_assets]

    max_sharpe_solver = sco.minimize(neg_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    
    max_sharpe_weights = max_sharpe_solver.x

    max_sharpe_return, max_sharpe_vol, max_sharpe_ratio_val = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_portfolio = {
        'return': max_sharpe_return,
        'volatility': max_sharpe_vol,
        'sharpe_ratio': max_sharpe_ratio_val,
        'weights': max_sharpe_weights
    }

    # --- Optimization for Minimum Volatility ---
    min_vol_solver = sco.minimize(portfolio_volatility, initial_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

    min_vol_weights = min_vol_solver.x
    min_vol_return, min_vol_vol, min_vol_sharpe = portfolio_performance(min_vol_weights, mean_returns, cov_matrix, risk_free_rate)

    min_vol_portfolio = {
        'return': min_vol_return,
        'volatility': min_vol_vol,
        'sharpe_ratio': min_vol_sharpe,
        'weights': min_vol_weights
    }
    
    print("Optimal portfolio calculations completed.")
    return max_sharpe_portfolio, min_vol_portfolio

def calculate_efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, num_points: int = 100) -> tuple:
    """
    Calculates the efficient frontier by finding the minimum volatility for a range of target returns.

    Args:
        mean_returns (pd.Series): A pandas Series of annualized mean returns.
        cov_matrix (pd.DataFrame): The annualized covariance matrix of returns.
        num_points (int): The number of points to calculate on the frontier curve.

    Returns:
        tuple: A tuple containing:
            - A list of portfolio volatilities for each point on the frontier.
            - A list of portfolio returns for each point on the frontier.
    """
    num_assets = len(mean_returns)
    
    # Objective function to minimize (volatility)
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Bounds and initial guess are the same as before
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    # We'll trace the frontier from the min volatility portfolio's return up to the max single asset return
    # First, find the min volatility return to set the start of our range
    min_vol_solver = sco.minimize(portfolio_volatility, initial_guess,
                                  args=(mean_returns, cov_matrix),
                                  method='SLSQP', bounds=bounds, 
                                  constraints=({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}))
    min_vol_return = np.sum(mean_returns * min_vol_solver.x)
    max_return = mean_returns.max()

    target_returns = np.linspace(min_vol_return, max_return, num_points)
    frontier_volatilities = []

    for target in target_returns:
        # Add the new constraint for the target return
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: np.sum(mean_returns * weights) - target}
        )
        
        solver = sco.minimize(portfolio_volatility, initial_guess,
                              args=(mean_returns, cov_matrix),
                              method='SLSQP', bounds=bounds, constraints=constraints)
        
        # We get the volatility from the solver's result (the minimized value)
        frontier_volatilities.append(solver.fun)

    return frontier_volatilities, target_returns