# app.py

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from components.layout import create_layout
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Import all our backend functions
from logic.portfolio_optimizer import (
    fetch_historical_data, 
    calculate_returns, 
    run_monte_carlo_simulation,
    calculate_optimal_portfolios,
    calculate_efficient_frontier
)

# Use the CYBORG theme for a dark look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# Set the app's layout
app.layout = create_layout()


@callback(
    Output('portfolio-graph', 'figure'),
    Output('portfolio-summary', 'children'),
    Input('run-opt-button', 'n_clicks'),
    State('ticker-dropdown', 'value'),
    State('custom-ticker-input', 'value'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date'),
    State('percentage-input', 'value'),
    State('number-of-portfolios', 'value'),
    prevent_initial_call=True  # Prevents firing on page load
)
def run_optimization(n_clicks,dropdown_tickers, custom_tickers_str, start_date, end_date, risk_free_rate, number_of_portfolios):
    
    # --- 1. Input Validation ---
    selected_tickers = dropdown_tickers or []
    if custom_tickers_str:
            # This splits by comma, strips whitespace, converts to upper, and filters empty strings
            custom_list = [ticker.strip().upper() for ticker in custom_tickers_str.split(',') if ticker.strip()]
            selected_tickers.extend(custom_list)
            
    # Remove any duplicates 
    tickers = list(dict.fromkeys(selected_tickers)) # Fast de-duplication

    if not tickers or not start_date or not end_date:
        return go.Figure(), html.P("Please select or enter at least one ticker and a full date range.", style={'color': 'red'})
    # Convert risk-free rate from % to decimal. Default to 0.02 (2%) if empty.
    if risk_free_rate is None:
        rf_rate = 0.02 
    else:
        rf_rate = risk_free_rate / 100.0

    # I'll set a reasonable number here. You can increase this.
    NUM_PORTFOLIOS = number_of_portfolios 

    # --- 2. Run Backend Functions ---
    print("Fetching data...")
    price_data, failed_fetch = fetch_historical_data(tickers, start_date, end_date)
    
    if price_data.empty:
        return go.Figure(), html.P(f"Failed to fetch data for all selected tickers: {', '.join(failed_fetch)}", style={'color': 'red'})

    print("Calculating returns...")
    mean_returns, cov_matrix, dropped_quality = calculate_returns(price_data)

    if mean_returns.empty:
        return go.Figure(), html.P(f"Could not process returns. All tickers may have poor data quality. Dropped: {', '.join(dropped_quality)}", style={'color': 'red'})
    
    # Get the final list of tickers that were successfully processed
    final_tickers = mean_returns.index.tolist()

    print("Running Monte Carlo simulation...")
    mc_results_df = run_monte_carlo_simulation(NUM_PORTFOLIOS, mean_returns, cov_matrix, rf_rate)
    
    print("Calculating optimal portfolios...")
    max_sharpe_port, min_vol_port = calculate_optimal_portfolios(mean_returns, cov_matrix, rf_rate)
    
    print("Calculating efficient frontier...")
    frontier_vol, frontier_ret = calculate_efficient_frontier(mean_returns, cov_matrix)

    # --- 3. Create Figure ---
    fig = go.Figure()

    # Add Monte Carlo scatter plot
    fig.add_trace(go.Scatter(
        x=mc_results_df['volatility'],
        y=mc_results_df['return'],
        mode='markers',
        marker=dict(
            color=mc_results_df['sharpe'],
            colorscale='Viridis',
            showscale=True,
            size=5,
            opacity=0.5,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Simulated Portfolios'
    ))

    # Add Efficient Frontier line
    fig.add_trace(go.Scatter(
        x=frontier_vol,
        y=frontier_ret,
        mode='lines',
        line=dict(color='white', width=3, dash='dash'),
        name='Efficient Frontier'
    ))

    # Add Max Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_port['volatility']],
        y=[max_sharpe_port['return']],
        mode='markers',
        marker=dict(color='#FF0000', size=14, symbol='star', line=dict(width=1, color='black')),
        name=f"Max Sharpe (SR: {max_sharpe_port['sharpe_ratio']:.2f})"
    ))

    # Add Min Volatility portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol_port['volatility']],
        y=[min_vol_port['return']],
        mode='markers',
        marker=dict(color='#00FF00', size=14, symbol='star', line=dict(width=1, color='black')),
        name=f"Min Volatility (Vol: {min_vol_port['volatility']:.2%})"
    ))
    
    fig.update_layout(
        title='Portfolio Optimization (Monte Carlo)',
        xaxis_title='Annualized Volatility (Std. Dev)',
        yaxis_title='Annualized Return',
        yaxis_tickformat='.1%',
        xaxis_tickformat='.1%',
        legend_title_text='Portfolios',
        template='plotly_dark' # Match the CYBORG theme
    )

    # --- 4. Create Summary ---
    
    # Helper function to create a weights table
    def create_weights_table(portfolio_dict, tickers):
        weights = portfolio_dict['weights']
        # Filter weights and tickers for non-zero values
        data = [
            {"Ticker": ticker, "Weight": f"{weight*100:.2f}%"}
            for ticker, weight in zip(tickers, weights) if weight > 0.0001 # Show weights > 0.01%
        ]
        df = pd.DataFrame(data).sort_values(by="Weight", ascending=False)
        return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, color="dark")

    summary = html.Div([
        html.H4("Optimization Results", className="mt-4"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H5("Max Sharpe Portfolio"),
                html.P(f"Return: {max_sharpe_port['return']:.2%}"),
                html.P(f"Volatility: {max_sharpe_port['volatility']:.2%}"),
                html.P(f"Sharpe Ratio: {max_sharpe_port['sharpe_ratio']:.2f}"),
                create_weights_table(max_sharpe_port, final_tickers)
            ], width=6),
            dbc.Col([
                html.H5("Min Volatility Portfolio"),
                html.P(f"Return: {min_vol_port['return']:.2%}"),
                html.P(f"Volatility: {min_vol_port['volatility']:.2%}"),
                html.P(f"Sharpe Ratio: {min_vol_port['sharpe_ratio']:.2f}"),
                create_weights_table(min_vol_port, final_tickers)
            ], width=6)
        ]),
        html.Hr(),
        html.P(f"Tickers dropped (failed fetch): {', '.join(failed_fetch) if failed_fetch else 'None'}", style={'fontSize': 'small'}),
        html.P(f"Tickers dropped (poor data quality): {', '.join(dropped_quality) if dropped_quality else 'None'}", style={'fontSize': 'small'}),
    ])

    print("Optimization complete.")
    return fig, summary


if __name__ == '__main__':
    app.run(debug=True)