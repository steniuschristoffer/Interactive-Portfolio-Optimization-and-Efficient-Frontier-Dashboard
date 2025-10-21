# src/components/layout.py

from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from datetime import date, datetime, timedelta
from .tickers import tx

tickerDropdown = dcc.Dropdown(
    options=tx(),
    multi=True,
    id='ticker-dropdown', # Renamed id for clarity
    closeOnSelect=False
)

# New component for free-text entry
customTickerInput = dcc.Input(
    id='custom-ticker-input',
    type='text',
    placeholder='Enter custom tickers here separated with comma (e.g., GOOG, TSLA) ',
    style={'width': '100%', 'marginTop': '10px'} # Add some space
)

def create_layout():
    """Creates the layout for the Dash application."""
    layout = dbc.Container(
        [
            html.H2("Portfolio Optimization Dashboard (Monte Carlo)", style={"paddingTop": "50px"}),
            html.H5("Select your portfolio", style={"paddingTop": "20px"}),
            tickerDropdown,
            customTickerInput,
            
            dbc.Row([
                dbc.Col([
                    html.H5("Number of portfolios", style={"paddingTop": "20px"}),
                    dbc.InputGroup(
                        [
                            dbc.Input(id="number-of-portfolios", type="number", min=0, step=1, debounce=True, max=250000, style={"width": "200px"}, placeholder="e.g., 25000")
                        ],
                        style={"paddingTop": "10px", "width": "290px"}
                    ),
                    html.H5("Risk free rate", style={"paddingTop": "20px"}),
                    dbc.InputGroup(
                        [
                            dbc.Input(id="percentage-input", type="number", min=0, step=0.1, debounce=True, max=100, style={"width": "200px"}, placeholder="e.g., 2.5"),
                            dbc.InputGroupText("%"),  # This adds a '%' sign
                        ],
                        style={"paddingTop": "10px", "width": "290px"}
                    )]),
                
                dbc.Col([
                    html.H5("Optimization interval", style={"paddingTop": "20px"}),
                    html.Div([
                        dcc.DatePickerRange(
                            id='my-date-picker-range',
                            min_date_allowed=date(1995, 8, 5),
                            max_date_allowed=datetime.today(),
                            initial_visible_month=(datetime.now() - timedelta(days=5*365)),
                            start_date=(datetime.now() - timedelta(days=5*365)).date(), # Set default start
                            end_date=datetime.today().date()
                        )
                    ], style={"width": "300px"}),
                    
                    dbc.Button(
                        "Run optimization", 
                        id="run-opt-button", 
                        color="primary", 
                        n_clicks=0,
                        style={"marginTop": "50px", "padding":"20px 80px 20px 80px"}
                    )
                ])
            ]),
            
            # --- NEW ROW FOR OUTPUTS ---
            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        id="loading-output",
                        type="default",
                        children=[
                            dcc.Graph(id="portfolio-graph"),
                            html.Div(id="portfolio-summary")
                        ]
                    ),
                    className="mt-4" # Margin-top 4
                )
            ])
        ],
        style={"padding": "20px", "width": "1200px"},
        fluid=True,
        className="mb-4"
    )
    return layout