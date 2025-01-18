import numpy as np
import pandas as pd
from services.monte_carlo_service import monteCarlo

def optimize_portfolio(df, assets):
    num_of_portfolios = 6000
    sim_df = monteCarlo(df, assets, num_of_portfolios)
    sim_df['Volatility'] = sim_df['Volatility'].round(2)
    idx = sim_df.groupby('Volatility')['Returns'].idxmax()
    max_df = sim_df.loc[idx].reset_index(drop=True)
    max_df = max_df.sort_values(by='Volatility').reset_index(drop=True)
    max_df['Weights'] = max_df['Weights'].apply(lambda x: {col: weight for col, weight in zip(df.columns, x)})
    max_df = max_df.to_dict(orient='records')

    # Selecting the portfolio with the highest Sharpe Ratio
    max_returns = sim_df.loc[sim_df['Sharpe Ratio'].idxmax()]
    optimal_weights = max_returns['Weights']

    # Expected annual return, volatility, and Sharpe ratio
    weights_df = pd.DataFrame(optimal_weights, columns=['Weights'])
    weights_df.index = assets  # Assign tickers as index

    return weights_df, max_returns['Returns'], max_returns['Volatility']*100, max_returns['Sharpe Ratio'], max_df
