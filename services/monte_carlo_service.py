import numpy as np
import pandas as pd

def monteCarlo(df, assets, num_of_portfolios=100):
    # Drop columns with all NaN values
    df = df.dropna(axis=1, how='all')
    if df.empty:
        raise ValueError("No valid data available for Monte Carlo simulation")

    log_returns = np.log(1 + df.pct_change(fill_method=None))

    all_weights = np.zeros((num_of_portfolios, len(df.columns)))

    ret_arr = np.zeros(num_of_portfolios)
    vol_arr = np.zeros(num_of_portfolios)
    sharpe_arr = np.zeros(num_of_portfolios)

    for i in range(num_of_portfolios):
        monte_weights = np.random.random(len(df.columns))
        monte_weights = monte_weights / np.sum(monte_weights)

        all_weights[i, :] = monte_weights

        portfolio_return = np.sum((log_returns.mean() * monte_weights) * 252)
        portfolio_std_dev = np.sqrt(np.dot(monte_weights.T, np.dot(log_returns.cov() * 252, monte_weights)))

        ret_arr[i] = portfolio_return * 100
        vol_arr[i] = portfolio_std_dev
        sharpe_arr[i] = portfolio_return / portfolio_std_dev

    simulations_data = [ret_arr, vol_arr, sharpe_arr, all_weights]
    simulations_df = pd.DataFrame(data=simulations_data).T
    simulations_df.columns = ["Returns", "Volatility", "Sharpe Ratio", "Weights"]

    simulations_df = simulations_df.infer_objects()

    return simulations_df
