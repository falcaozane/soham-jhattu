from fastapi import APIRouter, HTTPException
from models import PortfolioRequest
from services.data_service import get_data
from services.monte_carlo_service import monteCarlo
from utils import remove_spaces
from typing import Dict, List

router = APIRouter()

@router.post('/portfolio', response_model=Dict[str, List[Dict[str, float]]], summary="Optimize Portfolio", description="Optimize the portfolio based on Monte Carlo simulations")
async def optimise_port(request: PortfolioRequest):
    ticker = remove_spaces(request.Symbols)
    assets = ticker.split(',')
    df = get_data(assets)
    sim_no = request.sim_no

    # Drop columns with all NaN values
    df = df.dropna(axis=1, how='all')
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data available for the provided tickers")

    # Monte Carlo Simulation
    num_of_portfolios = sim_no
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

    # Creating DataFrame for weights
    weights_df = pd.DataFrame(optimal_weights, index=df.columns, columns=['Weights'])
    weights_dict = weights_df.to_dict()

    return {
        "weights_df": weights_dict,
        "annual_return": max_returns['Returns'],
        "port_volatility": max_returns['Volatility']*100,
        "sharpe_ratio": max_returns['Sharpe Ratio'],
        "array_of_allocation": max_df
    }
