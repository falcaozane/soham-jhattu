from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PortfolioRequest(BaseModel):
    Symbols: str = Field(..., example="AAPL,AMZN,GOOGL", description="Comma-separated list of stock symbols")
    sim_no: int = Field(1000, example=1000, description="Number of simulations to run")

class OptimizeRequest(BaseModel):
    lifestyle_risk: int = Field(..., example=1, description="Lifestyle risk level (0: low, 1: high, 2: mid)")
    expected_annual_roi: float = Field(..., example=0.15, description="Expected annual return on investment")
    principal_amount: float = Field(..., example=100000, description="Principal amount to invest")

class Weights(BaseModel):
    Weights: Dict[str, float]

class PortfolioResponse(BaseModel):
    weights_df: Dict[str, Weights]
    annual_return: float
    port_volatility: float
    sharpe_ratio: float
    array_of_allocation: List[Dict[str, Any]]

class OptimizeResponse(BaseModel):
    type_of_person: str
    results: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]

class WeightsResponse(BaseModel):
    type_of_person: str
    clusters: List[Dict[str, Any]]
