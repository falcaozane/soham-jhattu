from pydantic import BaseModel

class PortfolioRequest(BaseModel):
    Symbols: str
    sim_no: int = 1000

class OptimizeRequest(BaseModel):
    lifestyle_risk: int
    expected_annual_roi: float
    principal_amount: float
