from fastapi import APIRouter, HTTPException
from models import OptimizeRequest
from utils import calculate_probabilities
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, List

router = APIRouter()

@router.post('/weights', response_model=Dict[str, List[Dict[str, float]]], summary="Get Weights for Clusters", description="Get the weights for different clusters based on lifestyle risk and expected ROI")
async def weights(request: OptimizeRequest):
    lifestyle_risk = request.lifestyle_risk
    expected_annual_roi = request.expected_annual_roi
    principal_amount = request.principal_amount

    info_data = [
        {"Symbol": "AAPL", "Annualized ROI": 17.624598, "Volatility": 0.027930},
        {"Symbol": "AMZN", "Annualized ROI": 30.874074, "Volatility": 0.035528},
        {"Symbol": "ARKK", "Annualized ROI": 7.464424, "Volatility": 0.023803},
        {"Symbol": "BABA", "Annualized ROI": -1.149219, "Volatility": 0.026288},
        {"Symbol": "BTC-USD", "Annualized ROI": 57.557349, "Volatility": 0.036763},
        {"Symbol": "ELF", "Annualized ROI": 21.738374, "Volatility": 0.031783},
        {"Symbol": "ETH-USD", "Annualized ROI": 35.917395, "Volatility": 0.046763},
        {"Symbol": "GC=F", "Annualized ROI": 8.950140, "Volatility": 0.010886},
        {"Symbol": "GOOGL", "Annualized ROI": 22.442833, "Volatility": 0.019344},
        {"Symbol": "GSK", "Annualized ROI": 10.011961, "Volatility": 0.017046},
        {"Symbol": "ITC.NS", "Annualized ROI": 16.263788, "Volatility": 0.020341},
        {"Symbol": "JNJ", "Annualized ROI": 10.925850, "Volatility": 0.014432},
        {"Symbol": "MSFT", "Annualized ROI": 24.020376, "Volatility": 0.021158},
        {"Symbol": "NFLX", "Annualized ROI": 31.412082, "Volatility": 0.035304},
        {"Symbol": "NVDA", "Annualized ROI": 34.711791, "Volatility": 0.037871},
        {"Symbol": "PLD", "Annualized ROI": 5.721513, "Volatility": 0.023028},
        {"Symbol": "QCOM", "Annualized ROI": 18.908380, "Volatility": 0.030630},
        {"Symbol": "SQ", "Annualized ROI": 17.814773, "Volatility": 0.036852},
        {"Symbol": "TCEHY", "Annualized ROI": 17.299087, "Volatility": 0.023695},
        {"Symbol": "TSLA", "Annualized ROI": 37.054781, "Volatility": 0.035839},
        {"Symbol": "XOM", "Annualized ROI": 7.051585, "Volatility": 0.014561}
    ]

    info = pd.DataFrame(info_data)

    model_data = info[['Annualized ROI', 'Volatility']]
    kmeans = KMeans(n_clusters=3, random_state=4)
    kmeans.fit(model_data)
    clusters = kmeans.predict(model_data)
    info['Cluster'] = clusters
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(model_data, clusters)
    centroids = kmeans.cluster_centers_
    if lifestyle_risk == 0:
        expected_volatility = centroids[0][1]
        x = "low risk"
    elif lifestyle_risk == 1:
        expected_volatility = centroids[2][1]
        x = "high risk"
    elif lifestyle_risk == 2:
        expected_volatility = centroids[1][1]
        x = "mid risk"
    else:
        raise HTTPException(status_code=400, detail="Invalid lifestyle risk value")

    model_input = np.array([[expected_annual_roi, expected_volatility]])

    probabilities = calculate_probabilities(model_input, centroids)
    weighted_amounts = principal_amount * probabilities
    weights_df = pd.DataFrame({'Weight': weighted_amounts})

    clusters_data = {
        "Symbols": [
            "ARKK, BABA, GC=F, GSK, JNJ, PLD, XOM",
            "AMZN, BTC-USD, ETH-USD, NFLX, NVDA, TSLA",
            "AAPL, ELF, GOOGL, ITC.NS, MSFT, QCOM, SQ, TCEHY"
        ]
    }

    clusters_df = pd.DataFrame(clusters_data)
    clusters_df['Weights'] = weights_df['Weight']
    return {"type of person": x,"clusters": clusters_df.to_dict(orient='records')}
