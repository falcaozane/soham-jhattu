from fastapi import FastAPI
from endpoints import portfolio, optimize, weights, heartbeat
import threading
import time
import requests

app = FastAPI()

app.include_router(portfolio.router)
app.include_router(optimize.router)
app.include_router(weights.router)
app.include_router(heartbeat.router)

def send_heartbeat():
    while True:
        try:
            requests.get("https://montecarloapi.onrender.com/heartbeat")
        except requests.exceptions.RequestException as e:
            print(f"Heartbeat failed: {e}")
        time.sleep(300)  # Send a heartbeat request every 5 minutes

if __name__ == '__main__':
    # Start the heartbeat thread
    heartbeat_thread = threading.Thread(target=send_heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()

    # Run the FastAPI application
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
