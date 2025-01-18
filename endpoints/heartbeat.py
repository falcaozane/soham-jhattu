from fastapi import APIRouter

router = APIRouter()

@router.get('/heartbeat', summary="Heartbeat Endpoint", description="Check if the service is alive")
async def heartbeat():
    return {"status": "alive"}
