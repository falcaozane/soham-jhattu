from fastapi import APIRouter

router = APIRouter()

@router.get('/heartbeat')
async def heartbeat():
    return {"status": "alive"}
