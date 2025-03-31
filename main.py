from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.webhook import router as webhook_router

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(webhook_router)

@app.get("/")
def read_root():
    return {"message": "Netflix Recommender API is running"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
