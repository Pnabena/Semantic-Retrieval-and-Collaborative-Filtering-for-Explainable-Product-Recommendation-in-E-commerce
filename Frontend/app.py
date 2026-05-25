from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


def clean_for_json(obj):
    """
    Converts NumPy/Pandas objects into JSON-safe Python types
    and removes heavy/internal fields that the frontend does not need.
    """

    remove_keys = {
        "embedding",
        "user_factors",
        "item_factors",
        "product_text",
        "description_text",
        "features_text"
    }

    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if key in remove_keys:
                continue
            cleaned[key] = clean_for_json(value)
        return cleaned

    if isinstance(obj, list):
        return [clean_for_json(item) for item in obj]

    if isinstance(obj, tuple):
        return [clean_for_json(item) for item in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj
    
from dynamic_search_engine import DynamicSearchEngine, DEFAULT_USER_ID

app = FastAPI(
    title="Explainable Hybrid Recommender API",
    description="Dynamic product search using semantic retrieval, ALS personalization, and explanation generation.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DynamicSearchEngine()


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = DEFAULT_USER_ID
    top_k: Optional[int] = 10
    candidate_k: Optional[int] = 100


@app.get("/")
def root():
    return {
        "message": "Explainable Hybrid Recommender API is running.",
        "endpoints": ["/search", "/health"]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Search engine loaded successfully."
    }


@app.post("/search")
def search_products(request: SearchRequest):
    response = engine.search(
        query=request.query,
        user_id=request.user_id or DEFAULT_USER_ID,
        top_k=request.top_k,
        candidate_k=request.candidate_k
    )

    cleaned_response = clean_for_json(response)

    return JSONResponse(content=cleaned_response)
