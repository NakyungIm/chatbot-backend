from typing import Dict, Any
from pydantic import BaseModel

class Intent(BaseModel):
    displayName: str

class QueryResult(BaseModel):
    intent: Intent
    parameters: Dict[str, Any]

class DialogflowRequest(BaseModel):
    queryResult: QueryResult 