from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: float

class PredictResponse(BaseModel):
    predicted_income: float
