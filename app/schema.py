from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: float
    instrument: str
    lesson_price: int

class PredictResponse(BaseModel):
    predicted_income: float
