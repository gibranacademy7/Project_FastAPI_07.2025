from fastapi import FastAPI
from app.model import train_model_from_file, predict_income
from app.schema import PredictRequest, PredictResponse

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI is running!"}

@app.post("/train/fromfile")
def train_from_csv_file():
    result = train_model_from_file("Data/music_students_data.csv")
    return result

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    prediction = predict_income(data.age, data.instrument, data.lesson_price)
    return PredictResponse(predicted_income=prediction)
