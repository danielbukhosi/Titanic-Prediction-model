from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("titanic_model.joblib")

app = FastAPI()
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    fare: float
    SibSp: int
    Parch: int
    familysize: int

@app.post("/predict")
def predict_survival(data: PassengerData):
    sex_mapping = {"male": 0, "female": 1}
    sex_encoded = sex_mapping.get(data.Sex.lower(), -1)
    FamilyIsize = data.SibSp + data.Parch + 1
    if sex_encoded == -1:
        return {"error": "Invalid value for Sex. Use 'male' or 'female'."}  
    
    features = np.array([[data.Pclass, sex_encoded, data.Age, data.fare, data.SibSp, data.Parch, FamilyIsize]])
    prediction = model.predict(features)  
    survival = bool(prediction[0])
    return {"survival": survival}