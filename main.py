from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('saved_Model/attrition_model.pkl')
scaler = joblib.load('saved_Model/scaler.pkl')
feature_names = joblib.load("saved_Model/feature_names.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmployeeData(BaseModel):
    Age: float
    BusinessTravel: float
    DailyRate: float
    Department: float
    DistanceFromHome: float
    Education: float
    EducationField: float
    EnvironmentSatisfaction: float
    Gender: float
    HourlyRate: float
    JobInvolvement: float
    JobLevel: float
    JobRole: float
    JobSatisfaction: float
    MaritalStatus: float
    MonthlyIncome: float
    MonthlyRate: float
    NumCompaniesWorked: float
    OverTime: float
    PercentSalaryHike: float
    PerformanceRating: float
    RelationshipSatisfaction: float
    StockOptionLevel: float
    TotalWorkingYears: float
    TrainingTimesLastYear: float
    WorkLifeBalance: float
    YearsAtCompany: float
    YearsInCurrentRole: float
    YearsSinceLastPromotion: float
    YearsWithCurrManager: float

@app.post("/predict")
def predict(data: EmployeeData):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = "Attrition" if prediction == 1 else "No Attrition"
    return {"prediction": result}