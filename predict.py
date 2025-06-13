import joblib
import pandas as pd
import numpy as np

model = joblib.load('saved_Model/attrition_model.pkl')
scaler = joblib.load('saved_Model/scaler.pkl')

input_data = {
    'Age': 34,
    'BusinessTravel': 2,
    'DailyRate': 800,
    'Department': 1,
    'DistanceFromHome': 10,
    'Education': 3,
    'EducationField': 2,
    'EmployeeCount': 1,         # Always 1
    'EmployeeNumber': 205,
    'EnvironmentSatisfaction': 3,
    'Gender': 1,
    'HourlyRate': 72,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobRole': 6,
    'JobSatisfaction': 4,
    'MaritalStatus': 1,
    'MonthlyIncome': 4500,
    'MonthlyRate': 12000,
    'NumCompaniesWorked': 2,
    'Over18': 1,                # Always 1
    'OverTime': 1,
    'PercentSalaryHike': 13,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StandardHours': 80,        # Always 80
    'StockOptionLevel': 1,
    'TotalWorkingYears': 10,
    'TrainingTimesLastYear': 3,
    'WorkLifeBalance': 3,
    'YearsAtCompany': 6,
    'YearsInCurrentRole': 4,
    'YearsSinceLastPromotion': 2,
    'YearsWithCurrManager': 3
}

input_df = pd.DataFrame([input_data])
input_scaler = scaler.transform(input_df)
predictions = model.predict(input_scaler)

print("Prediction:", "Attrition" if predictions[0] == 1 else "No Attrition")