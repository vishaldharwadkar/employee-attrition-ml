import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('archive/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
# intead of columns=[] I can use axis=1 which means columns and axis=0 means rows

print(df.columns)