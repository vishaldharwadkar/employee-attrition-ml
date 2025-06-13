import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = joblib.load('saved_Model/attrition_model.pkl') 
scaler = joblib.load('saved_Model/scaler.pkl')  

df = pd.read_csv('archive/WA_Fn-UseC_-HR-Employee-Attrition.csv')

lb =LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = lb.fit_transform(df[col])

x= df.drop(columns=['Attrition'], axis=1)
y = df['Attrition']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train_scaled = scaler.transform(x_test)
y_pred = model.predict(x_train_scaled)

print("Predictions on test data:", accuracy_score(y_test, y_pred))