import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv('archive/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
# intead of columns=[] I can use axis=1 which means columns and axis=0 means rows

le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':  # check if the column is string type
        df[column] = le.fit_transform(df[column])  # convert string to numeric

x = df.drop(columns=['Attrition'],axis=1) #X contains everything except the “Attrition” column — this is the input data the model will learn from.
y = df['Attrition'] #y is just the "Attrition" column — the actual result we want the model to predict.

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train_test_split Splits your data into 4 parts: X_train, X_test, y_train, y_test.
#test_size=0.2	20% of data goes into testing, 80% into training.
#random_state=42	Ensures the split is always the same each time you run the code (for consistency).

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#model = LogisticRegression(max_iter=1000,class_weight='balanced')  # Initialize the model, using Logistic Regression with a maximum of 1000 iterations
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced_subsample',
    random_state=42
)


model.fit(x_train_scaled, y_train)  # Train the model using the training data

y_pred = model.predict(x_test_scaled)  # Make predictions on the test data

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, 'saved_Model/attrition_model.pkl')  # Save the trained model to a file
joblib.dump(scaler, 'saved_Model/scaler.pkl')  # Save the scaler to a file

# print(df.columns)
# print(df.head())
# print("X shape:", x.shape)
# print("y shape:", y.shape)
# print("Training rows:", x_train.shape[0])
# print("Testing rows:", x_test.shape[0])
