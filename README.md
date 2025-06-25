Employee Attrition Predictor – Backend (FastAPI)
This project predicts whether an employee is likely to leave the company using machine learning models trained on HR data.
It exposes a REST API using FastAPI for real-time predictions and is integrated with a React frontend.

📡 Live API Docs: https://employee-attrition-ml-04ho.onrender.com/docs

🔧 Tech Stack
Python

FastAPI

scikit-learn

pandas

seaborn & matplotlib (for visualization)

joblib (for model saving/loading)

Uvicorn (for API serving)

Visual Studio Code

Render (for deployment)

📁 Project Structure
css
Copy
Edit
employee-attrition-ml/
├── archive/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── saved_Model/
│   ├── attrition_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── train_model.py
├── train_model_test.py
├── predict.py
├── main.py
├── requirements.txt
└── README.md
📊 Dataset
Source: IBM HR Analytics Employee Attrition & Performance

Total rows: 1470

Features: 35

Target column: Attrition

📈 Model Performance
Model Used: Random Forest Classifier

Accuracy: ~87%

Logistic Regression also explored for comparison

📨 API Endpoint
POST /predict: Accepts JSON input with employee features and returns prediction (Attrition or No Attrition)

CORS configured to allow React frontend domain

Swagger UI: https://employee-attrition-ml-04ho.onrender.com/docs

🚀 How to Run Locally
bash
Copy
Edit
# 1. Clone this repository
git clone https://github.com/vishaldharwadkar/employee-attrition-ml.git
cd employee-attrition-ml

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (optional)
python train_model.py

# 5. Start the API server
uvicorn main:app --reload

# 6. (Optional) Test predictions manually
python predict.py
🌐 Deployment
Deployed on Render: https://employee-attrition-ml-04ho.onrender.com

Auto-loads saved model and scaler on startup

🔗 Connected Projects
Frontend (React): https://github.com/vishaldharwadkar/employee-attrition-frontend

Live Frontend: https://employee-attrition-vishal.netlify.app

✍️ Author
Vishal Dharwadkar
React & .NET Developer | Machine Learning Enthusiast
GitHub Profile