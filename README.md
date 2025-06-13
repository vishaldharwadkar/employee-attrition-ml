Employee Attrition Predictor
This project predicts whether an employee is likely to leave the company using machine learning models trained on HR data.
🔧 Tech Stack
- Python
- scikit-learn
- pandas
- seaborn & matplotlib (for visualization)
- joblib (for model saving/loading)
- Visual Studio Code
📁 Project Structure

employee-attrition-ml/
├── archive/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── saved_Model/
│   └── attrition_model.pkl
├── train_model.py
├── train_model_test.py
├── predict.py
├── .venv/
└── README.md

📊 Dataset

- Source: IBM HR Analytics Employee Attrition & Performance
- Total rows: 1470  
- Features: 35  
- Target column: Attrition

📈 Model Performance

- Model Used: Random Forest Classifier
- Accuracy: ~87%
- Logistic Regression also explored for comparison.

🚀 How to Run

1. Clone this repository
2. Create and activate a virtual environment:
   python -m venv .venv
   .venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Train the model:
   python train_model.py
5. Predict attrition on new data:
   python predict.py

✍️ Author
Vishal Dharwadkar
React & .NET Developer | Machine Learning Enthusiast
