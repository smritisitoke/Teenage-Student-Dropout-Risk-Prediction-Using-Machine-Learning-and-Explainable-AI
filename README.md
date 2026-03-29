# Teenage-Student-Dropout-Risk-Prediction-Using-Machine-Learning-and-Explainable-AI
Project Overview
Student dropout is a major concern for educational institutions. Early identification of at-risk students helps institutions take preventive actions and improve student retention.
This project aims to predict student dropout risk using Machine Learning and provide interpretable insights using Explainable AI techniques (LIME & SHAP).
The model analyzes demographic, academic, and behavioral features to predict whether a student is likely to drop out.

🎯 Project Objectives
Predict student dropout risk using Machine Learning
Perform Exploratory Data Analysis (EDA)
Handle class imbalance using multiple techniques
Compare multiple models
Implement Explainable AI (LIME & SHAP)
Identify key factors affecting student dropout

📊 Dataset Information
Total Records: 649
Total Features: 34
Target Variable: Dropped_Out
Dataset Type: Structured Educational Dataset
Features Include
Demographic Information
Academic Performance
Family Background
Behavioral Attributes
Attendance & Grades

⚙️ Project Workflow
Data Collection
↓
Data Cleaning
↓
Outlier Removal
↓
Encoding
↓
Exploratory Data Analysis (EDA)
↓
Train-Test Split
↓
Class Imbalance Handling
↓
Model Training
↓
Model Evaluation
↓
Explainability (LIME + SHAP)
↓
Final Insights

🔍 Exploratory Data Analysis (EDA)
EDA was performed using:
Count Plots
Histograms
Box Plots
Correlation Heatmap
Pairplots
Key Findings
Students with lower grades more likely to drop out
Higher absences increase dropout risk
Academic failures strongly impact dropout
Study time influences student retention

⚖️ Class Imbalance Handling Techniques
The dataset was imbalanced, so multiple techniques were applied:
Random Under Sampling
Random Over Sampling
SMOTE
Cost Sensitive Learning
Ensemble Learning
AdaBoost
XGBoost

🤖 Models Used
Logistic Regression
Random Forest
AdaBoost
XGBoost (Best Performing Model)

📈 Model Performance Comparison
Model	Performance
Logistic Regression	Moderate
Random Under Sampling	Lower
Random Over Sampling	Good
SMOTE	Good
Ensemble Learning	Very Good
Cost Sensitive Learning	Good
AdaBoost	Very Good
XGBoost	Best

🧠 Explainable AI
LIME
LIME was used to explain individual predictions.
Key Influential Features:
Grade_2
Absences
Family Support
Study Time
Gender
SHAP
SHAP was used to explain global feature importance.
Top Features:
Final Grade
Absences
Failures
Study Time
Family Support

🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Imbalanced-learn
XGBoost
Matplotlib
Seaborn
LIME
SHAP

💻 Platform Used
Google Colab
GitHub

📁 Project Structure
Student-Dropout-Prediction
│
├── Data
├── EDA Notebook
├── Model Notebook
├── LIME & SHAP Notebook
├── Final Report
├── Presentation
└── README.md

📌 Key Insights
Academic performance is the strongest predictor
Absences significantly impact dropout
Students with failures are at higher risk
Family support reduces dropout probability

🚀 Future Improvements
Use Deep Learning models
Build Web Dashboard
Real-time Prediction System
Deploy Model

📎 Project Links
Google Colab Notebook:
https://colab.research.google.com/drive/1O2gihhyMOb2Gk_ol8BXIHCoGBKO0bGdp?usp=sharing
Dataset Source:
https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset

⭐ Conclusion
This project successfully predicts student dropout risk using Machine Learning and Explainable AI. The model achieved strong performance and provided interpretable insights that can help educational institutions identify at-risk students and improve retention
