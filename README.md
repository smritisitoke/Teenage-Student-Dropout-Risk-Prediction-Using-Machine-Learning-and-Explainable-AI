# 🎓 Teenage Student Dropout Risk Prediction Using Machine Learning & Explainable AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20%26%20LIME-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**[🚀 Live Demo](https://oknyvgdstn7dqi5cbfnskh.streamlit.app/) · [📓 Google Colab Notebook](https://colab.research.google.com/drive/1O2gihhyMOb2Gk_ol8BXIHCoGBKO0bGdp?usp=sharing) · [📊 Dataset (Kaggle)](https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset)**

</div>

---

## 📌 Overview

Student dropout is a critical challenge facing educational institutions worldwide. Early identification of at-risk students enables timely intervention, improving retention rates and academic outcomes.

This project builds an end-to-end ML pipeline that:
- **Predicts** whether a teenage student is likely to drop out
- **Explains** individual and global predictions using LIME and SHAP (Explainable AI)
- **Deploys** a real-time prediction interface via Streamlit

> **Research Context:** Conducted as part of a Data Science research internship focused on ML and Explainable AI.

---

## 🎯 Objectives

- ✅ Predict student dropout risk from structured educational data
- ✅ Perform thorough Exploratory Data Analysis (EDA)
- ✅ Handle class imbalance using multiple strategies
- ✅ Train and compare multiple ML models
- ✅ Implement Explainable AI using LIME & SHAP
- ✅ Deploy an interactive prediction app on Streamlit

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Total Records** | 649 students |
| **Total Features** | 34 |
| **Target Variable** | `Dropped_Out` (Binary) |
| **Dataset Type** | Structured Educational Dataset |
| **Source** | [Kaggle — Student Dropout Analysis](https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset) |

### Feature Categories

| Category | Examples |
|----------|---------|
| 🧍 Demographic | Age, Gender, Family Size |
| 📚 Academic | Grades, Study Time, Failures |
| 👨‍👩‍👧 Family Background | Family Support, Parent Education |
| 📅 Behavioral | Attendance, Absences, Activities |

---

## ⚙️ Project Workflow

```
Data Collection → Data Cleaning → Outlier Removal → Encoding
       ↓
Exploratory Data Analysis (EDA)
       ↓
Train-Test Split → Class Imbalance Handling
       ↓
Model Training → Model Evaluation
       ↓
Explainability (LIME + SHAP) → Final Insights → Streamlit Deployment
```

---

## 🔍 Exploratory Data Analysis

EDA was performed using count plots, histograms, box plots, correlation heatmaps, and pairplots.

### Key Findings

| Finding | Insight |
|---------|---------|
| 📉 Lower grades | Strongly correlated with dropout |
| 🏫 High absences | Significantly increase dropout risk |
| ❌ Academic failures | Most impactful single predictor |
| 📖 Study time | Higher study time reduces dropout probability |
| 👨‍👩‍👧 Family support | Acts as a protective factor |

---

## ⚖️ Class Imbalance Handling

The dataset was imbalanced. The following techniques were applied and compared:

| Technique | Description |
|-----------|-------------|
| **Random Under Sampling** | Reduce majority class samples |
| **Random Over Sampling** | Duplicate minority class samples |
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **Cost-Sensitive Learning** | Penalize misclassification of minority class |
| **Ensemble Learning** | AdaBoost & XGBoost (built-in imbalance handling) |

---

## 🤖 Models & Performance

| Model | Performance |
|-------|-------------|
| Logistic Regression | Moderate |
| Random Under Sampling + Classifier | Lower |
| Random Over Sampling + Classifier | Good |
| SMOTE + Classifier | Good |
| Cost-Sensitive Learning | Good |
| AdaBoost | Very Good |
| **XGBoost** | ⭐ **Best** |

> **Best Model: XGBoost** — selected for deployment due to its superior performance on imbalanced educational data.

---

## 🧠 Explainable AI (XAI)

Explainability is central to this project. Both local and global explanation methods are implemented.

### 🔵 LIME — Local Interpretable Model-Agnostic Explanations
Explains **individual predictions** by approximating the model locally.

**Key Influential Features (per prediction):**
- Grade_2
- Absences
- Family Support
- Study Time
- Gender

### 🔴 SHAP — SHapley Additive exPlanations
Explains **global feature importance** across the entire dataset.

**Top Global Features:**
1. Final Grade
2. Absences
3. Number of Failures
4. Study Time
5. Family Support

---

## 📁 Repository Structure

```
Teenage-Student-Dropout-Risk-Prediction/
│
├── .devcontainer/                        # Dev Container configuration
├── EDA_OF_DROPOUT_PREDICTION (3).ipynb  # Exploratory Data Analysis notebook
├── Student_dropout_risk_prediction.ipynb # Model training & evaluation notebook
├── app.py                                # Streamlit entry point
├── streamlit_app.py                      # Main Streamlit application
├── requirements.txt                      # Python dependencies
├── student dropout.csv                   # Dataset
├── xg_model_pipe (1).pkl                 # Trained XGBoost pipeline (serialized)
└── README.md
```

---

## 🚀 Live Demo

The model is deployed on **Streamlit Community Cloud**.

👉 **[Try it here: https://oknyvgdstn7dqi5cbfnskh.streamlit.app/](https://oknyvgdstn7dqi5cbfnskh.streamlit.app/)**

Enter a student's demographic, academic, and behavioral details to get:
- ✅ Dropout risk prediction (Yes / No)
- 📊 Probability score
- 🔍 SHAP-based explanation of the prediction

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **ML Libraries** | Scikit-learn, XGBoost, Imbalanced-learn |
| **XAI** | SHAP, LIME |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit, Streamlit Community Cloud |
| **Development** | Google Colab, GitHub |

---

## 💻 Run Locally

```bash
# Clone the repository
git clone https://github.com/smritisitoke/Teenage-Student-Dropout-Risk-Prediction-Using-Machine-Learning-and-Explainable-AI.git
cd Teenage-Student-Dropout-Risk-Prediction-Using-Machine-Learning-and-Explainable-AI

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

---

## 📌 Key Insights

> 📢 These findings can help educational institutions design targeted intervention programs.

1. **Academic performance** is the strongest predictor of dropout risk
2. **Absenteeism** significantly increases the likelihood of dropping out
3. **Students with prior failures** are at the highest risk
4. **Family support** acts as a protective factor, reducing dropout probability
5. **Study time** is positively associated with student retention

---

## 🔮 Future Improvements

- [ ] Integrate Deep Learning models (LSTM for temporal patterns)
- [ ] Build a full-featured web dashboard with admin panel
- [ ] Real-time prediction with student management system integration
- [ ] Expand dataset with more diverse demographic sources
- [ ] Add multilingual support for broader accessibility

---

## 👩‍💻 Author

**Smriti Sitoke**  
B.Tech — Computer Science & Business Systems  
*Data Science Research Intern | ML & Explainable AI Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-smritisitoke-black?style=flat&logo=github)](https://github.com/smritisitoke)

---

## 📎 Links

| Resource | Link |
|----------|------|
| 🚀 Live App | [Streamlit Deployment](https://oknyvgdstn7dqi5cbfnskh.streamlit.app/) |
| 📓 Colab Notebook | [Google Colab](https://colab.research.google.com/drive/1O2gihhyMOb2Gk_ol8BXIHCoGBKO0bGdp?usp=sharing) |
| 📊 Dataset | [Kaggle](https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset) |

---

## ⭐ Star This Repo

If you found this project useful, please consider giving it a ⭐ — it helps others discover it!

---

<div align="center">
<i>Built with ❤️ to help educational institutions identify and support at-risk students.</i>
</div>
