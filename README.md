# 🧬 Parental Genetics & Child Height Prediction

## End-to-End Machine Learning Regression Project

Predicting a child's height using parental genetic and demographic characteristics through a complete machine learning workflow.

---

## 📌 Project Overview

This project develops a regression model capable of predicting a child's height based on parental biological and physical characteristics.

The project demonstrates the complete machine learning lifecycle, including:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Pipeline Construction
* Model Training
* Model Evaluation
* Cross Validation
* Residual Analysis
* Model Persistence

The objective is to identify the factors that influence child height and build a reliable predictive model.

---

## 🎯 Problem Statement

Can a child's height be predicted accurately using parental characteristics such as:

* Father Height
* Mother Height
* Father Age
* Mother Age
* Child Gender
* Eye Colour
* Hair Colour
* Skin Tone
* Blood Group
* Family Disease History

This project investigates the relationship between these variables and child height using supervised machine learning.

---

## 📂 Dataset Information

Dataset Shape:

* 1000 Rows
* 15 Features

Target Variable:

* Predicted_Child_Height_cm

Feature Categories:

### Numerical Features

* Father_Height_cm
* Mother_Height_cm
* Father_Age
* Mother_Age

### Categorical Features

* Child_Gender
* Father_Eye_Color
* Mother_Eye_Color
* Father_Hair_Color
* Mother_Hair_Color
* Skin_Tone
* Blood_Group
* Family_Disease_History

### Data Quality

Missing Values:

* Family_Disease_History (~35%)

Handling Method:

```python
df['Family_Disease_History'] = df['Family_Disease_History'].fillna('Unknown')
```

Columns Removed:

* Family_ID
* Predicted_Child_Blood_Group
* Predicted_Health_Risk

Reason:

These columns either contained no predictive value or introduced target leakage.

---

## 🔍 Exploratory Data Analysis

EDA was performed to understand:

* Missing values
* Feature distributions
* Correlations
* Outliers
* Relationships between variables

### Key Insights

* Child height strongly correlates with parental height.
* Male children tend to be taller on average.
* Family disease history shows minimal influence on height.
* Numerical variables follow approximately normal distributions.

---

## ⚙️ Feature Engineering

Four domain-driven features were created.

### 1. Mid Parental Height

```python
(Father_Height_cm + Mother_Height_cm) / 2
```

Purpose:

Represents the average parental height and serves as a strong genetic predictor.

---

### 2. Parent Height Difference

```python
abs(Father_Height_cm - Mother_Height_cm)
```

Purpose:

Captures variation between parental heights.

---

### 3. Average Parent Age

```python
(Father_Age + Mother_Age) / 2
```

Purpose:

Represents combined parental age influence.

---

### 4. Parent Age Difference

```python
abs(Father_Age - Mother_Age)
```

Purpose:

Captures parental age gap.

---

## 🏗️ Machine Learning Pipeline

The entire workflow was built using Scikit-Learn Pipelines.

Raw Data

↓

ColumnTransformer

├── StandardScaler (Numerical Features)

└── OneHotEncoder (Categorical Features)

↓

Linear Regression

↓

Predicted Child Height

Benefits:

* Prevents data leakage
* Simplifies deployment
* Ensures reproducibility
* Keeps preprocessing and modelling together

---

## 🤖 Models Compared

| Model             | Train R² | Test R² |
| ----------------- | -------- | ------- |
| Linear Regression | 0.79     | 0.79    |
| Ridge Regression  | 0.79     | 0.79    |
| Lasso Regression  | 0.79     | 0.78    |
| Random Forest     | 0.82     | 0.77    |
| Decision Tree     | 0.80     | 0.76    |
| KNN Regressor     | 0.65     | 0.61    |

### Selected Model

Linear Regression

Reasons:

* Best generalization
* Small train-test gap
* No signs of overfitting
* Fast and interpretable
* Consistent cross-validation performance

---

## 📈 Model Performance

| Metric              | Score   |
| ------------------- | ------- |
| R² Score            | 0.78    |
| Cross Validation R² | 0.79    |
| MAE                 | 3.17 cm |
| RMSE                | 4.02 cm |
| MSE                 | 16.16   |

### Interpretation

The model explains approximately 78% of the variance in child height while maintaining stable performance across validation folds.

---

## 🔬 Residual Analysis

Residual analysis was performed to validate model assumptions.

Observations:

* Residuals are centered around zero.
* Residual distribution is approximately normal.
* No clear pattern exists in residual scatter plots.
* Predicted values closely follow actual values.

These observations indicate a well-fitted regression model.

---

## 🏆 Feature Importance Insights

Most Influential Features:

1. Child Gender
2. Mid Parental Height
3. Father Height
4. Mother Height

Least Influential Features:

* Family Disease History
* Blood Group
* Eye Colour
* Hair Colour

The engineered feature `mid_parental_height` emerged as one of the strongest predictors.

---

## 🖼️ Visualizations

The project includes:

* Correlation Heatmap
* Distribution Plots
* Boxplots
* Actual vs Predicted Plot
* Residual Analysis Plot
* Feature Importance Plot

---

## 💾 Model Persistence

The final trained pipeline was saved using Joblib.

```python
import joblib

joblib.dump(best_pipe, "models/child_height_model.pkl")

model = joblib.load("models/child_height_model.pkl")
predictions = model.predict(X_new)
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Jupyter Notebook

---

## 🧠 Skills Demonstrated

* Data Cleaning
* Missing Value Handling
* Exploratory Data Analysis
* Feature Engineering
* Feature Selection
* One-Hot Encoding
* Feature Scaling
* ColumnTransformer
* Pipeline Construction
* Regression Modelling
* Cross Validation
* Residual Analysis
* Model Evaluation
* Model Serialization

---

## 📂 Project Structure

parental-genetics-child-height-prediction/

│

├── data/

│ └── parental_genetics.csv

│

├── notebooks/

│ └── parental_genetic_proj.ipynb

│

├── images/

│ ├── heatmap.png

│ ├── residual_plot.png

│ └── actual_vs_predicted.png

│

├── models/

│ └── child_height_model.pkl

│

├── requirements.txt

├── .gitignore

└── README.md

---

## 🚀 How To Run

```bash
git clone https://github.com/YOUR_USERNAME/parental-genetics-child-height-prediction.git

cd parental-genetics-child-height-prediction

pip install -r requirements.txt

jupyter notebook
```

Open:

```text
notebooks/parental_genetic_proj.ipynb
```

Run all cells.

---

## 📚 Key Learnings

Through this project I learned:

* End-to-End Machine Learning Workflow
* Data Preprocessing Pipelines
* Feature Engineering
* Regression Algorithms
* Model Comparison
* Cross Validation
* Residual Diagnostics
* Model Deployment Preparation

---

## 👨‍💻 Author

Rohan Janardan

Machine Learning | Data Science | Python

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

## 📄 License

This project is licensed under the MIT License.
