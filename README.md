# 📊 Student Exam Score Prediction using Multiple Linear Regression

---

## 📌 Project Overview

This project predicts **Student Exam Scores** using Multiple Linear Regression.

The model learns the relationship between study habits and student performance using multiple features.

---

## 📂 Dataset

Dataset: Student Performance Factors Dataset

Kaggle Link:

https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

---

## 🎯 Objective

To build a regression model that predicts

```text
Exam Score
```

using

```text
Hours Studied
Attendance
Sleep Hours
Previous Scores
Tutoring Sessions
Physical Activity
```

---

## ✅ Project Workflow

```text
Load Dataset
Understand Data
Check Missing Values
Handle Missing Values
Check Duplicates
EDA
Correlation Analysis
Feature Selection
Train Test Split
Train Model
Prediction
Evaluation
Coefficient Analysis
```

---

# 📊 Exploratory Data Analysis

## Dataset Preview

```python
students.head()
```

Displays the first 5 rows of the dataset.

---

## Dataset Information

```python
students.info()
```

Used to check:

- Columns
- Data Types
- Null Values

---

## Statistical Summary

```python
students.describe()
```

Used to understand:

- Mean
- Standard Deviation
- Min
- Max
- Quartiles

---

## Missing Value Analysis

```python
students.isnull().sum()
```

Missing values were identified and handled using mode().

```python
students["Teacher_Quality"] = students["Teacher_Quality"].fillna(
    students["Teacher_Quality"].mode()[0]
)
```

---

## Duplicate Check

```python
students.duplicated().sum()
```

Duplicates removed using:

```python
students.drop_duplicates(inplace=True)
```

---

# 📈 Pairplot Analysis

```python
sns.pairplot(
    students,
    kind="scatter",
    plot_kws={"alpha":0.3},
    diag_kws={"bins":40}
)
```

### Observation

Pairplot helps understand relationships between numerical columns and identify trends.

---

# 📉 Scatter Plot

```python
sns.scatterplot(
    x="Attendance",
    y="Exam_Score",
    hue="Hours_Studied",
    data=students
)
```

### Observation

Attendance and Exam Score show a positive correlation.

Students with higher attendance tend to achieve better exam scores.

Hours Studied also appears to have a positive impact on performance.

---

# 🔥 Correlation Analysis

```python
students.corr(numeric_only=True)
```

Correlation values:

```text
+1 = Strong Positive Correlation
 0 = No Correlation
-1 = Strong Negative Correlation
```

---

# 🌡️ Heatmap

```python
sns.heatmap(
    students.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
```

### Observation

- Hours Studied positively affects Exam Score.
- Attendance positively affects Exam Score.
- Tutoring Sessions positively affects Exam Score.
- Sleep Hours has very little impact.

---

# 🤖 Model Building

## Feature Selection

```python
X = students[
[
"Hours_Studied",
"Attendance",
"Sleep_Hours",
"Previous_Scores",
"Tutoring_Sessions",
"Physical_Activity"
]
]

y = students["Exam_Score"]
```

---

## Train Test Split

```python
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

---

## Create Model

```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
```

---

## Train Model

```python
lm.fit(X_train,y_train)
```

### Why fit()?

```text
Learns the relationship between input features and target variable.
```

---

# 🔮 Prediction

```python
prediction = lm.predict(X_test)
```

### Why predict()?

```text
Predicts Exam Scores for unseen test data.
```

---

# 📊 Actual vs Predicted

```python
plt.scatter(y_test,prediction)

plt.xlabel("Exam Score")
plt.ylabel("Prediction")
plt.title("Exam Score vs Prediction")
```

### Observation

The points follow a positive trend, showing that predictions are reasonably close to actual values.

---

# 📈 Model Evaluation

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print(mean_absolute_error(y_test,prediction))
print(mean_squared_error(y_test,prediction))
print(r2_score(y_test,prediction))
```

| Metric | Value |
|----------|------:|
| MAE | 1.27 |
| MSE | 4.90 |
| R² Score | 0.64 |

---

# ✅ Result Interpretation

### MAE

```text
1.27
```

Average prediction error is only 1.27 marks.

---

### MSE

```text
4.90
```

Prediction error remains low.

---

### R² Score

```text
0.64
```

The model explains approximately 64% of the variation in Exam Scores.

---

# 📊 Coefficients

```python
coef = pd.DataFrame(
    lm.coef_,
    X.columns,
    columns=["Coefficient"]
)

coef
```

| Feature | Coefficient |
|----------|-----------:|
| Hours Studied | 0.288 |
| Attendance | 0.199 |
| Sleep Hours | -0.034 |
| Previous Scores | 0.049 |
| Tutoring Sessions | 0.514 |
| Physical Activity | 0.134 |

---

# 📌 Coefficient Meaning

- Tutoring Sessions has the highest impact.
- Hours Studied positively affects Exam Score.
- Attendance positively affects Exam Score.
- Sleep Hours has a very small negative effect.
- Physical Activity has a small positive impact.

---

# 📌 Final Conclusion

The Multiple Linear Regression model performs reasonably well for predicting student exam scores.

Main findings:

- Tutoring Sessions strongly influence Exam Score.
- Hours Studied positively impacts performance.
- Attendance contributes to better scores.
- The model achieved an R² Score of 0.64.

---

# 🚀 Future Improvements

- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Feature Engineering
- Hyperparameter Tuning

---

# 👨‍💻 Author

**Arulprasath D**

Machine Learning | Data Science | Python

⭐ Star this repository if you found it useful.
