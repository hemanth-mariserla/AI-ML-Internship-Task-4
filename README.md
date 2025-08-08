# AI-ML-Internship-Task-4
# Breast Cancer Classification using Logistic Regression

## Task Overview

This project is submitted for the AI & ML Internship task – *Classification with Logistic Regression*. The objective is to build a **binary classifier** using **Logistic Regression** on a real-world medical dataset and evaluate its performance using standard classification metrics.

---

## 📊 Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Target Variable**: `diagnosis`  
  - `M` = Malignant (1)  
  - `B` = Benign (0)

---

## Objective

- Perform binary classification using **Logistic Regression**
- Evaluate the model using:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC-AUC Curve
- Understand the effect of thresholds
- Visualize the **Sigmoid Function**

---

## Tools and Libraries

- Python
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## Steps Performed

1. **Loaded and cleaned the dataset**
2. **Mapped categorical values to numerical (M → 1, B → 0)**
3. **Standardized features using StandardScaler**
4. **Split dataset into training and testing sets**
5. **Trained Logistic Regression model**
6. **Made predictions and evaluated using metrics**
7. **Plotted ROC Curve and Sigmoid function**

---

## Results

- **Accuracy**: ~96%
- **ROC-AUC Score**: ~0.99
- **Confusion Matrix**: True Positives and True Negatives well separated
- **ROC Curve**: Close to the top-left (ideal classifier)

---

## Evaluation Metrics

              precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

ROC-AUC Score: 1.00

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/b4da4199-b55a-44b8-a2f0-851920efd08a" />
<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/82307083-7017-4cca-83b6-5c2f824677b3" />
<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/02061107-5103-4eec-b580-f6c7b881a792" />


