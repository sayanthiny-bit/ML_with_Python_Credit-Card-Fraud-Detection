# ML_with_Python_Credit-Card-Fraud-Detection
# 💳 Credit Card Fraud Detection using Logistic Regression

This project builds a binary classification model to detect fraudulent credit card transactions using the **Logistic Regression** algorithm. The model is evaluated based on **accuracy**, **precision**, **recall**, and **F1-score**, providing reliable results despite class imbalance.

---

## 📁 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraudulent Transactions**: ~0.17%
- **Features**: 30 (V1–V28 are anonymized PCA features, plus `Time` and `Amount`)
- **Target**: `Class` (0 = Legitimate, 1 = Fraud)

---

## ⚙️ Technologies Used

- Python 🐍
- Pandas, NumPy
- Scikit-learn (LogisticRegression, train_test_split, classification_report)
- Jupyter Notebook

---

## 🧠 Model Building

python
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, Y_train)
Model Evaluation
✅ Accuracy
text
Copy
Edit
✔️ Accuracy on Training Data: 93.39%
✔️ Accuracy on Test Data: 91.88%
✅ Classification Report (on Test Data)
text
Copy
Edit
              precision    recall  f1-score   support

           0       0.87      0.98      0.92        99
           1       0.98      0.86      0.91        98

    accuracy                           0.92       197
   macro avg       0.93      0.92      0.92       197
weighted avg       0.93      0.92      0.92       197


🔍 Key Insights
High Precision (0.98) for Fraud Class: Very few legitimate transactions were wrongly flagged as fraud.

Strong F1-Score (0.91) for Fraud Detection: Balanced performance between precision and recall.

Slightly Lower Recall (0.86): Indicates a few fraudulent transactions were missed.

Overall, the model performs reliably in identifying frauds with minimal false alarms.

📌 Future Work
Address class imbalance using SMOTE or undersampling

Evaluate with confusion matrix and ROC-AUC

Try ensemble models (Random Forest, XGBoost)

Deploy model using Flask or Streamlit for real-time predictions
