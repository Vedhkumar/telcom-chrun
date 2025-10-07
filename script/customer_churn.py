# Customer Churn Analysis - End-to-End Project

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Step 1: Setup directories
# -------------------------------
if not os.path.exists("reports"):
    os.makedirs("reports")

# -------------------------------
# Step 2: Load dataset
# -------------------------------
dataset = pd.read_csv("data/Telco-Customer-Churn.csv")
print(f"Dataset shape: {dataset.shape}")
print(dataset.head())

# -------------------------------
# Step 3: Data overview
# -------------------------------
print(dataset.info())
print(dataset.describe())
print(dataset.isnull().sum())

# -------------------------------
# Step 4: Data preprocessing
# -------------------------------

# Convert TotalCharges to numeric, handle missing
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)

# Encode categorical variables
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'Churn']

labelencoder = LabelEncoder()
for col in categorical_cols:
    dataset[col] = labelencoder.fit_transform(dataset[col])

# -------------------------------
# Step 5: Exploratory Data Analysis
# -------------------------------

# 5a. Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=dataset, palette="coolwarm", hue=None)
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/churn_distribution.png")
plt.close()

# 5b. Correlation heatmap (numerical features)
plt.figure(figsize=(14,12))
numeric_features = dataset.select_dtypes(include=[np.number])
corr = numeric_features.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 10})
plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/correlation_matrix.png")
plt.close()

# -------------------------------
# Step 6: Feature-target split and scaling
# -------------------------------
X = dataset.drop(['customerID', 'Churn'], axis=1)
y = dataset['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 7: Model training
# -------------------------------
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# -------------------------------
# Step 8: Prediction
# -------------------------------
y_pred = clf.predict(X_test)

# -------------------------------
# Step 9: Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="coolwarm")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.close()

# Feature importance
feature_importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
plt.title("Feature Importances", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("reports/feature_importances.png")
plt.close()

print("End-to-End Churn Project completed. All reports saved in 'reports/' folder.")
