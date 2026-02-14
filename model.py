# =========================================
# 🔹 IMPORT LIBRARIES
# =========================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# =========================================
# 🔹 CREATE FOLDER FOR EDA PLOTS
# =========================================
os.makedirs("eda_plots", exist_ok=True)

# =========================================
# 🔹 LOAD DATA
# =========================================
df = pd.read_csv("load_data.csv")

# =========================================
# 🔹 DATETIME FEATURE ENGINEERING
# =========================================
df["Date_Time"] = pd.to_datetime(df["Date_Time"], dayfirst=True)

df["hour"] = df["Date_Time"].dt.hour
df["day"] = df["Date_Time"].dt.day
df["month"] = df["Date_Time"].dt.month
df["hour_from_nsm"] = df["NSM"] // 3600

df = df.drop(columns=["Date_Time"])

# =========================================
# 🔹 HANDLE MISSING VALUES
# =========================================
df = df.fillna(df.mean(numeric_only=True))

# =========================================
# 🔹 EDA SECTION
# =========================================
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 🔹 Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Load_Type", data=df)
plt.title("Load Type Distribution")
plt.savefig("eda_plots/load_type_distribution.png")
plt.show()
plt.close()

# 🔹 Numeric Feature Distribution
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols].hist(figsize=(12,10))
plt.suptitle("Numeric Feature Distributions")
plt.savefig("eda_plots/numeric_distributions.png")
plt.show()
plt.close()

# 🔹 Correlation Heatmap (NUMERIC ONLY)
plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("eda_plots/correlation_heatmap.png")
plt.show()
plt.close()

# 🔹 Usage vs Load Type
plt.figure(figsize=(6,4))
sns.boxplot(x="Load_Type", y="Usage_kWh", data=df)
plt.title("Usage_kWh vs Load Type")
plt.savefig("eda_plots/usage_vs_loadtype.png")
plt.show()
plt.close()

# =========================================
# 🔹 ENCODE TARGET
# =========================================
le = LabelEncoder()
df["Load_Type"] = le.fit_transform(df["Load_Type"])

# =========================================
# 🔹 TRAIN TEST SPLIT (LAST MONTH TEST)
# =========================================
train_df = df[df["month"] != df["month"].max()]
test_df  = df[df["month"] == df["month"].max()]

X_train = train_df.drop("Load_Type", axis=1)
y_train = train_df["Load_Type"]

X_test = test_df.drop("Load_Type", axis=1)
y_test = test_df["Load_Type"]

# =========================================
# 🔹 SCALING (for LR & SVM)
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================================
# 🔹 DEFINE MODELS
# =========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    ),
    "SVM": SVC(kernel="rbf")
}

# =========================================
# 🔹 TRAIN + EVALUATE
# =========================================
results = []

for name, model in models.items():

    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n🔹 {name}")
    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    results.append([name, acc, precision, recall, f1])

# =========================================
# 🔹 RESULT COMPARISON TABLE
# =========================================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

results_df = results_df.sort_values(by="F1 Score", ascending=False)

print("\n📊 Model Comparison:")
print(results_df)

# =========================================
# 🔹 BEST MODEL
# =========================================
best_model = results_df.iloc[0]["Model"]
print(f"\n🏆 Best Model: {best_model}")
