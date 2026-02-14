## 🔌 Load Type Prediction using Machine Learning

### 📌 Project Overview

This project focuses on building a machine learning classification model to predict the **Load Type** of a power system based on historical electrical and temporal data.

The load is classified into three categories:

* Light Load
* Medium Load
* Maximum Load

The objective is to design a complete ML pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and temporal validation.

---

## 📂 Dataset Description

The dataset contains the following features:

| Feature                              | Description                           |
| ------------------------------------ | ------------------------------------- |
| Date_Time                            | Timestamp of measurement              |
| Usage_kWh                            | Energy consumption                    |
| Lagging_Current_Reactive_Power_kVArh | Reactive power (lagging)              |
| Leading_Current_Reactive_Power_kVArh | Reactive power (leading)              |
| CO2                                  | CO2 emission level                    |
| Lagging_Current_Power_Factor         | Power factor (lagging)                |
| Leading_Current_Power_Factor         | Power factor (leading)                |
| NSM                                  | Number of seconds from midnight       |
| Load_Type                            | Target label (Light, Medium, Maximum) |

---

## ⚙️ ML Pipeline

### 🔹 1. Data Preprocessing

* Converted `Date_Time` to datetime format
* Handled missing values using mean imputation
* Dropped unused columns
* Encoded target labels using `LabelEncoder`

---

### 🔹 2. Feature Engineering

New time-based features were created:

* Hour
* Day
* Month
* Hour_from_NSM

These features helped capture temporal load patterns.

---

### 🔹 3. Exploratory Data Analysis (EDA)

The following analyses were performed:

* Load type distribution
* Numeric feature histograms
* Correlation heatmap (numeric features only)
* Usage_kWh vs Load_Type boxplot

All plots are saved in the `eda_plots/` folder.

---

### 🔹 4. Train-Test Strategy

A **temporal validation approach** was used:

* Last month → Test set
* Remaining data → Training set

This ensures the model is evaluated on unseen future data.

---

### 🔹 5. Models Implemented

The following classification models were trained and compared:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* XGBoost

Standard scaling was applied for Logistic Regression and SVM.

---

## 📊 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision (Weighted)
* Recall (Weighted)
* F1-Score (Weighted)

---

## 🏆 Model Comparison

| Model               | Accuracy       | Precision      | Recall         | F1 Score       |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| XGBoost             | **0.94** | **0.95** | **0.94** | **0.94** |
| Random Forest       | 0.92           | 0.93           | 0.92           | 0.92           |
| SVM                 | 0.69           | 0.69           | 0.69           | 0.69           |
| Logistic Regression | 0.59           | 0.68           | 0.59           | 0.63           |

---

## ✅ Final Model

**XGBoost** achieved the highest F1-score and overall performance and was selected as the final model.

Reasons:

* Handles tabular data effectively
* Captures non-linear relationships
* No need for feature scaling
* Best generalization on temporal test data

---

## 📁 Project Structure

<pre class="overflow-visible! px-0!" data-start="3171" data-end="3381"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>ASSIGN_2/
│── load_data.csv
│── model.py
│── README.md
│
└── eda_plots/
    ├── load_type_distribution.png
    ├── numeric_distributions.png
    ├── correlation_heatmap.png
    ├── usage_vs_loadtype.png
</span></span></code></div></div></pre>

---

## ▶️ How to Run

### 1️⃣ Install dependencies

<pre class="overflow-visible! px-0!" data-start="3435" data-end="3511"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install pandas numpy matplotlib seaborn scikit-learn xgboost
</span></span></code></div></div></pre>

### 2️⃣ Run the model

<pre class="overflow-visible! px-0!" data-start="3535" data-end="3562"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python model.py
</span></span></code></div></div></pre>

---

## 📌 Key Observations

* Light Load instances are more frequent → class imbalance present
* Usage_kWh strongly influences Load_Type
* Power factor and reactive power show meaningful correlation with load
* Time-based features improve model performance

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Handling class imbalance using SMOTE
* Cross-validation for robustness
* Deployment as a real-time prediction API
