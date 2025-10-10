Data Source & Attribution:
Data used in this project is sourced from SMARD (Bundesnetzagentur) and is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

Disclaimer:
This analysis is for educational and demonstrative purposes only. While the data is publicly available, its accuracy or completeness is not guaranteed by SMARD or Bundesnetzagentur. Any insights, predictions, or visualizations are the result of independent analysis and should not be used for commercial or operational decision-making without proper verification.

# Comparison-of-Regression-Models-for-Electricity-Consumption-Prediction
This project forecasts electricity consumption using machine learning models trained on historical energy generation data. It employs Random Forest, Linear Regression, and K-Nearest Neighbors to predict daily grid load. The code includes data preprocessing, model training, evaluation, and visualization of result
Here’s a polished **GitHub README.md** for your notebook, summarizing the workflow, methodology, and findings in a clear and professional format 👇

---

# ⚡ Electricity Grid Load Prediction using Machine Learning

This project applies **Machine Learning regression models** to predict **electric grid load (in MWh)** based on renewable and conventional energy sources data.
The dataset includes multiple power generation sources such as wind, solar, biomass, and fossil fuels over several years.

---

## 📘 Project Overview

Accurate grid load forecasting is essential for managing electricity supply and demand.
This notebook compares the performance of **Random Forest**, **Linear Regression**, and **K-Nearest Neighbors (KNN)** models to predict the variable `gridload_`.

---

## 📂 Dataset

**Source: Data obtained from SMARD.de
 — the official electricity market data platform provided by the German Federal Network Agency (Bundesnetzagentur).**

**File used:** `ActualCons.csv`

**Shape:** 1827 rows × 18 columns

**Features include:**

* Start and End Dates
* Renewable energy contributions (`Wind_onshore_`, `Photovoltaics_`, `Biomass_`, etc.)
* Conventional energy sources (`Lignite_`, `Fossil_gas_`, `Nuclear_`, etc.)
* Target variable: **`gridload_`**

---

## 🧠 Machine Learning Pipeline

### 1. **Data Preparation**

```python
X = df.drop(["Start_date","End_date","Gridload_hydrops","Residualload_","gridload_"], axis=1)
y = df["gridload_"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. **Models Implemented**

| Model                     | Description                     | Library                |
| ------------------------- | ------------------------------- | ---------------------- |
| **RandomForestRegressor** | Ensemble-based non-linear model | `sklearn.ensemble`     |
| **LinearRegression**      | Baseline linear model           | `sklearn.linear_model` |
| **KNeighborsRegressor**   | Distance-based model (K=3)      | `sklearn.neighbors`    |

---

## ⚙️ Model Training & Evaluation

Each model was trained and evaluated using **R² score** and **Mean Squared Error (MSE)**.

### 🔹 Random Forest

```python
RandomForestRegressor().fit(X_train, y_train)
MSE = 4.44e+09
```

### 🔹 Linear Regression

```python
LinearRegression().fit(X_train, y_train)
MSE = 3.31e+09
```

### 🔹 K-Nearest Neighbors

```python
KNeighborsRegressor(n_neighbors=3).fit(X_train_scaled, y_train)
MSE = 5.60e+09
```

---

## 📊 Model Comparison

| Model             |      R² Score (%)     | Mean Squared Error |
| :---------------- | :-------------------: | :----------------: |
| Random Forest     | ≈ *varies by dataset* |    **4.44×10⁹**    |
| Linear Regression | ≈ *varies by dataset* |    **3.31×10⁹**    |
| KNN               | ≈ *varies by dataset* |    **5.60×10⁹**    |

> 💡 *Linear Regression achieved the lowest MSE, indicating the best overall performance for this dataset.*

---

## 📈 Visualization

### 1. **Predicted vs True Values**

A scatter plot comparing predicted and actual `gridload_` values:

```python
plt.scatter(y_pred, y_test, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Comparison of Predicted and True Values of Grid Load (MWh)")
```

### 2. **Model Performance Bar Chart**

```python
labels = ["RandomForest","LinearRegression","KNeighbors"]
values = [scoreRF*100, scoreLR*100, scoreKnn*100]
plt.bar(labels, values, color=['red','blue','green'])
```

---

## 🧾 Requirements

To run this project, install the following dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/GridLoadPrediction.git
   cd GridLoadPrediction
   ```

2. Place `ActualCons.csv` in the same directory.

3. Run the Jupyter notebook or Python script:

   ```bash
   jupyter notebook GridLoadPrediction.ipynb
   ```

4. Review model results and performance visualizations.

---

## 📚 References

* [Scikit-learn: RandomForestRegressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [Scikit-learn: LinearRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [Scikit-learn: KNeighborsRegressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)

---

## 🧩 Future Work

* Incorporate **time-series modeling (LSTM, Prophet)** for temporal forecasting.
* Apply **hyperparameter tuning (GridSearchCV)** to optimize Random Forest and KNN.
* Add **feature importance analysis** to interpret energy source influence.

---

## ✨ Author

**Aditya Kulkarni**
📧 *kulkarniaditya1026@gmail.com*
🌐 [GitHub Profile]([https://github.com/KulkarniA26])

---

