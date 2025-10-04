# 🇹🇿 Dar es Salaam Air Quality Forecasting with AutoRegressive Model

## ✍️ Description

This project analyzes and forecasts hourly PM2.5 (particulate matter 2.5) air pollution levels in Dar es Salaam, Tanzania, using time series analysis techniques. The core objective is to build a predictive model based on past PM2.5 readings that outperforms a simple baseline.

The analysis involves connecting to a **MongoDB** database to extract and preprocess the time series data, using Autocorrelation/Partial Autocorrelation functions to guide model selection, and validating the final model's performance using **Walk-Forward Validation**.

### Key Steps:

* **Data Acquisition:** Querying hourly PM2.5 data from a MongoDB collection.
* **Data Preparation:** Resampling to hourly frequency, handling outliers, and localizing time to "Africa/Dar\_es\_Salaam".
* **Time Series Analysis:** Using ACF and PACF plots to identify significant lagged features.
* **Model Selection:** Hyperparameter tuning an **AutoRegressive (AR) model** to find the optimal number of lags ($p$).
* **Validation:** Employing **Walk-Forward Validation (WFV)** to simulate a real-world forecasting scenario.

---

## ⚙️ Installation

To run this project, you need Python and a few specific libraries for database connectivity, time series modeling, and plotting.

1.  **Clone the repository:**
    ```bash
    git clone 
    cd dar-es-salaam-air-quality
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib statsmodels pymongo plotly
    ```

3.  **MongoDB Connection:**
    * Ensure a MongoDB instance containing the `air-quality` database and `dar-es-salaam` collection is running.
    * **CRITICAL:** Update the `host` variable in the notebook to the correct IP address of your MongoDB server.

---

## 🏃 Usage

Execute the Jupyter Notebook cell-by-cell. The notebook demonstrates the following workflow:

1.  **Data Wrangle (`wrangle` function):** Extracts PM2.5 readings from Site 11, converts the index to the correct timezone, filters extreme outliers, and resamples the data to hourly means.
2.  **Baseline Model:** Calculates the **Mean Absolute Error (MAE)** using the mean of the training data as the persistent prediction ($\approx 4.05$).
3.  **Hyperparameter Tuning:** A loop tests AR models with **lags from 1 to 30**.
    * The optimal number of lags, **$p=26$**, is selected as it yields the lowest MAE on the training set ($\approx 1.01$).
4.  **Model Evaluation:** The final model is evaluated on the test set using **Walk-Forward Validation**, resulting in a **Test MAE of $\approx 3.97$**.

---

## 🛠️ Technologies Used

| Technology | Purpose | Badge/Icon |
| :--- | :--- | :--- |
| **Python** | Primary scripting and analysis language | [![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/doc/) |
| **Pandas** | Data wrangling, time series indexing, and resampling | [![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/docs/) |
| **NumPy** | Numerical operations and array manipulation | [![NumPy](https://img.shields.io/badge/NumPy-1.x-blue?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/doc/) |
| **statsmodels** | Implementation of **AutoReg** model and ACF/PACF analysis | [![statsmodels](https://img.shields.io/badge/Statsmodels-0.14-green?style=flat-square)](https://www.statsmodels.org/stable/index.html) |
| **Scikit-learn** | Calculating the **Mean Absolute Error (MAE)** | [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/documentation.html) |
| **Plotly/Matplotlib** | Visualization of time series and predictions | [![Plotly](https://img.shields.io/badge/Plotly-4.x-blueviolet?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/python/getting-started/) |
| **PyMongo** | Interface for querying the **MongoDB** database | [![MongoDB](https://img.shields.io/badge/MongoDB-4.x-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/docs/drivers/pymongo/) |

---

## 📊 Results & Analysis

### Data Visualization (Time Series)
The initial time series plot  shows significant **volatility** and **daily/weekly seasonality** in PM2.5 readings, with frequent high-level spikes.

### Autocorrelation Analysis
* **PACF Plot :** The Partial Autocorrelation Function plot shows significant spikes at multiple lags, indicating that the PM2.5 level is directly correlated with readings from many past hours. This justified the use of a high-order AutoRegressive model.
* **Optimal Hyperparameter:** Hyperparameter tuning confirmed **$p=26$ lags** as the best fit for the training data, minimizing the training MAE.

### Model Performance
| Metric | Baseline (Train Mean) | AR Model (Train) | AR Model (Test - WFV) |
| :--- | :--- | :--- | :--- |
| **MAE** | $\approx 4.05$ | $\approx 1.01$ | $\approx 3.97$ |

### Residuals & Model Fit
* **Residuals Histogram :** The histogram of training residuals is **centered around zero**, which indicates that the model's errors are unbiased. This is a characteristic of a well-fitted model.
* **Residuals ACF Plot :** The ACF plot for the residuals shows that almost all correlation spikes are within the blue confidence interval, meaning there is **no significant autocorrelation left** in the errors. This suggests that the AR(26) model effectively captured the time-dependent patterns.

### Walk-Forward Validation (WFV)
The WFV process accurately simulates real-time forecasting. The plot of actual vs. predicted values  shows the AR(26) model successfully capturing the **sharp daily/weekly spikes** and the general trend of the PM2.5 data on the test set. The **Test MAE of 3.97** suggests the model is slightly better than the simple baseline mean, but the large spikes inherent to the data make achieving a very low MAE challenging.

---

## 🧠 Key Learnings

1.  **Time Series Fundamentals:** Solidified understanding of **stationarity** and using PACF to determine the **order of an AR model**.
2.  **Robust Validation:** Gained practical experience with **Walk-Forward Validation**, a crucial technique for avoiding look-ahead bias and simulating realistic forecasting for time series data.
3.  **Model Diagnostics:** Learned to use the **residuals plot (histogram and ACF)** as a diagnostic tool to confirm that a time series model is well-specified (unbiased errors, no remaining autocorrelation).

---

## 📚 References

* *WorldQuant University Applied Data Science Lab Project 3*
