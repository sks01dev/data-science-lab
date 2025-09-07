# Mexico City Housing Price Analysis

This project explores apartment prices in Mexico City using **data wrangling**, **visualization**, and **predictive modeling** techniques.  
The goal is to understand key factors that influence housing costs and to build a regression model that can provide approximate price predictions.  

This analysis was developed as part of my **Data Science Lab coursework**, and it demonstrates practical applications of data preprocessing, feature engineering, and machine learning.

---

## 📌 Project Overview

- **Objective:**  
  Analyze real-estate listings from Mexico City and predict apartment prices based on property features and location.

- For the complete analysis, see the [analysis.md](analysis.md) file.

- **Tasks Covered:**  
  - 📂 **Data Wrangling:** Reading, cleaning, and structuring the dataset for analysis.  
  - 🧹 **Data Preprocessing:** Handling missing values and ensuring consistency across features.  
  - 🌍 **Geospatial Analysis:** Mapping apartment locations with Plotly (scatter mapbox) to study spatial patterns.  
  - 📊 **Exploratory Data Analysis (EDA):**  
    - Distributions of price and apartment sizes  
    - Relationship between surface area and price  
    - Price trends across boroughs (`alcaldías`) of Mexico City  
  - 🏗 **Feature Engineering:** Creating and encoding features (e.g., boroughs) to use in predictive modeling.  
  - 🤖 **Model Training:** Training a Ridge regression model using scikit-learn to predict housing prices.  
  - 📈 **Model Interpretation:** Extracting coefficients to evaluate the importance of each feature.  

---

## 📊 Example Visualizations

- **Geospatial distribution of housing prices**  
  ![Scatter Mapbox](images/output_29_0.png)

- **Feature importance from Ridge regression**  
  ![Feature Importance](images/output_53_0.png)

---

## 🛠 Tools & Libraries

- **Python** (pandas, numpy)
- **Data Visualization:** matplotlib, seaborn, plotly  
- **Machine Learning:** scikit-learn (Ridge regression, preprocessing, evaluation)  
- **Jupyter Notebook** for interactive analysis  

---

## 🚀 Results & Insights

- Apartment **surface area** is one of the strongest predictors of price.  
- Geographic location matters: boroughs such as *Benito Juárez* and *Miguel Hidalgo* tend to have higher housing costs.  
- Lat/Lon coordinates can enhance models by capturing **local spatial variations** in housing prices.  
- Ridge regression provided a stable baseline model, balancing interpretability with predictive power.  

---

## 📌 Key Takeaways

- Data preprocessing and careful handling of missing values are essential for reliable modeling.  
- Geospatial visualization highlights strong location-driven effects in real estate data.  
- Even a relatively simple linear model like Ridge regression can provide meaningful insights into housing markets when features are well prepared.  
