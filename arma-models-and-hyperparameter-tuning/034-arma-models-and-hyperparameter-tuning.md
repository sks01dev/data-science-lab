<font size="+3"><strong>3.4. ARMA Models</strong></font>


```python
import inspect
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
```


```python
VimeoVideo("665851728", h="95c59d2805", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851728?h=95c59d2805"
    frameborder="0"
    allowfullscreen

></iframe>




<div style="padding: 1em; border: 1px solid #f0ad4e; border-left: 6px solid #f0ad4e; background-color: #fcf8e3; color: #8a6d3b; border-radius: 4px;">

<strong>🛠️ Instruction:</strong> Locate the IP address of the machine running MongoDB and assign it to the variable <code>host</code>. Make sure to use a <strong>string</strong> (i.e., wrap the IP in quotes).<br><br>

<strong>⚠️ Note:</strong> The IP address is <strong>dynamic</strong> — it may change every time you start the lab. Always check the current IP before proceeding.

</div>

<img src="../images/mongo_ip.png" alt="MongoDB" width="600"/>



```python
host = "..."
```

# Prepare Data

## Import

**Task 3.4.1:** Create a client to connect to the MongoDB server running at `host` on port `27017`, then assign the `"air-quality"` database to `db`, and the `"nairobi"` collection to `nairobi`.

- [<span id='technique'>Create a client object for a <span id='tool'>MongoDB</span> instance.](../%40textbook/11-databases-mongodb.ipynb#Servers-and-Clients) 
- [<span id='technique'>Access a database using <span id='tool'>PyMongo.](../%40textbook/11-databases-mongodb.ipynb#Servers-and-Clients)
- [<span id='technique'>Access a collection in a database using <span id='tool'>PyMongo.](../%40textbook/11-databases-mongodb.ipynb#Collections)


```python
client = MongoClient(host="192.99.132.2", port=27017)
db = client["air-quality"]
nairobi = db["nairobi"]
```


```python
VimeoVideo("665851670", h="3efc0c20d4", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851670?h=3efc0c20d4"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.2:** Change your `wrangle` function so that it has a `resample_rule` argument that allows the user to change the resampling interval. The argument default should be `"1H"`.

- [What's an <span id='term'>argument?](../%40textbook/02-python-advanced.ipynb#Functions)
- [<span id='technique'>Include an argument in a function in <span id='tool'>Python.](../%40textbook/02-python-advanced.ipynb#Functions)


```python
def wrangle(collection, resample_rule="1H"):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read results into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers
    df = df[df["P2"] < 500]

    # Resample and forward-fill
    y = df["P2"].resample(resample_rule).mean().fillna(method="ffill")

    return y
```


```python
# Check your work
func_params = set(inspect.signature(wrangle).parameters.keys())
assert func_params == set(
    ["collection", "resample_rule"]
), f"Your function should take two arguments: `'collection'`, `'resample_rule'`. Your function takes the following arguments: {func_params}"
```

**Task 3.4.3:** Use your wrangle function to read the data from the `nairobi` collection into the Series `y`.


```python
y = wrangle(nairobi)
y.head()
```




    timestamp
    2018-09-01 03:00:00+03:00    17.541667
    2018-09-01 04:00:00+03:00    15.800000
    2018-09-01 05:00:00+03:00    11.420000
    2018-09-01 06:00:00+03:00    11.614167
    2018-09-01 07:00:00+03:00    17.665000
    Freq: H, Name: P2, dtype: float64




```python
# Check your work
assert isinstance(y, pd.Series), f"`y` should be a Series, not a {type(y)}."
assert len(y) == 2928, f"`y` should have 2,928 observations, not {len(y)}."
assert (
    y.isnull().sum() == 0
), f"There should be no null values in `y`. Your `y` has {y.isnull().sum()} null values."
```

## Explore


```python
VimeoVideo("665851654", h="687ff8d5ee", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851654?h=687ff8d5ee"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.4:** Create an ACF plot for the data in `y`. Be sure to label the x-axis as `"Lag [hours]"` and the y-axis as `"Correlation Coefficient"`.

- [What's an <span id='term'>ACF plot?](../%40textbook/17-ts-core.ipynb#ACF-Plot)
- [<span id='technique'>Create an ACF plot using <span id='tool'>statsmodels](../%40textbook/18-ts-models.ipynb#Creating-an-ACF-Plot)


```python
fig, ax = plt.subplots(figsize=(15, 6))

plot_acf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
```




    Text(0, 0.5, 'Correlation Coefficient')




    
![png](output_19_1.png)
    



```python
VimeoVideo("665851644", h="e857f05bfb", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851644?h=e857f05bfb"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.5:** Create an PACF plot for the data in `y`. Be sure to label the x-axis as `"Lag [hours]"` and the y-axis as `"Correlation Coefficient"`.

- [What's a PACF plot?](../%40textbook/17-ts-core.ipynb#PACF-Plot)
- [Create an PACF plot using statsmodels](../%40textbook/18-ts-models.ipynb#Creating-a-PACF-Plot)


```python
fig, ax = plt.subplots(figsize=(15, 6))

plot_pacf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
```




    Text(0, 0.5, 'Correlation Coefficient')




    
![png](output_22_1.png)
    


## Split

**Task 3.4.6:** Create a training set `y_train` that contains only readings from October 2018, and a test set `y_test` that contains readings from November 1, 2018.

- [<span id='technique'>Subset a DataFrame by selecting one or more rows in <span id='tool'>pandas.](../%40textbook/04-pandas-advanced.ipynb#Subsetting-with-Masks)


```python
y[0:5]
```




    timestamp
    2018-09-01 03:00:00+03:00    17.541667
    2018-09-01 04:00:00+03:00    15.800000
    2018-09-01 05:00:00+03:00    11.420000
    2018-09-01 06:00:00+03:00    11.614167
    2018-09-01 07:00:00+03:00    17.665000
    Freq: H, Name: P2, dtype: float64




```python
y_train = y['2018-10-01': '2018-10-31']
y_test = y['2018-11-01']

y_train.head()
```




    timestamp
    2018-10-01 00:00:00+03:00    7.030000
    2018-10-01 01:00:00+03:00    6.559167
    2018-10-01 02:00:00+03:00    6.692500
    2018-10-01 03:00:00+03:00    8.011667
    2018-10-01 04:00:00+03:00    8.885833
    Freq: H, Name: P2, dtype: float64




```python
# Check your work
assert (
    len(y_train) == 744
), f"`y_train` should have 744 observations, not {len(y_train)}."
assert len(y_test) == 24, f"`y_test` should have 24 observations, not {len(y_test)}."
```

# Build Model

## Baseline

**Task 3.4.7:** Calculate the baseline mean absolute error for your model.


```python
y_train_mean = y.mean()
y_pred_baseline = [y_train_mean] * len(y)
mae_baseline = mean_absolute_error(y, y_pred_baseline)

print("Mean P2 Reading:", round(y_train_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))
```

    Mean P2 Reading: 9.11
    Baseline MAE: 3.66


## Iterate


```python
VimeoVideo("665851576", h="36e2dc6269", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851576?h=36e2dc6269"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.8:** Create ranges for possible $p$ and $q$ values. `p_params` should range between 0 and 25, by steps of 8. `q_params` should range between 0 and 3 by steps of 1.

- [What's a <span id='term'>hyperparameter?](../%40textbook/17-ts-core.ipynb#Hyperparameters)
- [What's an <span id='term'>iterator?](../%40textbook/02-python-advanced.ipynb#Iterators-and-Iterables-)
- [<span id='technique'>Create a range in <span id='tool'>Python.](../%40textbook/18-ts-models.ipynb#Hyperparameters)


```python
p_params = range(0, 25, 8) # normal data from past, lookback range to choose features
q_params = range(0, 3, 1) # residuals from past, kind of shock events in which the predictions and true values differ heavily, hence error term from past 
```


```python
VimeoVideo("665851476", h="d60346ed30", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851476?h=d60346ed30"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.9:** Complete the code below to train a model with every combination of hyperparameters in `p_params` and `q_params`. Every time the model is trained, the mean absolute error is calculated and then saved to a dictionary. If you're not sure where to start, do the code-along with Nicholas!

- [What's an <span id='term'>ARMA model?](../%40textbook/17-ts-core.ipynb#ARMA-Models)
- [<span id='technique'>Append an item to a list in <span id='tool'>Python.](../%40textbook/01-python-getting-started.ipynb#Appending-Items)
- [<span id='technique'>Calculate the mean absolute error for a list of predictions in <span id='tool'>scikit-learn.](../%40textbook/15-ml-regression.ipynb#Calculating-the-Mean-Absolute-Error-for-a-List-of-Predictions)
- [<span id='technique'>Instantiate a predictor in <span id='tool'>statsmodels.](../%40textbook/18-ts-models.ipynb#Iterating)
- [<span id='technique'>Train a model in <span id='tool'>statsmodels.](../%40textbook/18-ts-models.ipynb#Iterating)
- [<span id='technique'>Write a for loop in <span id='tool'>Python.](../%40textbook/01-python-getting-started.ipynb#Working-with-for-Loops)


```python
# Create dictionary to store MAEs
mae_grid = {}
# Outer loop: Iterate through possible values for `p`
for p in p_params:
    # Create key-value pair in dict. Key is `p`, value is empty list.
    mae_grid[p] = []
    # Inner loop: Iterate through possible values for `q`
    for q in q_params:
        # Combination of hyperparameters for model
        order = (p, 0, q) # 3nd param is zero
        # Note start time
        start_time = time.time()
        # Train model
        model = ARIMA(y_train, order=order).fit()
        # Calculate model training time
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict()
        # Calculate training MAE
        mae = mean_absolute_error(y_train, y_pred)
        # Append MAE to list in dictionary
        mae_grid[p].append(mae)

print()
print(mae_grid)
```

    Trained ARIMA (0, 0, 0) in 0.08 seconds.
    Trained ARIMA (0, 0, 1) in 0.04 seconds.
    Trained ARIMA (0, 0, 2) in 0.06 seconds.
    Trained ARIMA (8, 0, 0) in 0.19 seconds.
    Trained ARIMA (8, 0, 1) in 0.71 seconds.
    Trained ARIMA (8, 0, 2) in 1.25 seconds.
    Trained ARIMA (16, 0, 0) in 0.9 seconds.
    Trained ARIMA (16, 0, 1) in 2.83 seconds.
    Trained ARIMA (16, 0, 2) in 4.05 seconds.
    Trained ARIMA (24, 0, 0) in 5.41 seconds.
    Trained ARIMA (24, 0, 1) in 9.68 seconds.
    Trained ARIMA (24, 0, 2) in 12.46 seconds.
    
    {0: [4.171460443827197, 3.3506427433167634, 3.1057222588243443], 8: [2.9384480570053584, 2.9149013330115974, 2.9132458963032315], 16: [2.9201084725449653, 2.9294363113458335, 2.9120798462337034], 24: [2.914390327410196, 2.9136013236816525, 2.8979121370798664]}



```python
VimeoVideo("665851464", h="12f4080d0b", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851464?h=12f4080d0b"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.10:** Organize all the MAE's from above in a DataFrame names `mae_df`. Each row represents a possible value for $q$ and each column represents a possible value for $p$. 

- [<span id='technique'>Create a DataFrame from a dictionary using <span id='tool'>pandas.](../%40textbook/03-pandas-getting-started.ipynb#Working-with-DataFrames)


```python
# let's see the hyperparam grid
mae_df = pd.DataFrame(
    mae_grid
)
mae_df.round(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>8</th>
      <th>16</th>
      <th>24</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.1715</td>
      <td>2.9384</td>
      <td>2.9201</td>
      <td>2.9144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.3506</td>
      <td>2.9149</td>
      <td>2.9294</td>
      <td>2.9136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.1057</td>
      <td>2.9132</td>
      <td>2.9121</td>
      <td>2.8979</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(mae_df.min(axis=1))
```

    0    2.914390
    1    2.913601
    2    2.897912
    dtype: float64



```python
VimeoVideo("665851453", h="dfd415bc08", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851453?h=dfd415bc08"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.11:** Create heatmap of the values in `mae_grid`. Be sure to label your x-axis `"p values"` and your y-axis `"q values"`. 

- [<span id='technique'>Create a heatmap in <span id='tool'>seaborn.](../%40textbook/09-visualization-seaborn.ipynb#Correlation-Heatmaps)


```python
sns.heatmap(mae_df, cmap='Blues')

plt.xlabel("p values")
plt.ylabel("q values")
```




    Text(50.722222222222214, 0.5, 'q values')




    
![png](output_45_1.png)
    



```python
VimeoVideo("665851444", h="8b58161f26", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851444?h=8b58161f26"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.12:** Use the [`plot_diagnostics`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.plot_diagnostics.html) method to check the residuals for your model. Keep in mind that the plot will represent the residuals from the last model you trained, so make sure it was your best model, too!

- [<span id='technique'>Examine time series model residuals using <span id='tool'>statsmodels.](../%40textbook/18-ts-models.ipynb#Residuals)


```python
fig, ax = plt.subplots(figsize=(15, 12))

model.plot_diagnostics(fig=fig)
```




    
![png](output_48_0.png)
    




    
![png](output_48_1.png)
    


## Evaluate


```python
VimeoVideo("665851439", h="c48d80cdf4", width=600)
```





<iframe
    width="600"
    height="300"
    src="https://player.vimeo.com/video/665851439?h=c48d80cdf4"
    frameborder="0"
    allowfullscreen

></iframe>




**Task 3.4.13:** Complete the code below to perform walk-forward validation for your model for the entire test set `y_test`. Store your model's predictions in the Series `y_pred_wfv`. Choose the values for $p$ and $q$ that best balance model performance and computation time. Remember: This model is going to have to train 24 times before you can see your test MAE!<span style='color: transparent; font-size:1%'>WQU WorldQuant University Applied Data Science Lab QQQQ</span>


```python
y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = ARIMA(history, order=(8, 0, 1)).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])
```


```python
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))
```


```python
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))
```

# Communicate Results


```python
VimeoVideo("665851423", h="8236ff348f", width=600)
```

**Task 3.4.14:** First, generate the list of training predictions for your model. Next, create a DataFrame `df_predictions` with the true values `y_test` and your predictions `y_pred_wfv` (don't forget the index). Finally, plot `df_predictions` using plotly express. Make sure that the y-axis is labeled `"P2"`. 

- [<span id='technique'>Generate in-sample predictions for a model in <span id='tool'>statsmodels.](../%40textbook/18-ts-models.ipynb#Iterating)
- [<span id='technique'>Create a DataFrame from a dictionary using <span id='tool'>pandas.](../%40textbook/03-pandas-getting-started.ipynb#Working-with-DataFrames)
- [<span id='technique'>Create a line plot in <span id='tool'>pandas.](../%40textbook/07-visualization-pandas.ipynb#Line-Plots)    


```python
df_predictions = ...
fig = ...
fig.show()
```

---
Copyright 2023 WorldQuant University. This
content is licensed solely for personal use. Redistribution or
publication of this material is strictly prohibited.

