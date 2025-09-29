# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


data = pd.read_excel("Coffe_sales.xlsx")

data['date'] = pd.to_datetime(data['date'])
daily = data.groupby('date')['money'].sum()

result = adfuller(daily.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

train_size = int(len(daily) * 0.8)
train, test = daily.iloc[:train_size], daily.iloc[train_size:]

plot_acf(daily.dropna(), lags=30)
plot_pacf(daily.dropna(), lags=30)
plt.show()

model = AutoReg(train, lags=5).fit()
print(model.summary())

preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

error = mean_squared_error(test, preds)
print("Mean Squared Error:", error)

plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, preds, label='Predicted', color='red')
plt.legend()
plt.title("Daily Coffee Sales: Actual vs Predicted")
plt.show()

```
### OUTPUT:

GIVEN DATA


<img width="405" height="115" alt="image" src="https://github.com/user-attachments/assets/4dbc9944-bf52-40b9-8bf3-4e66fb1167d4" />


PACF - ACF


<img width="568" height="435" alt="TS EXP 7 PIC" src="https://github.com/user-attachments/assets/404448e7-5b0f-411b-8520-f71613c10603" />
<img width="568" height="435" alt="TS EXP 7 PIC2" src="https://github.com/user-attachments/assets/a9ecde43-e0c8-4084-9e55-6c066d8fc7d4" />


PREDICTION


<img width="304" height="20" alt="image" src="https://github.com/user-attachments/assets/496eaf15-3ba5-434f-9dea-bb16a86323ab" />


FINIAL PREDICTION


<img width="831" height="451" alt="TS EXP 7 PIC3" src="https://github.com/user-attachments/assets/c98d615b-6030-4189-bd51-38a959777d67" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
