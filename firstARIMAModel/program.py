import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
import seaborn as sns
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima.model import ARIMA
import warnings
import yfinance as yf
from datetime import datetime
warnings.filterwarnings("ignore")

#yahoo finance information to download dataframe
ticker = 'AMZN'
start_date = '2013-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

data = yf.download(ticker, start=start_date, end=end_date, interval='1mo')
data = data.dropna()

data_close = data['Close']

#Define paramters to take any value between 0 and 3. Then generate all different combinations.
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

aic = []
parameters = []
for param in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(data_close, order=param, enforce_stationarity=True, enforce_invertibility=True)
        results = mod.fit()
        aic.append(results.aic)
        parameters.append(param)
        print('ARIMA{} - AIC: {}'.format(param, results.aic))
    except Exception as e:
        print(f"Model ARAIMA{param} failed: {e}")
        continue

index_min = min(range(len(aic)), key=aic.__getitem__)
print('The optimal model is: ARIMA{} - AIC{}'.format(parameters[index_min], aic[index_min]))


model = ARIMA(data_close, order=parameters[index_min])
model_fit = model.fit()

forecast_steps = 10
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

forecast_index = pd.date_range(data_close.index[-1], periods=forecast_steps+1, freq='M')[1:]

plt.figure(figsize=(10, 6))
plt.plot(data_close.index, data_close, label='Actual', color='blue')

plt.plot(forecast_index, forecast_values, label='Forecast', color='red')

plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.3)

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual and Forecasted Data for {ticker}')
plt.legend()
plt.show()