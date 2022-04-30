# Author: Chao Li
# cholianli970518@gmail.com
# github: https://github.com/Cholianlll

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    
    # Print test outputs
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','pvalue','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput.round(4))

def plot_curve(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12,center=True).mean()
    rolstd = timeseries.rolling(12,center=True).std()

    # Plot rolling statistics:
    plt.figure(figsize=(15,6))
    plt.plot(timeseries, color='midnightblue',label='Original')
    plt.plot(rolmean, color='orangered', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.grid(linestyle = "--",color='lightgrey')  
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()