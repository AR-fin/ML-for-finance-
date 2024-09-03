#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:01:38 2024

@author: alessandroricchiuti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Simulated data, two regressors

mu, sigma = 0, 1 # mean and standard deviation

X1 = np.random.normal(mu, sigma, 10000)
X2= np.random.exponential(10,10000)
y=3*X1**2+10/4 + X2**2

X1 = pd.Series(X1)
X2 = pd.Series(X2)
y = pd.Series(y)

X=pd.concat([X1, X2], axis=1)

#%%test data
X1_new=np.random.normal(mu, sigma, 10000)
X2_new=np.random.exponential(10,10000)

X1_new=pd.Series(X1_new)
X2_new=pd.Series(X2_new)

X_new =pd.concat([X1_new, X2_new], axis=1)


noise = np.random.exponential(3, 10000)
y_new = 3*X1_new**2+10/4 + X2_new**2 + noise

#%%OLS
import statsmodels.api as sm
X_reg = sm.add_constant(X)
reg = sm.OLS(y, X_reg).fit()


reg.summary()


score = reg.rsquared_adj
print("R^2 Score: ", score)


X_reg_new = sm.add_constant(X_new)
predictions = reg.predict(X_reg_new)


plt.scatter(y_new, predictions, color='black', s=10)
#%% Training

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42)


svr_rbf = SVR(kernel='rbf', gamma=0.01, C=100, epsilon=1)  
svr_rbf.fit(X_train, y_train)

score = svr_rbf.score(X_test, y_test)
print("R^2 Score: ", score)

rmse = np.sqrt(mean_squared_error(y_test, svr_rbf.predict(X_test)))
print("RMSE: ", rmse)


#%% Test
X_new_scaled = scaler.transform(X_new)

preds = svr_rbf.predict(X_new_scaled)


print("R^2 Score: ", svr_rbf.score(X_new_scaled, y_new))
print("RMSE: ", np.sqrt(mean_squared_error(y_new, preds)))

#%% QQ-Plot

plt.scatter(y_new, preds, color='black', s=10)

#%% Distributions of data
#X1
plt.hist(X1, color='black', bins=30, alpha=0.5)  
plt.hist(X1_new, color='green', bins=30, alpha=0.5)  

plt.figure() 
#X2
plt.hist(X2, color='black', bins=30, alpha=0.5)  
plt.hist(X2_new, color='green', bins=30, alpha=0.5)  

#y
plt.figure() 
plt.hist(y, color='black', bins=30, alpha=0.5)  
plt.hist(y_new, color='green', bins=30, alpha=0.5) 
plt.hist(preds, color='blue', bins=30, alpha=0.5) 

#%% REAL DATA

import yfinance as yf
import pandas as pd


indici = {
    'DJ30': '^DJI',  # Dow Jones 30
    'Russell 3000': '^RUA',
    'Russell 2000': '^RUT',
    'S&P 500': '^GSPC',
    'EuroStoxx 50': '^STOXX50E',
    'MSCI World Index': 'URTH',  
    'MSCI Emerging Markets Index (EM)': 'EEM',  
    'MSCI Europe Index': 'IEUR'  
}


dati_indici = {indice: yf.download(ticker, start="2000-01-01", end="2023-01-01")['Adj Close'] for indice, ticker in indici.items()}


data = {indice: dati.pct_change().dropna() for indice, dati in dati_indici.items()}

data=pd.DataFrame(data)



url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


ff_fattori = pd.read_csv(url, skiprows=3)


ff_fattori.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
ff_fattori['Date'] = pd.to_datetime(ff_fattori['Date'], format='%Y%m%d')


ff_fattori = ff_fattori[ff_fattori['Date'] >= '2000-01-01']


print(ff_fattori.head())

#%% Plot RF
plt.plot(ff_fattori['Date'],ff_fattori['RF'])

#%%
df=pd.merge(left=data, right=ff_fattori, on='Date',how='inner')

df=df.dropna()

#%%

X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
colonne_da_escludere=df[['Date','Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA','RF']]
y = df.drop(columns=colonne_da_escludere)

X_new=X.tail(71)
y_new=y.tail(71)

X=X.head(2000)
y=y.head(2000)


#%%OLS
 
import statsmodels.api as sm
X_reg = sm.add_constant(X)
X_reg_new = sm.add_constant(X_new)


scores={}
predictions={}
plots_OLS={}
for column in y.columns:
   
    reg = sm.OLS(y[column], X_reg).fit()
   
    scores[column] = reg.rsquared_adj
    
    print(f"R^2 Score for {column}: {scores[column]}")
    predictions[column] = reg.predict(X_reg_new)
    
    
  
    fig, ax = plt.subplots()
    
    
    ax.scatter(y_new[column], predictions[column], color='black', s=10)
    
    ax.set_title(f'Scatter plot for {column}')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    
   
    plots_OLS[column] = fig


predictions=pd.DataFrame(predictions)
    




#%%

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  
pi=scaler.fit_transform(y)
pi=pd.DataFrame(pi)

y[:] = scaler.fit_transform(y)

score = {}
rmse = {}


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

#%%
for column in y.columns: 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y[column], test_size=0.33, random_state=42
       
    )

    svr_rbf = SVR(kernel='rbf', gamma=0.01, C=100, epsilon=1) 
    svr_rbf.fit(X_train, y_train)  

    score[column] = svr_rbf.score(X_test, y_test)  
    print(f"R^2 Score for {column}: ", score[column])

    rmse[column] = np.sqrt(mean_squared_error(y_test, svr_rbf.predict(X_test)))  
    print(f"RMSE for {column}: ", rmse[column])

#%%
X_new_scaled = scaler.transform(X_new)
predictions={}
plots_SVR={}
for column in y_new.columns:  
    preds = svr_rbf.predict(X_new)
    predictions[column] = preds
    fig, ax = plt.subplots()
    
    
    ax.scatter(y_new[column], predictions[column], color='black', s=10)
    
    ax.set_title(f'Scatter plot for {column}')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    
   
    plots_SVR[column] = fig



predictions=pd.DataFrame(predictions)   



#%%
import matplotlib.pyplot as plt


n_plots = len(plots_OLS)
 
fig, axes = plt.subplots(n_plots, 2, figsize=(10, 5 * n_plots))


if n_plots == 1:
    axes = [axes]


for i, key in enumerate(plots_OLS.keys()):
   
    axes[i, 0].imshow(plots_OLS[key].canvas.buffer_rgba())
    axes[i, 0].set_title(f'OLS {key}')
    axes[i, 0].axis('off')
    
  
    axes[i, 1].imshow(plots_SVR[key].canvas.buffer_rgba())
    axes[i, 1].set_title(f'SVR {key}')
    axes[i, 1].axis('off')


plt.tight_layout()


plt.show() #for linear relationships OLS is a better fit

