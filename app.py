import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
rcParams['figure.figsize']= 15, 16
import warnings
warnings.filterwarnings('ignore')
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
base="dark"
primaryColor="red"

st.title('**WELL PRODUCTION FORECASTING WITH TIME SERIES ANALYSIS**')
st.markdown('**This is first project in data science**')
with st.expander("Brief explanation"):
     st.write("""
         **Decline curve analysis (DCA) is a graphical procedure used for analyzing declining production rates and forecasting future performance of oil and gas wells. Oil and gas production rates decline as a function of time; loss of reservoir pressure, or changing relative volumes of the produced fluids, are usually the cause. Fitting a line through the performance history and assuming this same trend will continue in future forms the basis of DCA concept(PetroWiki).
         The basic assumption in this procedure is that whatever causes controlled the trend of a curve in the past will continue to govern its trend in the future in a uniform manner. J.J. Arps collected these ideas into a comprehensive set of equations defining the exponential, hyperbolic and harmonic declines.(Representative figure below)
         The major application of DCA in the industry today is still based on equations and curves described by Arps. Arps applied the equation of Hyperbola to define three general equations to model production declines.**
         """)
     st.image("OIL WELL.jpg")

st.header('**PROBLEM STATEMENT**')
st.subheader('**This project aims at replacing the traditional DCA and discusses the application of the widely accepted concepts of time series analysis for forecasting well production data by analyzing statistical trends from historical data.**')

st.header('**SAMPLE DATASET**')
column = ['Month','Production_rate']
data = pd.read_csv("ProductionData2.xlsx - Sheet1.csv", names = column);
st.table(data)

st.header('**DATETIME ANALYSIS**')
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
data = pd.read_csv("ProductionData2.xlsx - Sheet1.csv",names = column, parse_dates=['Month'], index_col=['Month'],date_parser=dateparse);
st.line_chart(data)
st.pyplot()

st.header('**AUTOCORRELATION PLOT**')
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(data)
st.pyplot()

st.header('**STATIONARITY REQUIREMENT OF TIME SERIES**')
#Let's visualize the production trend available for the well.
ts = data['Production_rate'][1:]
st.line_chart(ts)
st.pyplot()

st.header('**TREND ELIMINATION-MOVING AVERAGE APPROACH**')
ts_log = np.log(ts)
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(ts_log, label= 'log(Original)')
st.pyplot(fig)

moving_avg = ts_log.rolling(10).mean()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(moving_avg, color='red')
ax.plot(ts_log)
st.pyplot(fig)

ts_log_moving_avg_diff = ts_log - moving_avg
st.line_chart(ts_log_moving_avg_diff)
fig, ax = plt.subplots(figsize=(10,6))
ts_log_moving_avg_diff.dropna(inplace =True)
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(ts_log_moving_avg_diff)
ax.plot(test_stationarity(ts_log_moving_avg_diff))
st.pyplot(fig)

st.header('TREND ELIMINATION-EXPONENTIALLY WEIGHTED MOVING AVERAGE APPROACH')
exp_weighted_avg = ts_log.ewm(halflife=2).mean()
st.pyplot.figure(figsize=(10,6))
st.pyplot.plot(ts_log)
st.pyplot.plot(exp_weighted_avg, color ='red')

ts_log_ewma_diff = ts_log - exp_weighted_avg
st.line_chart(test_stationarity(ts_log_ewma_diff))

ts_log_diff = ts_log - ts_log.shift()
st.pyplot.figure(figsize=(10,6))
st.pyplot(ts_log_diff)
st.pyplot(ts_log)
st.pyplot(ts_log.shift())
st.pyplot(ts_log.diff())

ts_log_diff.dropna(inplace=True)
st.pyplot.figure(figsize=(10,6))
st.line_chart(test_stationarity(ts_log_diff))

st.header('DECLINE CURVE FORECASTING-ARIMA')
ts_log_diff_active = ts_log_diff

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff_active, nlags=5)
lag_pacf = pacf(ts_log_diff_active, nlags=5, method='ols')
st.pyplot.figure(figsize=(10,5))
#Plot ACF: 
st.pyplot.subplot(121) 
st.pyplot.plot(lag_acf)
st.pyplot.axhline(y=0,linestyle='--',color='gray')
st.pyplot.axhline(y=-1.96/np.sqrt(len(ts_log_diff_active)),linestyle='--',color='gray')
st.pyplot.axhline(y=1.96/np.sqrt(len(ts_log_diff_active)),linestyle='--',color='gray')
st.pyplot.title('Autocorrelation Function')
#plt.figure(figsize=(15,5))

#Plot PACF:
st.pyplot.subplot(122)
st.pyplot.plot(lag_pacf)
st.pyplot.axhline(y=0,linestyle='--',color='gray')
st.pyplot.axhline(y=-1.96/np.sqrt(len(ts_log_diff_active)),linestyle='--',color='gray')
st.pyplot.axhline(y=1.96/np.sqrt(len(ts_log_diff_active)),linestyle='--',color='gray')
st.pyplot.title('Partial Autocorrelation Function')
st.pyplot.tight_layout()


model_AR = ARIMA(ts_log, order=(2, 1, 0))  
results_ARIMA_AR = model_AR.fit(disp=-1)  
st.pyplot.figure(figsize=(10,5))
st.pyplot.plot(ts_log_diff_active)
st.pyplot.plot(results_ARIMA_AR.fittedvalues, color='red')
st.pyplot.title('RSS: %.4f'% sum((results_ARIMA_AR.fittedvalues-ts_log_diff)**2))

model_MA = ARIMA(ts_log, order=(0, 1, 2))  
results_ARIMA_MA = model_MA.fit(disp=-1)  
st.pyplot.figure(figsize=(10,5))
st.pyplot.plot(ts_log_diff_active)
st.pyplot.plot(results_ARIMA_MA.fittedvalues, color='red')
st.pyplot.title('RSS: %.4f'% sum((results_ARIMA_MA.fittedvalues-ts_log_diff)**2))

model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
st.pyplot.figure(figsize=(10,5))
st.pyplot.plot(ts_log_diff_active)
st.pyplot.plot(results_ARIMA.fittedvalues, color='red')
st.pyplot.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

predictions_ARIMA_diff = pd.Series(results_ARIMA_AR.fittedvalues, copy=True)
st.write(predictions_ARIMA_diff())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
st.write(predictions_ARIMA_diff_cumsum)

predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
st.write(predictions_ARIMA_log)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
st.pyplot.figure(figsize=(10,5))
st.pyplot.plot()
st.pyplot.plot(predictions_ARIMA)

st.pyplot.gca().legend(('Original Decline Curve','ARIMA Model Decline Curve'))

st.balloons()
