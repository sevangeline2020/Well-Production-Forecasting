''Time Series Forecasting in OIL and gas well''

Decline curve analysis (DCA) is a graphical procedure used for analyzing declining production rates and forecasting future performance of oil and gas wells. Oil and gas production rates decline as a function of time; loss of reservoir pressure, or changing relative volumes of the produced fluids, are usually the cause. Fitting a line through the performance history and assuming this same trend will continue in future forms the basis of DCA concept(PetroWiki). The basic assumption in this procedure is that whatever causes controlled the trend of a curve in the past will continue to govern its trend in the future in a uniform manner. J.J. Arps collected these ideas into a comprehensive set of equations defining the exponential, hyperbolic and harmonic declines. The major application of DCA in the industry today is still based on equations and curves described by Arps. Arps applied the equation of Hyperbola to define three general equations to model production declines.

Stationarity Requirement of Time Series:

The Time series modeling algorithms assume that the series is stationary. A time series is said to be stationary when the mean is not varying locally i.e. it is constant with thme. Additionally the variance and covariance of the i th term and the (i + m)th term should not be a function of time. For cases when there is no stationarity in the time series it needs to be converted to made stationary and then apply stochastic techniques for prediction/forecasting. As expected, the production has a declining trend owing to loss of reservoir pressure, or changing relative volumes of the produced fluids. Thus this declining trend makes the local mean production rate a function of the time (Month-Year). Hence this time series is non-stationary and needs to be made stationary. We implement Dickey-Fuller test which can provide a quick check and confirmatory evidence that your time series is stationary or non-stationary.

(Machine Learning mastery Website: https://machinelearningmastery.com/time-series-data-stationary-python/)

We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%) suggests that we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary). p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary. p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

Trend Elimination- Exponentially Weighted Moving Average Approach:

Trend removal using simple moving approach requires a time window to be defined for the moving average calculation. However the definition of this time period is not very straightforward. We take a ‘weighted moving average’ where more recent values are given a higher weight. A popular one is exponentially weighted moving average where weights are assigned to all the previous values with a decay factor. (http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-moment-functions) The rolling values appear to be varying slightly but there is no specific trend.The p-value for the Dickey Fuller test done here os 0.000018 (<<0.05), hence the null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure. Trend and Seasonality Removal Using Differencing One of the most common methods of dealing with both trend and seasonality is differencing. In this technique, we take the difference of the observation at a particular instant with that at the previous instant. This mostly works well in improving stationarity.

Decline Curve Forecasting- ARIMA:

Source: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

An ARIMA model is a class of statistical models for analyzing and forecasting time series data. It explicitly caters to a suite of standard structures in time series data, and as such provides a simple yet powerful method for making skillful time series forecasts. ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. It is a generalization of the simpler AutoRegressive Moving Average and adds the notion of integration. This acronym is descriptive, capturing the key aspects of the model itself.

Briefly they are: AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations. I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary. MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. Each of these components are explicitly specified in the model as a parameter.

A standard notation of ARIMA:

A standard notation of ARIMA is used of ARIMA(p,d,q) where the parameters are substituted with integer values to quickly indicate the specific ARIMA model being used.

The parameters of the ARIMA model are defined as follows:

p: The number of lag observations included in the model, also called the lag order. d: The number of times that the raw observations are differenced, also called the degree of differencing. q: The size of the moving average window, also called the order of moving average.

A linear regression model is constructed including the specified number and type of terms, and the data is prepared by a degree of differencing in order to make it stationary, i.e. to remove trend and seasonal structures that negatively affect the regression model. A value of 0 can be used for a parameter, which indicates to not use that element of the model. This way, the ARIMA model can be configured to perform the function of an ARMA model, and even a simple AR, I, or MA model.

p – The lag value where the PACF chart crosses the upper confidence interval for the first time. Here, p=1. This can be taken as the starting value. q – The lag value where the ACF chart crosses the upper confidence interval for the first time. Here, q=1. This can be taken as the starting value.

