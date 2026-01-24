# Main for the macroeconometrics and machine learning project
# Packages import
from math import isnan

import numpy as np
import pandas as pd
import datetime
import scipy.stats
import matplotlib.pyplot as plt
import sklearn as sk
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Import of the FRED/MD enhanced database
data = pd.read_excel("data/BDD.xlsx")
print(data.head())

# Start with basic description of the data
print(f"Number of observations {data.shape[0]} and number of features {data.shape[1]}")
print("Type of the data:", data.dtypes)

# Keep only dates for which financial variables are available
dates_to_use: list = data.loc[(data["Time"]>="2004-09-01")&(data["Time"]<"2025-01-01"), "Time"].tolist()

# Set date as index
data.set_index(data["Time"], drop=True, inplace=True)
print(data.head())
print(round(data.describe(),2))

# Number of missing values per column
print('Number of missing values per feature:', data.isna().sum())

# We limit to the period for which we have bonds financial data
data_to_use: pd.DataFrame =data[data.index.isin(dates_to_use)]

# Missing value for EA-MD Data (FFill for now, perform EM instead)
# First non EA-MD data
eurostoxx_index = data.columns.get_loc("Eurostoxx")
data = data.iloc[0:np.shape(data)[0], 1:eurostoxx_index].ffill()


"""
First part: exploratory analysis of data and pre-treatment
"""

# Function to test the seasonality

# Function to estimate the ADF regression
def adf_reg(feature: pd.Series, p:int, regression:str):
    """
    Function to compute an ADF regression if we reject the stationarity assumption of ADF test
    to estimate the trend and constant coefficient
    :param feature: serie we are estimating
    :param p: number of lags (chosen following an information criteria)
    :return:
    """

    y = feature.values

    # We differenciate the serie
    dy = np.diff(y)

    # length of the dependent variable
    t:int = len(dy)

    # We only keep the last p-1 values for y_lag
    y_lag = y[p:-1]

    # We build our regressor matrix
    x = [y_lag]

    # Add regression if we are testing with a constant or trend component: we add a const
    if regression in {"c","ct"}:
        x.insert(0, np.ones(len(y_lag)))

    # If we are testing the specification with the trend, we add the trend
    if regression  == "ct":
        trend = np.arange(1, len(y_lag)+1)
        x.insert(1, trend)

    # Cas where we have a positive number of lags for ADF reg
    if p > 0:
        # We add the lagged matrix
        for i in range(p):
            x.insert(3+i, dy[p - i:t - i])

    x = np.column_stack(x)

    # We adjust the number of values for dy
    dy = dy[p:]
    return sm.OLS(dy, x).fit()

# Function to test for stationarity and perform appropriate standardization procedure
def stationarity_test(feature: pd.Series, alpha)->str:
    """
    Function to perform a sequential analysis and retrieve the transformation to perform
    for all variables in the database to ensure stationarity
    :param feature:
    :return:
    """

    # Check for the type and missing value


    # list with potential results for the sequential test
    list_configs = ["ct","c","n"]
    list_res_nonstatio = ["I(1) + C + T", "I(1) + C", "I(1)"]
    list_res_statio = ["I(0) + C + T", "I(0) + C", "I(0)"]

    for i, reg in enumerate(list_configs):

        # Perform the ADF test
        res_adf = adfuller(
            feature.dropna(),
            regression = reg,
            autolag = "AIC",
            store = True,
            regresults=True
        )

        # adf-test stat
        adf_stat: float = res_adf[0]

        # critical value for ADF test
        critical_val: float = res_adf[2][f"{int(alpha*100)}%"]

        # optimal number of lag used
        nb_lags: int = res_adf[-1].usedlag

        # Estimate ADF regression
        res = adf_reg(feature.dropna(), nb_lags, reg)

        # Check for the significance of the deterministic terms
        if reg == "ct":
            pvals = res.pvalues[:2]
            signi = np.all(pvals < alpha)
        elif reg == "c":
            pval = res.pvalues[1]
            signi = pval < alpha
        else:
            signi = True

        # If the deterministic terms are not significative, we turn to the next specification
        if not signi:
            continue

        # We stop whenever we reject and the procedure is significative
        if adf_stat > critical_val:
            return list_res_nonstatio[i]
        else:
            return list_res_statio[i]

# List to stock the integration degree of all series
statio_df: list = []
for i in range(1,eurostoxx_index):
    feature = data_to_use.iloc[0:np.shape(data)[0], i]

    # Plot the figure and show the autocorrelograms


    statio_df.append(stationarity_test(feature, 0.05))

print(statio_df)

# If the series are not stationary, perform the recommended transformation from EA-MD

# Similar analysis for financial series

"""
Second part: factor-based index construction
"""

# Function to compute the number of factors to retrieve following By & NG

"""
Third part: ML-based index construction
"""

"""
Fourth part: comparison of both approaches (metrics)
"""

"""
Fifth part: comparison with existing indexes
"""
