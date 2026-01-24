# Main for the macroeconometrics and machine learning project
# Packages import

import numpy as np
import pandas as pd
import datetime
import scipy.stats
import matplotlib.pyplot as plt
import sklearn as sk
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

"""
Functions block
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
def sequential_procedure(feature: pd.Series, alpha)->str:
    """
    Function to perform a sequential analysis and retrieve the transformation to perform
    for all variables in the database to ensure stationarity
    :param feature:
    :return:
    """
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
            pvals = res.pvalues[2]
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

# Function to determine the integration degree for several series
def stationarity_test(df_feature: pd.DataFrame)->list:
    """
    Function to perform a sequential procedure to test for stationarity over several
    variables
    :param df_feature:
    :return:
    """
    list_statio: list = []
    for i in range(df_feature.shape[1]):
        # get the feature
        feature = df_feature.iloc[:, i]

        # Plot the figure and show the autocorrelograms ==> For debug / look at the series
        #fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        #axes[0].plot(feature)
        #axes[0].set_title(f"Time serie for {df_feature.columns[i]}")
        #plot_acf(feature, ax=axes[1], lags=36)
        #axes[1].set_title(f"ACF for {df_feature.columns[i]}")
        #plot_pacf(feature, ax=axes[2], lags=36)
        #axes[2].set_title(f"PACF for {df_feature.columns[i]}")
        #plt.show()

        # Stationarity analysis from ADF test (sequential strategy)
        list_statio.append(sequential_procedure(feature, 0.05))
    return list_statio

# Function to perform adequate transformation on the EA-MD Monthly series
def transform_data(feature: pd.Series, code: int)-> pd.Series:
    """
    Function to apply a given transformation depending on a feature
    depending on the code given by the user
    :param feature: serie to transform (industrial production, unemployment, ...)
    :param code: type of transformation to perform
    :return:
    """

    list_code: list = [0, 1,2,3,4,5]
    if code not in list_code:
        raise Exception(f"code {code} is not part of the proposed transformations"
                        f"for this function")
    match code:
        case 0:
            return feature
        case 1:
            return 100 * np.log(feature)
        case 2:
            modif_feature: np.array = np.array(np.nan)
            return np.append(modif_feature, 100 * np.diff(np.log(feature)))
        case 3:
            modif_feature: np.array = np.array([np.nan, np.nan])
            return np.append(modif_feature, np.diff(np.log(feature),
                           n=2))
        case 4:
            modif_feature: np.array = np.array(np.nan)
            return np.append(modif_feature,np.diff(feature))
        case 5:
            modif_feature: np.array = np.array([np.nan, np.nan])
            return np.append(modif_feature, np.diff(feature,
                           n=2))

def remove_outlier(df:pd.DataFrame, q:float)->pd.DataFrame:
    """
    Function removing outliers from a dataframe for a given quantile
    :param df:
    :param q:
    :return:
    """
    for i in range(df.shape[1]):
        feature:pd.Series = df.iloc[:,i]
        vmin:float = feature.quantile(q)
        vmax:float = feature.quantile(1-q)
        df.iloc[:,i] = df.iloc[:,i].clip(lower=vmin, upper=vmax)

    return df

# Function to standardize the series prior to factors extraction

# Function to compute the number of factors to retrieve following By & NG

# Generic function to build the index using regression/ML results


"""
Data imports 
"""

# Import of the FRED/MD enhanced database
data:pd.DataFrame = pd.read_excel("data/BDD.xlsx", "Database")
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

# We distinguish between EA-MD data and financial variables
eurostoxx_index:int = data.columns.get_loc("Eurostoxx")
df_fred_md:pd.DataFrame = data.iloc[:, 1:eurostoxx_index].ffill()
df_fi_variables: pd.DataFrame = data.iloc[:, eurostoxx_index:np.shape(data)[1]].ffill()

# FFill performed at this stage, should EM when computing factors

"""
First part: exploratory analysis of data and pre-treatment
"""

# We limit ourselves to the period for which we have bonds financial data
df_macro: pd.DataFrame =df_fred_md[df_fred_md.index.isin(dates_to_use)]
df_fi: pd.DataFrame = df_fi_variables[df_fi_variables.index.isin(dates_to_use)]

# Stationarity test for all the EA-MD series in our dataset
statio_list: list = stationarity_test(df_macro)
print(statio_list)

# We apply the EA-MD recommended corrections for all series and perform the sequential procedure again
df_code_transfo: pd.DataFrame = pd.read_excel("data/BDD.xlsx","Transformations EA-MD")

# We apply the Heavy transformations by default (no prior on the integration degree)
list_transfo_features: list = df_code_transfo.iloc[1,1:np.shape(df_code_transfo)[1]]
df_macro_transfo: pd.DataFrame = df_macro.copy()

# Loop to build the transformed dataset
for i in range(df_macro.shape[1]):
    # Retrieve the feature
    feature = df_macro.iloc[:, i]

    # Perform the appropriate transformation
    code:int = list_transfo_features[i]
    feature_transfo = transform_data(feature, code = code)
    df_macro_transfo.iloc[:,i] = feature_transfo


statio_list_post_transfo:list = stationarity_test(df_macro_transfo)
print(statio_list_post_transfo)

# The results are a bit different from what could be expected (e.g. unemployment still non stationary)
# Might be a problem with the sequential estimation function

# Similar analysis for financial series
statio_list_varfi: list = stationarity_test(df_fi)
print(statio_list_varfi)

# For all series which are not I(0), we perform a differenciation at first order
df_fi_transfo: pd.DataFrame = df_fi.copy()
for i in range(df_fi.shape[1]):
    # Retrieve the feature
    feature = df_fi.iloc[:, i]
    integration_type:str = statio_list_varfi[i]
    if integration_type in ["I(0) + C + T", "I(0) + C", "I(0)"]:
        code:int = 0
    else:
        code:int = 4
    feature_transfo = transform_data(feature, code=code)
    df_fi_transfo.iloc[:, i] = feature_transfo

statio_list_fi_post_transfo:list = stationarity_test(df_fi_transfo)
print(statio_list_fi_post_transfo)

# All financial variables are stationary (with potentially trend / constant)

# We concatenate the dataframe with all stationarity treatment performed
df_final: pd.DataFrame = pd.concat([df_macro_transfo, df_fi_transfo], axis=1)
print(df_final.head())
print(df_final.describe())

# We remove outliers for further treatments
df_final = remove_outlier(df_final, 0.05)
print(df_final.describe())

"""
Second part: factor-based index construction
"""

"""
Third part: ML-based index construction
"""


"""
Fourth part: comparison of both approaches (metrics)
"""

"""
Fifth part: comparison with existing indexes
"""
