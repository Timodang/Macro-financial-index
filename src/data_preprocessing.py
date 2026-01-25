import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

"""
Functions block
"""

def adf_reg(feature: pd.Series, p: int, regression: str):
    """
    Function to compute an ADF regression to estimate the trend and constant coefficient.
    Corrected to ensure proper lag indexing and variable order.
    """
    y = feature.values
    dy = np.diff(y)
    t = len(dy)

    # We keep values corresponding to t-1
    y_lag = y[p:-1]

    # Construction of the regressor matrix components
    x_components = []

    # 1. Constant
    if regression in {"c", "ct"}:
        x_components.append(np.ones(len(y_lag)))

    # 2. Trend (placed at index 1 if present)
    if regression == "ct":
        trend = np.arange(1, len(y_lag) + 1)
        x_components.append(trend)

    # 3. Lagged level (Gamma coefficient)
    x_components.append(y_lag)

    # 4. Lagged differences (Dynamic components)
    if p > 0:
        for i in range(p):
            # Corrected indexing: use previous values (p-1-i) to explain current diff
            x_components.append(dy[p - 1 - i : t - 1 - i])

    x = np.column_stack(x_components)

    # Dependent variable adjusted for lags
    dy_dependent = dy[p:]

    return sm.OLS(dy_dependent, x).fit()


def sequential_procedure(feature: pd.Series, alpha: float) -> str:
    """
    Function to perform a sequential analysis to determine stationarity.
    """
    list_configs = ["ct", "c", "n"]
    list_res_nonstatio = ["I(1) + C + T", "I(1) + C", "I(1)"]
    list_res_statio = ["I(0) + C + T", "I(0) + C", "I(0)"]

    for i, reg in enumerate(list_configs):

        # Perform the ADF test
        res_adf = adfuller(
            feature.dropna(),
            regression=reg,
            autolag="AIC",
            store=True,
            regresults=True
        )

        adf_stat = res_adf[0]
        # We use the p-value for the stationarity decision
        adf_pvalue = res_adf[1]
        
        # Optimal number of lags used
        nb_lags = res_adf[-1].usedlag

        # If 'n' (no deterministic terms), we just check stationarity
        if reg == "n":
            if adf_pvalue < alpha:
                return list_res_statio[i]
            else:
                return list_res_nonstatio[i]

        # Estimate ADF regression to test deterministic terms
        res = adf_reg(feature.dropna(), nb_lags, reg)

        # Check for the significance of the deterministic terms
        # The indices depend on the construction order in adf_reg: [Const, Trend, Gamma, Lags...]
        signi = False
        if reg == "ct":
            # Test Trend (Index 1)
            trend_pval = res.pvalues[1]
            signi = trend_pval < alpha
        elif reg == "c":
            # Test Constant (Index 0)
            const_pval = res.pvalues[0]
            signi = const_pval < alpha

        # If the deterministic terms are not significant, we move to the next specification
        if not signi:
            continue

        # If terms are significant, we check stationarity
        if adf_pvalue < alpha:
            return list_res_statio[i]
        else:
            return list_res_nonstatio[i]
            
    # Fallback return
    return "Unknown"


def stationarity_test(df_feature: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform a sequential procedure to test for stationarity over several variables.
    Returns a DataFrame with variable names.
    """
    results = {}
    
    for col in df_feature.columns:
        feature = df_feature[col]
        # Store result in dictionary
        results[col] = sequential_procedure(feature, 0.05)
        
    # Convert dictionary to DataFrame for clear display
    return pd.DataFrame(results, index=["Integration Order"])


def transform_data(feature: pd.Series, code: int) -> pd.Series:
    """
    Function to apply a given transformation depending on a feature.
    """
    list_code = [0, 1, 2, 3, 4, 5]
    if code not in list_code:
        raise Exception(f"Code {code} is not part of the proposed transformations.")
        
    match code:
        case 0:
            return feature
        case 1:
            return 100 * np.log(feature)
        case 2:
            modif_feature = np.array(np.nan)
            return pd.Series(np.append(modif_feature, 100 * np.diff(np.log(feature))), index=feature.index)
        case 3:
            modif_feature = np.array([np.nan, np.nan])
            return pd.Series(np.append(modif_feature, np.diff(np.log(feature), n=2)), index=feature.index)
        case 4:
            modif_feature = np.array(np.nan)
            return pd.Series(np.append(modif_feature, np.diff(feature)), index=feature.index)
        case 5:
            modif_feature = np.array([np.nan, np.nan])
            return pd.Series(np.append(modif_feature, np.diff(feature, n=2)), index=feature.index)


def remove_outlier(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Function removing outliers from a dataframe for a given quantile.
    """
    df_cleaned = df.copy()
    for i in range(df_cleaned.shape[1]):
        feature = df_cleaned.iloc[:, i]
        vmin = feature.quantile(q)
        vmax = feature.quantile(1 - q)
        df_cleaned.iloc[:, i] = feature.clip(lower=vmin, upper=vmax)

    return df_cleaned