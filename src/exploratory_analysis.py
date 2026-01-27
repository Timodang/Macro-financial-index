from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Tuple[pd.Series, float, float]:
    """
    Detect outliers in a time series using the Interquartile Range (IQR) method.
    
    :param series: Input time series
    :param multiplier: IQR multiplier defining the outlier threshold
    :return: Boolean series indicating outliers, lower bound, upper bound
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound


def lead_corr(df: pd.DataFrame, target: str, h: int) -> pd.Series:
    """
    Compute lead correlations corr(x_t, y_{t+h}) for all variables in a DataFrame.
    
    :param df: DataFrame containing predictors and target
    :param target: Name of the target variable
    :param h: Forecast horizon
    :return: Series of lead correlations sorted in ascending order
    """
    y_lead = df[target].shift(-h)
    X = df.drop(columns=[target], errors="ignore")

    corrs = {}
    for col in X.columns:
        aligned = pd.concat([X[col], y_lead], axis=1).dropna()
        if len(aligned) >= 10:
            corrs[col] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        else:
            corrs[col] = np.nan

    return pd.Series(corrs).dropna().sort_values()

def plot_top_corrs(
    corrs: pd.Series,
    title: str,
    top_n: int = 20,
    threshold: float = 0.3
) -> None:
    """
    Plot the strongest positive & negative correlations as a horizontal bar chart.
    
    :param corrs: Series of correlations indexed by variable names
    :param title: Plot title
    :param top_n: Total number of variables displayed
    :param threshold: Reference threshold for correlation magnitude
    :return: None
    """
    neg = corrs.head(top_n // 2)
    pos = corrs.tail(top_n - len(neg))
    selected = pd.concat([neg, pos]).sort_values()

    plt.figure(figsize=(12, 7))
    colors = ["tab:red" if v < 0 else "tab:green" for v in selected.values]
    plt.barh(selected.index, selected.values, color=colors)
    plt.axvline(0, linewidth=1)
    plt.axvline(threshold, linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(-threshold, linestyle="--", linewidth=1, alpha=0.7)
    plt.title(title)
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.show()