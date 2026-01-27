from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional, Dict, List, Literal

def create_lagged_dataframe(df: pd.DataFrame, n_lags: int, drop_na: bool = True) -> pd.DataFrame:
    """
    Create a DataFrame with original features + their lags
    """
    if n_lags < 0:
        raise ValueError(f"n_lags={n_lags} must be non-negative")
    if n_lags == 0:
        return df.copy()
    df_result = df.copy()
    for col in df.columns:
        for lag in range(1, n_lags + 1):
            lag_col_name = f"{col}_lag{lag}"
            df_result[lag_col_name] = df[col].shift(lag)
    if drop_na:
        df_result = df_result.dropna()
    return df_result

@dataclass
class EM_PCAResult:
    """
    Store EM-PCA estimation results
    """
    factors: pd.DataFrame          # T × r : Factor matrix
    loadings: pd.DataFrame         # N × r : Loading matrix
    residuals: pd.DataFrame        # T × N : Residuals
    imputed_data: pd.DataFrame     # T × N : Fully imputed panel (i.e. with added missing values)
    eigenvalues: np.ndarray        # Eigenvalues from PCA
    n_factors: int                 # Number of factors selected
    n_iterations: int              # EM convergence iterations
    convergence_error: float       # Final convergence error
    r2_by_variable: pd.Series      # R-squared for each variable
    variance_explained: float      # Total variance explained by factors

class EM_PCA:
    """
    EM-PCA with Bai-Ng Information Criterion for factor selection
    Implements EM algorithm for PCA with missing data
    Bai & Ng (2002) information criteria for optimal number of factors are used.
    """
    
    def __init__(self, kmax: int = 10, criterion: Literal['IC1', 'IC2', 'IC3'] = 'IC2',
        demean: Literal[0, 1, 2] = 2, max_iter: int = 50, tol: float = 1e-6,
        fallback_n_factors: int = 3, verbose: bool = True, n_factors: int = None
                ) -> None:
        """
        Parameters:
        -----------
        - kmax       : int
            Maximum number of factors to consider (1 to min(T, N))
        - criterion  : {'IC1', 'IC2', 'IC3'}. 
            Bai-Ng criterion selector
                - IC1 (PC_p1): most aggressive penalty
                - IC2 (PC_p2): recommended (default)
                - IC3 (PC_p3): most parsimonious
        - demean : {0, 1, 2}
                - 0: no transformation
                - 1: demean only
                - 2: demean + standardize 
        - max_iter  : int
            Maximum EM iterations
        - tol       : float
            Convergence tolerance (relative change in fitted values)
        - fallback_n_factors : int
            Used if Bai-Ng selects 0 factors
        - verbose   : bool
            Print iteration logs
        """
        self.kmax = kmax
        self.criterion = criterion
        self.demean = demean
        self.max_iter = max_iter
        self.tol = tol
        self.fallback_n_factors = fallback_n_factors
        self.verbose = verbose
        self.mean_ = None
        self.std_ = None
        self.n_factors = n_factors
        self.n_factors_ = None
        # Mapping criterion name to numeric code
        self._criterion_map = {'IC1': 1, 'IC2': 2, 'IC3': 3}
        self.loadings_ = None          
        self.factor_names_ = None      
        self.feature_names_ = None     

    def fit(self, X: pd.DataFrame) -> EM_PCAResult:
        """
        Fit EM-PCA on panel data with missing values
        
        Parameters:
        -----------
        X : pd.DataFrame (T × N)
            Panel data with NaN for missing observations
            Index: datetime (periods)
            Columns: variable names
            
        Returns:
        --------
        EM_PCAResult containing factors, loadings, diagnostics
        """
        # Validation
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        X_array = X.values
        self._check_input_matrix(X_array)
        T, N = X_array.shape
        if not (1 <= self.kmax <= min(T, N)):
            raise ValueError(f"kmax must be between 1 and min(T={T}, N={N})")
        
        # STEP 1: INITIALIZATION. 
        # Use the unconditionnal mean to initialize
        x_missing = np.isnan(X_array)
        col_mean = np.nanmean(X_array, axis=0)
        x2 = X_array.copy()
        x2[x_missing] = np.take(col_mean, np.where(x_missing)[1])
        
        # STEP 2: INITIAL TRANSFORMATION 
        # We standardize the dataset where we put the unconditionnal mean to replace NaN
        x_transformed, self.mean_, self.std_ = self._transform_data(x2)
        
        # STEP 3: SELECT NUMBER OF FACTORS VIA BAI-NG CRITERION 
        if self.n_factors is not None:
            ic_selected = self.n_factors
            if self.verbose: print(f"Using fixed n_factors = {ic_selected}")

        jj = self._criterion_map[self.criterion]
        ic_selected = self._bai_ng(x_transformed, jj)
        
        if ic_selected == 0:
            if self.verbose:
                print(f"CAUTION: Bai-Ng {self.criterion} selected 0 factors "
                      f"using fallback r={self.fallback_n_factors}")
            ic_selected = self.fallback_n_factors
        else:
            if self.verbose:
                print(f"Bai-Ng {self.criterion} selected r = {ic_selected} factors")
        
        self.n_factors_ = ic_selected
        
        # STEP 4: INITIAL PCA 
        chat0, fhat, lambdahat, eigval = self._pca(x_transformed, self.n_factors_)
        
        # STEP 5: EM ALGORITHM
        # The goal is to update missing values using values predicted by the latest set of factors.
        # We initialize a large error 
        err = 999.0
        it = 0
        
        while err > self.tol and it < self.max_iter:
            it += 1
            
            # Expectation-step: Impute missing with lambda x factor (in original scale)
            x_fitted_orig = chat0 * self.std_ + self.mean_
            x2[x_missing] = x_fitted_orig[x_missing]
            x2[~x_missing] = X_array[~x_missing]  # Keep observed values
            
            # Maximisation-step: We re-transform and re-estimate PCA on the "new" dataset
            x_transformed, _, _ = self._transform_data(x2)
            # We take the fitted values, factors, loadings, eigenvalues
            chat, fhat, lambdahat, eigval = self._pca(x_transformed, self.n_factors_)
            
            # Check convergence
            diff = chat - chat0
            err = np.sum(diff ** 2) / max(np.sum(chat0 ** 2), 1e-12)
            if self.verbose and (it % 10 == 0 or err < self.tol):
                print(f"Iteration {it:02d}: err = {err:.8f}")
            chat0 = chat
        
        if self.verbose:
            print(f"Converged in {it} iterations (final err = {err:.2e})")
        
        # STEP 6: FINAL RECONSTRUCTION AND DIAGNOSTICS
        x_fitted_orig = chat0 * self.std_ + self.mean_
        
        # Residuals (only for observed values)
        residuals = np.full_like(X_array, np.nan)
        residuals[~x_missing] = X_array[~x_missing] - x_fitted_orig[~x_missing]
        
        # R-squared by variable
        r2_by_var = self._compute_r2(X_array, x_fitted_orig, x_missing)
        
        # Total variance explained
        resid = x_transformed - chat0
        variance_explained = 1.0 - (np.mean(resid**2) / np.mean(x_transformed**2))
        
        factors_df = pd.DataFrame(
            fhat, index=X.index, columns=[f"F{i+1}" for i in range(self.n_factors_)]
        )
        loadings_df = pd.DataFrame(
            lambdahat, index=X.columns, columns=[f"F{i+1}" for i in range(self.n_factors_)]
        )

        self.loadings_ = loadings_df.copy()
        self.feature_names_ = list(X.columns)
        self.factor_names_ = list(factors_df.columns)  

        # STEP 7: RETURN RESULTS
        return EM_PCAResult(
            factors=pd.DataFrame(
                fhat,
                index=X.index,
                columns=[f'F{i+1}' for i in range(self.n_factors_)]
            ),
            loadings=pd.DataFrame(
                lambdahat,
                index=X.columns,
                columns=[f'F{i+1}' for i in range(self.n_factors_)]
            ),
            residuals=pd.DataFrame(
                residuals,
                index=X.index,
                columns=X.columns
            ),
            imputed_data=pd.DataFrame(
                x2,
                index=X.index,
                columns=X.columns
            ),
            eigenvalues=eigval,
            n_factors=self.n_factors_,
            n_iterations=it,
            convergence_error=err,
            r2_by_variable=pd.Series(r2_by_var, index=X.columns),
            variance_explained=variance_explained
        )

    def transform(
        self,
        X: pd.DataFrame,
        em: bool = False,
        max_iter: int = 50,
        tol: float = 1e-6,
        return_imputed: bool = False,
        return_fitted: bool = False
    ):
        """
        Project new data onto fitted factors (out-of-sample).

        Parameters
        ----------
        X : pd.DataFrame (T_new × N)
        em : bool, default False
            If True, perform EM on X with FIXED loadings (no re-estimation).
        max_iter : int
            Max EM iterations (used only if em=True)
        tol : float
            EM convergence tolerance (used only if em=True)
        return_imputed : bool
            If True, return imputed X (original scale)
        return_fitted : bool
            If True, return fitted X = F Lambda' (original scale)

        Returns
        -------
        F_new : pd.DataFrame
        [X_imputed] : pd.DataFrame (optional)
        [X_fitted]  : pd.DataFrame (optional)
        """

        # Checks 
        if self.n_factors_ is None or self.loadings_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if self.feature_names_ is None:
            raise ValueError("feature_names_ missing. Re-fit the model.")

        # Align columns
        missing_cols = set(self.feature_names_) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_names_)
        if missing_cols:
            raise ValueError(f"X missing columns seen in fit(): {sorted(missing_cols)[:10]}")
        if extra_cols:
            raise ValueError(f"X has extra columns not seen in fit(): {sorted(extra_cols)[:10]}")

        X_aligned = X.loc[:, self.feature_names_]

        X0 = X_aligned.values.astype(float, copy=True)
        miss = np.isnan(X0)

        mu = self.mean_[0, :]
        sd = self.std_[0, :]
        Lambda = self.loadings_.values  
        N = Lambda.shape[0]

        # Helpers: same transform as in fit
        def _to_transformed(X_orig):
            if self.demean == 0:
                return X_orig
            elif self.demean == 1:
                return X_orig - mu
            elif self.demean == 2:
                return (X_orig - mu) / sd
            else:
                raise ValueError("demean must be 0, 1, or 2")

        def _to_original(X_tr):
            if self.demean == 0:
                return X_tr
            elif self.demean == 1:
                return X_tr + mu
            elif self.demean == 2:
                return X_tr * sd + mu
            else:
                raise ValueError("demean must be 0, 1, or 2")

        # Initial imputation 
        X_imp = X0.copy()
        if miss.any():
            X_imp[miss] = np.take(mu, np.where(miss)[1])

        # CASE 1: simple projection
        if not em:
            X_tr = _to_transformed(X_imp)
            F = (X_tr @ Lambda) / N
            Xhat_tr = F @ Lambda.T
            Xhat = _to_original(Xhat_tr)

        # CASE 2: EM with fixed loadings 
        else:
            err = np.inf
            it = 0

            while it < max_iter and err > tol:
                it += 1

                # M-step: factor scores
                X_tr = _to_transformed(X_imp)
                F = (X_tr @ Lambda) / N
                Xhat_tr = F @ Lambda.T

                # E-step: update missing values only
                X_tr_new = X_tr.copy()
                if miss.any():
                    X_tr_new[miss] = Xhat_tr[miss]

                X_imp_new = _to_original(X_tr_new)

                if miss.any():
                    num = np.sum((X_imp_new[miss] - X_imp[miss]) ** 2)
                    den = max(np.sum(X_imp[miss] ** 2), 1e-12)
                    err = num / den
                else:
                    err = 0.0

                X_imp = X_imp_new

            # final reconstruction
            X_tr = _to_transformed(X_imp)
            F = (X_tr @ Lambda) / N
            Xhat_tr = F @ Lambda.T
            Xhat = _to_original(Xhat_tr)

        # Outputs
        factor_cols = (
            self.factor_names_
            if self.factor_names_ is not None
            else [f"F{i+1}" for i in range(Lambda.shape[1])]
        )

        F_df = pd.DataFrame(F, index=X_aligned.index, columns=factor_cols)

        outputs = [F_df]

        if return_imputed:
            outputs.append(pd.DataFrame(X_imp, index=X_aligned.index, columns=self.feature_names_))
        if return_fitted:
            outputs.append(pd.DataFrame(Xhat, index=X_aligned.index, columns=self.feature_names_))

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # ==================== PRIVATE METHODS ====================
    
    def _check_input_matrix(self, x: np.ndarray) -> None:
        """
        Validate input matrix
        """
        if x.ndim != 2:
            raise ValueError("X must be a 2D array (T × N)")
        if np.all(np.isnan(x), axis=1).any():
            raise ValueError("At least one row is fully missing")
        if np.all(np.isnan(x), axis=0).any():
            raise ValueError("At least one column is fully missing")
    
    def _transform_data(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply demeaning and/or standardization
        
        Returns:
        --------
        x_transformed, mean_array, std_array
        """
        T, N = x.shape
        
        col_mean = np.nanmean(x, axis=0)
        col_std = np.nanstd(x, axis=0, ddof=0)
        col_std = np.where(col_std == 0.0, 1.0, col_std)  # Avoid division by 0
        
        # Broadcast to (T, N) for easy reversal
        mean_array = np.tile(col_mean, (T, 1))
        std_array = np.tile(col_std, (T, 1))
        
        if self.demean == 0:
            x_transformed = x.copy()
        elif self.demean == 1:
            x_transformed = x - mean_array
        elif self.demean == 2:
            x_transformed = (x - mean_array) / std_array
        else:
            raise ValueError("demean must be 0, 1, or 2")
        
        return x_transformed, mean_array, std_array
    
    def _pca(
        self,
        X: np.ndarray,
        nfac: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        PCA via spectral decomposition of X'X
        Parameters: 
        --------
            - X: transformed data (standardized + after EM)
            - nfac: number of factors to conserve (via Bai-Ng or forced by the user)
        
        Returns:
        --------
            - chat: fitted values F @ Lambda' (explained part)
            - fhat: estimated factors 
            - lambdahat: estimated loadings
            - s: all eigenvalues from decomposition
        """
        T, N = X.shape
        
        XtX = X.T @ X
        U, s, _ = np.linalg.svd(XtX, full_matrices=False)
        
        # Stock & Watson (2002) normalization
        lambdahat = U[:, :nfac] * np.sqrt(N)      # N × r
        fhat = (X @ lambdahat) / N                # T × r
        chat = fhat @ lambdahat.T                 # T × N (fitted values)
        
        return chat, fhat, lambdahat, s
    
    def _bai_ng(self, X: np.ndarray, jj: int) -> int:
        """
        Bai-Ng information criterion
        
        Parameters:
        -----------
        X : np.ndarray (T × N)
            Transformed data (no NaN)
        jj : {1, 2, 3}
            Criterion type
            
        Returns:
        --------
        ic_optimal : int
            Optimal number of factors (0 if r=0 selected)
        """
        if np.isnan(X).any():
            raise ValueError("Bai-Ng requires fully observed matrix")
        
        T, N = X.shape
        NT = N * T
        NT1 = N + T
        GCT = min(N, T)
        
        # Penalty terms
        ii = np.arange(1, self.kmax + 1, dtype=float)
        
        if jj == 1:
            CT = np.log(NT / NT1) * ii * (NT1 / NT)
        elif jj == 2:
            CT = (NT1 / NT) * np.log(GCT) * ii
        elif jj == 3:
            CT = ii * np.log(GCT) / GCT
        else:
            raise ValueError("jj must be 1, 2, or 3")
    
        if T < N:
            M = X @ X.T
        else:
            M = X.T @ X
        
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        
        # Compute factors and loadings
        if T < N:
            Fhat0 = np.sqrt(T) * U
            Lambda0 = (X.T @ Fhat0) / T
        else:
            Lambda0 = np.sqrt(N) * U
            Fhat0 = (X @ Lambda0) / N
        
        # Information criteria
        IC = np.full(self.kmax + 1, np.nan)
        
        for k in range(1, self.kmax + 1):
            Fhat_k = Fhat0[:, :k]
            Lambda_k = Lambda0[:, :k]
            
            chat_k = Fhat_k @ Lambda_k.T
            residuals = X - chat_k
            
            # Variance of residuals
            V_k = np.mean(residuals ** 2)
            
            IC[k - 1] = np.log(V_k) + CT[k - 1]
        
        # IC for r=0 (no factors)
        V_0 = np.mean(X ** 2)
        IC[self.kmax] = np.log(V_0)
        ic_optimal_idx = np.argmin(IC)
        
        if self.verbose:
            # debug 
            print(f"T={T}, N={N}, kmax={self.kmax}")
            print(f"Penalty coef: (NT1/NT)*log(GCT) = {(NT1/NT)*np.log(GCT):.6f}\n")
    
            print(f"{'r':>3} | {'V(r)':>12} | {'log(V)':>10} | {'CT':>10} | {'IC':>12} | {'Min?':>5}")
            print(f"{'-'*65}")
    
            for i in range(len(IC)):
                is_min = '  ✓' if i == ic_optimal_idx else ''
        
                if i < self.kmax:  # r = 1 to kmax
                    r_val = i + 1
                    Fhat_k = Fhat0[:, :r_val]
                    Lambda_k = Lambda0[:, :r_val]
                    chat_k = Fhat_k @ Lambda_k.T
                    V_k = np.mean((X - chat_k) ** 2)
            
                    print(f"{r_val:3d} | {V_k:12.8f} | {np.log(V_k):10.6f} | "
                  f"{CT[i]:10.6f} | {IC[i]:12.8f} |{is_min}")
                else:  # r = 0
                    V_0 = np.mean(X ** 2)
                    print(f"  0 | {V_0:12.8f} | {np.log(V_0):10.6f} | "
                  f"{'0.000000':>10} | {IC[i]:12.8f} |{is_min}")
    
            print(f"{'-'*65}")
            print(f"Selected: r = {ic_optimal_idx + 1 if ic_optimal_idx < self.kmax else 0}")
            print(f"{'='*70}\n")

        if ic_optimal_idx == self.kmax:
            return 0
        else:
            return ic_optimal_idx + 1
    
    def _compute_r2(
        self,
        X_true: np.ndarray,
        X_fitted: np.ndarray,
        missing_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute R-squared for each variable (on observed values only)
        
        Returns:
        --------
        r2_array : np.ndarray (N,)
        """
        N = X_true.shape[1]
        r2 = np.zeros(N)
        
        for i in range(N):
            observed = ~missing_mask[:, i]
            
            if observed.sum() == 0:
                r2[i] = np.nan
                continue
            
            y_true = X_true[observed, i]
            y_fitted = X_fitted[observed, i]
            
            ss_res = np.sum((y_true - y_fitted) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            
            r2[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return r2
    

@dataclass
class RollingRegressionResult:
    """Store rolling regression results"""
    dates: pd.DatetimeIndex
    alpha: pd.Series
    betas: pd.DataFrame
    fitted: pd.Series
    r2: pd.Series

class RollingRegression:
    """
    Implements rolling linear regression with optional factor extraction.
    """
    def __init__(
        self, 
        y: pd.Series, 
        X: pd.DataFrame, 
        window: int,
        type_window: str = "expanding",
        extract_factors: bool = False,
        empca_params: dict = None
    ) -> None:
        """
        Parameters:
        -----------
        y : pd.Series 
            Dependent variable
        X : pd.DataFrame 
            Independent variables (raw data or pre-computed factors)
        window : int 
            Rolling window size
        extract_factors : bool, default False
            If True, extract factors from X in each rolling window using EM-PCA
            If False, use X directly as regressors
        empca_params : dict, optional
            Parameters for EM_PCA when extract_factors=True
            Example: {'kmax': 10, 'criterion': 'IC2', 'n_factors': 3, 'demean': 2}
        """
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length: {len(y)} != {len(X)}")
        if window >= len(y):
            raise ValueError(f"window ({window}) must be < data length ({len(y)})")
        
        self.y = y
        self.X = X
        self.window = window
        self.T = len(y)
        self.extract_factors = extract_factors
        
        # Default EM-PCA params
        self.empca_params = empca_params or {
            'kmax': 10,
            'criterion': 'IC2',
            'demean': 2,
            'max_iter': 50,
            'tol': 1e-6,
            'fallback_n_factors': 3,
            'verbose': False,
            'n_factors': None  # None = Bai-Ng, int = fixed
        }
        
        self.result_ = None
        self.factor_info_ = None  
        self.type_window = type_window
    
    def fit(self) -> RollingRegressionResult:
        """
        Fit rolling OLS regression.
        
        If extract_factors=True:
            1. For each window, estimate factors via EM-PCA
            2. Use last period's factors as regressors for OOS prediction
        
        If extract_factors=False:
            1. Use X directly as regressors
        
        Returns:
        --------
        RollingRegressionResult with dates, coefficients, predictions, R2
        """
        dates, alphas, betas, fitted, r2s = [], [], [], [], []
        # case the window chosen is not rolling or expanding
        if self.type_window != "rolling" and self.type_window != "expanding":
            raise ValueError(f"The regression for a {self.type_window} window is not implemented")
        # Track factor info if extracting
        if self.extract_factors:
            factor_info = {
                'n_factors': [],
                'variance_explained': [],
                'loadings': {},
                'factors': {}
            }
        
        for t in range(self.window, self.T):
            # case for a rolling window
            if self.type_window == "rolling":
                y_window = self.y.iloc[t - self.window : t]
                X_window = self.X.iloc[t - self.window : t]
            else:
                y_window = self.y.iloc[0:t]
                X_window = self.X.iloc[0:t]
            
            # ========== FACTOR EXTRACTION ==========
            if self.extract_factors:
                # Fit EM-PCA on this window
                empca = EM_PCA(**self.empca_params)
                result_pca = empca.fit(X_window)
                
                # Use factors as regressors
                F_window = result_pca.factors  # (window × r)
                
                # Store info
                factor_info['n_factors'].append(result_pca.n_factors)
                factor_info['variance_explained'].append(result_pca.variance_explained)
                factor_info['loadings'][self.y.index[t-1]] = result_pca.loadings.copy()
                factor_info['factors'][self.y.index[t-1]] = result_pca.factors.copy()
                
                # OOS factor: project next period's data
                X_oos_raw = self.X.iloc[[t]]  # Next period (1 × N)
                F_oos = empca.transform(X_oos_raw, em=True)  # (1 × r)
                
            else:
                # Use X directly
                F_window = X_window
                F_oos = self.X.iloc[[t]]
            
            # ========== OLS REGRESSION ==========
            # Add constant and fit
            F_window_c = sm.add_constant(F_window)
            model = sm.OLS(y_window, F_window_c).fit()
            
            # Predict at time t (OOS)
            F_oos_values = F_oos.iloc[0].values  # shape (K,)
            F_oos_c = np.concatenate([[1.0], F_oos_values])  # shape (K+1,) with constant
            y_pred = np.dot(F_oos_c, model.params)
            
            # Store results
            dates.append(self.y.index[t])
            alphas.append(model.params[0])
            betas.append(model.params[1:].values)
            fitted.append(y_pred)
            r2s.append(model.rsquared)
        
        # Convert to pandas objects
        dates_idx = pd.DatetimeIndex(dates)
        
        # Determine column names for betas
        if self.extract_factors and len(factor_info['n_factors']) > 0:
            # Use max number of factors across windows
            max_r = max(factor_info['n_factors'])
            beta_cols = [f"F{i+1}" for i in range(max_r)]
            
            # Pad betas with NaN if factor count varies
            betas_padded = []
            for i, beta_array in enumerate(betas):
                r_i = factor_info['n_factors'][i]
                if r_i < max_r:
                    beta_padded = np.concatenate([beta_array, [np.nan] * (max_r - r_i)])
                else:
                    beta_padded = beta_array
                betas_padded.append(beta_padded)
            betas = betas_padded
        else:
            beta_cols = self.X.columns
        
        self.result_ = RollingRegressionResult(
            dates=dates_idx,
            alpha=pd.Series(alphas, index=dates_idx, name='alpha'),
            betas=pd.DataFrame(betas, index=dates_idx, columns=beta_cols),
            fitted=pd.Series(fitted, index=dates_idx, name='indicator_raw'),
            r2=pd.Series(r2s, index=dates_idx, name='r2_train')
        )
        
        # Store factor info
        if self.extract_factors:
            self.factor_info_ = {
                'n_factors': pd.Series(
                    factor_info['n_factors'], 
                    index=dates_idx, 
                    name='n_factors'
                ),
                'variance_explained': pd.Series(
                    factor_info['variance_explained'], 
                    index=dates_idx, 
                    name='variance_explained'
                ),
                'loadings': factor_info['loadings'],
                'factors': factor_info['factors']
            }
        
        return self.result_
    
    def summary(self) -> None:
        """Print summary statistics of rolling regression"""
        if self.result_ is None:
            raise ValueError("Must call fit() before summary()")
        
        print(f"\n{'='*60}")
        print(f"ROLLING REGRESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Window size      : {self.window}")
        print(f"Total periods    : {self.T}")
        print(f"OOS predictions  : {len(self.result_.dates)}")
        print(f"Mean train R²    : {self.result_.r2.mean():.4f}")
        print(f"Std train R²     : {self.result_.r2.std():.4f}")
        
        if self.extract_factors and self.factor_info_ is not None:
            print(f"\nFACTOR EXTRACTION:")
            print(f"  Mean # factors : {self.factor_info_['n_factors'].mean():.1f}")
            print(f"  Mean var. expl.: {self.factor_info_['variance_explained'].mean():.2%}")
        
        print(f"\nLast window coefficients:")
        print(f"  Alpha          : {self.result_.alpha.iloc[-1]:.4f}")
        for col, val in zip(self.result_.betas.columns, self.result_.betas.iloc[-1]):
            if not np.isnan(val):
                print(f"  {col:12s}   : {val:8.4f}")
        print(f"{'='*60}\n")




@dataclass
class RollingResult:
    """Container for rolling regression results"""
    dates: pd.DatetimeIndex
    y_true: pd.Series
    y_pred: pd.Series
    err: pd.Series
    rmse_cum: pd.Series
    rmse_roll: Optional[pd.Series] = None
    coefficients: Optional[pd.DataFrame] = None
    has_missing_by_window: pd.Series = None
    n_nonzero_coefs: pd.Series = None
    additional_stats: Optional[Dict] = None  # For model-specific statistics


class BaseRollingRegressor(ABC):
    """
    Abstract base class for rolling/expanding window regression with EM imputation.
    
    Common functionality:
    - Window management (rolling vs expanding)
    - Missing data imputation (EM-PCA, median, forward fill)
    - Standardization
    - Performance metrics (RMSE cumulative and rolling)
    - Results storage
    
    Subclasses must implement:
    - _fit_model(): Fit the specific model
    - _get_model_name(): Return model name for printing
    """
    
    def __init__(
        self,
        window: int = 36,
        window_type: Literal['rolling', 'expanding'] = 'rolling',
        n_lags: int = 0,
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter_em: int = 50,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        window : int
            Initial window size (for rolling) or minimum window size (for expanding)
        window_type : str
            'rolling' : Fixed-size window that slides forward
            'expanding' : Window grows from initial size to include all past data
        imputation_method : str
            Method to handle missing values: 'em', 'median', 'forward_fill', 'none'
        n_factors_imputation : int
            Number of factors for EM-PCA imputation
        max_iter_em : int
            Maximum iterations for EM algorithm
        rmse_window : int, optional
            Window size for rolling RMSE computation
        store_coefs : bool
            Whether to store coefficients at each time step
        verbose : bool
            Print progress information
        """
        self.window = window
        self.window_type = window_type
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter_em = max_iter_em
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_: Optional[RollingResult] = None
        self.n_lags = n_lags 
        if n_lags < 0: 
            raise ValueError("n_lags must be >=0")
        
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================
    
    @abstractmethod
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit the specific model.
        
        Parameters
        ----------
        X_train : np.ndarray (n, p)
            Standardized training features
        y_train : np.ndarray (n,)
            Training target
            
        Returns
        -------
        coef : np.ndarray (p,)
            Model coefficients
        intercept : float
            Model intercept
        """
        pass
    
    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name for display"""
        pass
    
    # ========================================================================
    # COMMON METHODS - Shared across all models
    # ========================================================================
    
    def _has_missing(self, X: pd.DataFrame) -> bool:
        """Check if DataFrame contains missing values"""
        return X.isna().any().any()
    
    def _impute_median(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with median"""
        return X.fillna(X.median())
    
    def _impute_forward_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with forward fill + median fallback"""
        X_filled = X.ffill()
        return X_filled.fillna(X_filled.median())
    
    def _impute_em(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        """Impute missing values using EM-PCA"""
        empca = EM_PCA(
            kmax=self.n_factors_imputation,
            criterion='IC2',
            demean=2,
            max_iter=self.max_iter_em,
            verbose=False,
            fallback_n_factors=min(3, self.n_factors_imputation)
        )
        result = empca.fit(X)
        return result.imputed_data, empca
    
    def _impute_data(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Impute missing values in train and test sets.
        
        For test set: uses transformation from train imputation model
        """
        has_missing_train = self._has_missing(X_train)
        has_missing_test = X_test is not None and self._has_missing(X_test)
        
        if self.imputation_method == 'none':
            if has_missing_train or has_missing_test:
                raise ValueError("Missing values detected but imputation_method='none'")
            return X_train, X_test
        
        # Impute train set
        if not has_missing_train:
            X_train_imputed = X_train.copy()
            empca_model = None
        else:
            if self.imputation_method == 'em':
                X_train_imputed, empca_model = self._impute_em(X_train)
            elif self.imputation_method == 'median':
                X_train_imputed = self._impute_median(X_train)
                empca_model = None
            elif self.imputation_method == 'forward_fill':
                X_train_imputed = self._impute_forward_fill(X_train)
                empca_model = None
            else:
                raise ValueError(f"Unknown imputation method: {self.imputation_method}")
        
        # Impute test set using train statistics
        if X_test is None or not has_missing_test:
            X_test_imputed = X_test
        else:
            if self.imputation_method == 'em' and empca_model is not None:
                _, X_test_imputed = empca_model.transform(X_test, em=True, return_imputed=True)
            elif self.imputation_method == 'median':
                X_test_imputed = X_test.fillna(X_train.median())
            elif self.imputation_method == 'forward_fill':
                X_test_imputed = X_test.ffill()
                X_test_imputed = X_test_imputed.fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def _get_train_window(self, t: int, T: int) -> Tuple[int, int]:
        """
        Get training window indices based on window_type.
        
        Returns
        -------
        start_idx : int
            Start of training window
        end_idx : int
            End of training window (exclusive)
        """
        if self.window_type == 'rolling':
            # Fixed-size rolling window
            start_idx = max(0, t - self.window)
            end_idx = t
        else:  # expanding
            # Growing window from beginning
            start_idx = 0
            end_idx = t
            
            # Optionally enforce minimum window size
            if end_idx - start_idx < self.window:
                start_idx = max(0, end_idx - self.window)
        
        return start_idx, end_idx
    
    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """Compute error and RMSE metrics"""
        err = y_true - y_pred
        err.name = "error"
        
        # Cumulative RMSE
        rmse_cum = np.sqrt((err ** 2).expanding().mean())
        rmse_cum.name = "rmse_cum"
        
        # Rolling RMSE
        rmse_roll = None
        if self.rmse_window is not None and self.rmse_window >= 2:
            rmse_roll = np.sqrt((err ** 2).rolling(self.rmse_window).mean())
            rmse_roll.name = f"rmse_roll_{self.rmse_window}"
        
        return err, rmse_cum, rmse_roll
    
    def fit(self, y: pd.Series, X: pd.DataFrame) -> RollingResult:
        """
        Main fitting loop: iterate through time, fit model, make predictions.
        
        Parameters
        ----------
        y : pd.Series
            Target variable with DatetimeIndex
        X : pd.DataFrame
            Features with DatetimeIndex matching y
            
        Returns
        -------
        result : RollingResult
            Container with predictions, errors, coefficients, etc.
        """
        # Validation
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length: {len(y)} != {len(X)}")
        if self.window >= len(y):
            raise ValueError(f"window ({self.window}) must be < data length ({len(y)})")
        
        # Combine and align
        df = pd.concat([y.rename("y"), X], axis=1)
        y_al = df["y"]
        X_al = df.drop(columns=["y"])
        T = len(df)
        
        # Storage
        y_pred_list = []
        y_true_list = []
        dates_list = []
        has_missing_list = []
        n_nonzero_list = []
        coefs_list = [] if self.store_coefs else None
        additional_stats_list = []
        
        # Main loop
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
            
            # Get train window
            start_idx, end_idx = self._get_train_window(t, T)
            
            X_train = X_al.iloc[start_idx:end_idx]
            y_train = y_al.iloc[start_idx:end_idx]
            X_test = X_al.iloc[[t]]
            y_test = y_al.iloc[t]
            
            # Track missing values
            has_missing = self._has_missing(X_train) or self._has_missing(X_test)
            has_missing_list.append(has_missing)
            
            # Impute
            try:
                X_train_imputed, X_test_imputed = self._impute_data(X_train, X_test)
            except Exception as e:
                if self.verbose:
                    print(f"  WARNING: Imputation failed at t={t}: {e}")
                continue
            
            # Add lags
            if self.n_lags > 0:
                X_all = pd.concat([X_train_imputed, X_test_imputed], axis=0)
                X_all_lag = create_lagged_dataframe(X_all, n_lags=self.n_lags, drop_na=False)
                X_train_lag = X_all_lag.iloc[:-1].copy()
                X_test_lag = X_all_lag.iloc[[-1]].copy()
                # Drop NA caused by lags (train side)
                X_train_lag = X_train_lag.dropna()

                # if test still has NA -> not enough history, skip this t
                if X_test_lag.isna().any(axis=1).iloc[0]:
                    if self.verbose:
                        print(f"  WARNING: Not enough history for n_lags={self.n_lags} at t={t}, skipping")
                    continue

                # align y with surviving X rows
                y_train_al = y_train.loc[X_train_lag.index]
                X_train_al = X_train_lag
                X_test_al = X_test_lag

            else:
                common_idx = y_train.index.intersection(X_train_imputed.index)
                y_train_al = y_train.loc[common_idx]
                X_train_al = X_train_imputed.loc[common_idx]
                X_test_al = X_test_imputed
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
            X_test_scaled = scaler.transform(X_test_al.values)
            
            # Fit model (subclass-specific)
            try:
                coef, intercept = self._fit_model(X_train_scaled, y_train_al.values)
            except Exception as e:
                if self.verbose:
                    print(f"  WARNING: Model fitting failed at t={t}: {e}")
                continue
            
            # Predict
            y_pred = intercept + np.dot(X_test_scaled, coef)[0]
            
            # Store results
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
            
            n_nonzero = np.sum(np.abs(coef) > 1e-8)
            n_nonzero_list.append(n_nonzero)
            
            if self.store_coefs:
                coef_dict = {'intercept': intercept}
                feat_cols = list(X_train_al.columns)   # includes lag columns
                coef_dict.update({col: c for col, c in zip(feat_cols, coef)})
                coefs_list.append(coef_dict)
            
            # Store model-specific statistics
            stats = self._get_additional_stats(coef, intercept)
            additional_stats_list.append(stats)
        
        # Build result object
        dates_idx = pd.DatetimeIndex(dates_list)
        y_true = pd.Series(y_true_list, index=dates_idx, name="y_true")
        y_pred = pd.Series(y_pred_list, index=dates_idx, name="y_pred")
        
        err, rmse_cum, rmse_roll = self._compute_metrics(y_true, y_pred)
        
        has_missing_series = pd.Series(has_missing_list, index=dates_idx, name="has_missing")
        n_nonzero_series = pd.Series(n_nonzero_list, index=dates_idx, name="n_nonzero_coefs")
        
        coefs_df = None
        if self.store_coefs:
            coefs_df = pd.DataFrame(coefs_list, index=dates_idx)
        
        self.result_ = RollingResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=coefs_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=n_nonzero_series,
            additional_stats=self._aggregate_additional_stats(additional_stats_list, dates_idx)
        )
        
        if self.verbose:
            self._print_summary()
        
        return self.result_
    
    def _get_additional_stats(self, coef: np.ndarray, intercept: float) -> Dict:
        """
        Hook for subclasses to compute model-specific statistics.
        
        Override in subclass if needed.
        """
        return {}
    
    def _aggregate_additional_stats(self, stats_list: List[Dict], dates: pd.DatetimeIndex) -> Dict:
        """
        Aggregate model-specific statistics into Series.
        
        Override in subclass if needed.
        """
        if not stats_list or not stats_list[0]:
            return {}
        
        result = {}
        for key in stats_list[0].keys():
            result[key] = pd.Series([s[key] for s in stats_list], index=dates, name=key)
        return result
    
    def _print_summary(self):
        """Print summary statistics"""
        res = self.result_
        print(f"\n{'='*70}")
        print(f"{self._get_model_name().upper()} - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Window type          : {self.window_type}")
        print(f"Imputation method    : {self.imputation_method}")
        print(f"N° of predictions    : {len(res.dates)}")
        
        self._print_model_specific_summary()
        
        print(f"\nVariable selection:")
        print(f"  Mean n° non-zero   : {res.n_nonzero_coefs.mean():.1f}")
        print(f"  Std n° non-zero    : {res.n_nonzero_coefs.std():.1f}")
        print(f"  Min - Max          : {res.n_nonzero_coefs.min()} - {res.n_nonzero_coefs.max()}")
        
        print(f"\nMissing data:")
        pct_missing = 100 * res.has_missing_by_window.sum() / len(res.has_missing_by_window)
        print(f"  Windows with missing: {pct_missing:.1f}% ({res.has_missing_by_window.sum()} / {len(res.dates)})")
        
        print(f"\nPerformance (out-of-sample):")
        print(f"  Final RMSE (cum)   : {res.rmse_cum.iloc[-1]:.4f}")
        print(f"  Mean absolute error: {res.err.abs().mean():.4f}")
        print(f"  Std of errors      : {res.err.std():.4f}")
        print(f"{'='*70}\n")
    
    def _print_model_specific_summary(self):
        """Hook for model-specific summary info. Override in subclass."""
        pass
    
    def predict(self) -> pd.Series:
        """Return predictions"""
        if self.result_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.result_.y_pred


# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

class RollingLasso(BaseRollingRegressor):
    """Lasso regression with L1 penalty"""
    
    def __init__(
        self,
        alpha: float = 0.01,
        max_iter: int = 10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.max_iter = max_iter
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
        lasso = Lasso(
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=self.max_iter,
            random_state=42
        )
        lasso.fit(X_train, y_train)
        return lasso.coef_, lasso.intercept_
    
    def _get_model_name(self) -> str:
        return f"Lasso (α={self.alpha})"
    
    def _print_model_specific_summary(self):
        print(f"Alpha (Lasso)        : {self.alpha}")


class RollingRidge(BaseRollingRegressor):
    """Ridge regression with L2 penalty"""
    
    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.max_iter = max_iter
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
        ridge = Ridge(
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=self.max_iter,
            random_state=42
        )
        ridge.fit(X_train, y_train)
        return ridge.coef_, ridge.intercept_
    
    def _get_model_name(self) -> str:
        return f"Ridge (α={self.alpha})"
    
    def _print_model_specific_summary(self):
        print(f"Alpha (Ridge)        : {self.alpha}")
        print(f"  (Ridge keeps all variables, but shrinks coefficients)")


class RollingElasticNet(BaseRollingRegressor):
    """Elastic Net with L1 + L2 penalty"""
    
    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
        elasticnet = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            max_iter=self.max_iter,
            random_state=42
        )
        elasticnet.fit(X_train, y_train)
        return elasticnet.coef_, elasticnet.intercept_
    
    def _get_model_name(self) -> str:
        return f"Elastic Net (α={self.alpha}, l1_ratio={self.l1_ratio})"
    
    def _print_model_specific_summary(self):
        print(f"Alpha (overall)      : {self.alpha}")
        print(f"L1 ratio             : {self.l1_ratio} ({self.l1_ratio*100:.0f}% Lasso, {(1-self.l1_ratio)*100:.0f}% Ridge)")


class RollingAdaptiveLasso(BaseRollingRegressor):
    """
    Adaptive Lasso with adaptive weights based on initial estimation.
    
    Two-step procedure:
    1. Initial Ridge/Lasso/OLS to get β̂
    2. Adaptive Lasso with weights wⱼ = 1/|β̂ⱼ|^γ
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 1.0,
        initial_estimator: Literal['ridge', 'lasso', 'ols'] = 'ridge',
        alpha_initial: float = 1.0,
        max_iter: int = 10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.initial_estimator = initial_estimator
        self.alpha_initial = alpha_initial
        self.max_iter = max_iter
        self._weights_history = []
    
    def _get_initial_weights(self, X_scaled: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute adaptive weights from initial estimation"""
        if self.initial_estimator == 'ridge':
            model = Ridge(alpha=self.alpha_initial, fit_intercept=False)
            model.fit(X_scaled, y)
            beta_init = model.coef_
        elif self.initial_estimator == 'lasso':
            model = Lasso(alpha=self.alpha_initial, fit_intercept=False, max_iter=self.max_iter)
            model.fit(X_scaled, y)
            beta_init = model.coef_
        elif self.initial_estimator == 'ols':
            try:
                beta_init = np.linalg.lstsq(X_scaled, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback to Ridge
                model = Ridge(alpha=0.1, fit_intercept=False)
                model.fit(X_scaled, y)
                beta_init = model.coef_
        
        # wⱼ = 1/|βⱼ|^γ
        epsilon = 1e-6
        weights = 1.0 / (np.abs(beta_init) + epsilon) ** self.gamma
        return weights
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
        # Step 1: Get adaptive weights
        weights = self._get_initial_weights(X_train, y_train)
        self._weights_history.append(weights)
        
        # Step 2: Weighted Lasso (rescale features)
        X_train_weighted = X_train / weights
        
        lasso = Lasso(
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=self.max_iter,
            random_state=42
        )
        lasso.fit(X_train_weighted, y_train)
        
        # Recover true coefficients
        coef_adaptive = lasso.coef_ / weights
        
        return coef_adaptive, lasso.intercept_
    
    def _get_model_name(self) -> str:
        return f"Adaptive Lasso (α={self.alpha}, γ={self.gamma})"
    
    def _print_model_specific_summary(self):
        print(f"Alpha                : {self.alpha}")
        print(f"Gamma (weight power) : {self.gamma}")
        print(f"Initial estimator    : {self.initial_estimator}")


class RollingGroupLasso(BaseRollingRegressor):
    """
    Group Lasso with group-level sparsity.
    
    Minimizes: ||y - Xβ||² + λ Σ_g √|G_g| ||β_g||₂
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        groups: Optional[Dict[str, List[str]]] = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.groups = groups
        self.max_iter = max_iter
        self.tol = tol
        self._n_active_groups_history = []
        self._feature_names = None
    
    def _setup_groups(self, feature_names: List[str]) -> Dict[str, List[int]]:
        """Convert group definitions to indices"""
        if self.groups is None:
            # Each variable is its own group
            return {var: [i] for i, var in enumerate(feature_names)}
        
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        group_indices = {}
        
        for group_name, var_names in self.groups.items():
            indices = [name_to_idx[var] for var in var_names if var in name_to_idx]
            if indices:
                group_indices[group_name] = indices
        
        # Add ungrouped variables as singletons
        grouped_indices = set()
        for indices in group_indices.values():
            grouped_indices.update(indices)
        
        for i, name in enumerate(feature_names):
            if i not in grouped_indices:
                group_indices[f"_singleton_{name}"] = [i]
        
        return group_indices
    
    def _fit_group_lasso_blockwise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_indices: Dict[str, List[int]]
    ) -> np.ndarray:
        """Fit Group Lasso via block coordinate descent"""
        n, p = X.shape
        beta = np.zeros(p)
        
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            
            for group_name, group_idx in group_indices.items():
                group_idx = np.array(group_idx)
                
                # Partial residual
                r = y - X @ beta + X[:, group_idx] @ beta[group_idx]
                z = X[:, group_idx].T @ r
                
                # Group soft-thresholding
                z_norm = np.linalg.norm(z)
                group_size = len(group_idx)
                threshold = self.alpha * np.sqrt(group_size)
                
                if z_norm <= threshold:
                    beta[group_idx] = 0
                else:
                    # Block update
                    XgXg = X[:, group_idx].T @ X[:, group_idx]
                    try:
                        beta_g = np.linalg.solve(XgXg + 1e-6 * np.eye(group_size), z)
                        beta_g = beta_g * (1 - threshold / z_norm)
                        beta[group_idx] = beta_g
                    except np.linalg.LinAlgError:
                        beta[group_idx] = 0
            
            if np.linalg.norm(beta - beta_old) < self.tol:
                break
        
        return beta
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
        # Setup groups (use stored feature names from fit())
        if self._feature_names is None:
            raise ValueError("Feature names not set. This should not happen.")
        
        group_indices = self._setup_groups(self._feature_names)
        
        # Fit Group Lasso
        beta = self._fit_group_lasso_blockwise(X_train, y_train, group_indices)
        
        # Count active groups
        n_active_groups = sum(
            1 for group_idx in group_indices.values()
            if np.any(np.abs(beta[group_idx]) > 1e-8)
        )
        self._n_active_groups_history.append(n_active_groups)
        
        intercept = y_train.mean()
        return beta, intercept
    
    def fit(self, y: pd.Series, X: pd.DataFrame) -> RollingResult:
        """Override to store feature names"""
        self._feature_names = list(X.columns)
        return super().fit(y, X)
    
    def _get_additional_stats(self, coef: np.ndarray, intercept: float) -> Dict:
        if self._n_active_groups_history:
            return {'n_active_groups': self._n_active_groups_history[-1]}
        return {}
    
    def _get_model_name(self) -> str:
        n_groups = len(self.groups) if self.groups else "auto"
        return f"Group Lasso (α={self.alpha}, groups={n_groups})"
    
    def _print_model_specific_summary(self):
        print(f"Alpha                : {self.alpha}")
        n_groups = len(self.groups) if self.groups else "auto"
        print(f"Number of groups     : {n_groups}")
        if self.result_.additional_stats and 'n_active_groups' in self.result_.additional_stats:
            n_active = self.result_.additional_stats['n_active_groups']
            print(f"Mean active groups   : {n_active.mean():.1f}")


def plot_rolling_results(result: RollingResult, figsize=(14, 12)):
    fig, axes = plt.subplots(5, 1, figsize=figsize)
    
    # 1. RMSE
    ax = axes[0]
    result.rmse_cum.plot(ax=ax, label="Cumulative RMSE", linewidth=2, color='tab:blue')
    if result.rmse_roll is not None:
        result.rmse_roll.plot(
            ax=ax, 
            label=f"RMSE rolling ({result.rmse_roll.name})",
            linestyle="--", 
            alpha=0.8,
            color='tab:orange'
        )
    ax.set_title("Out-of-Sample RMSE", fontsize=13, fontweight='bold')
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Predictions vs true values
    ax = axes[1]
    result.y_true.plot(ax=ax, label="Observed", alpha=0.7, linewidth=1.5, color='black')
    result.y_pred.plot(ax=ax, label="Predicted", alpha=0.8, linewidth=1.5, color='tab:red')
    ax.set_title("Predictions vs Observed Values", fontsize=13, fontweight='bold')
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Prediction errors
    ax = axes[2]
    result.err.plot(ax=ax, label="Prediction error", color="tab:red", alpha=0.7)
    ax.axhline(0, color="black", linewidth=1, linestyle='--')
    ax.fill_between(result.err.index, 0, result.err.values, alpha=0.2, color='tab:red')
    ax.set_title("Prediction Errors", fontsize=13, fontweight='bold')
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Number of non-zero coefficients (sparsity)
    ax = axes[3]
    result.n_nonzero_coefs.plot(ax=ax, marker='o', linestyle='-', markersize=3, color='tab:green')
    ax.axhline(result.n_nonzero_coefs.mean(), color='red', linestyle='--', 
               label=f'Mean: {result.n_nonzero_coefs.mean():.1f}')
    ax.set_title("Number of Non-Zero Coefficients (Variable Selection)", fontsize=13, fontweight='bold')
    ax.set_ylabel("N° non-zero coefs")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Missing data indicator
    ax = axes[4]
    result.has_missing_by_window.astype(int).plot(ax=ax, marker='|', linestyle='', 
                                                   markersize=10, color='tab:purple')
    ax.set_title("Windows with Missing Values", fontsize=13, fontweight='bold')
    ax.set_ylabel("Has missing (1/0)")
    ax.set_xlabel("Date")
    ax.set_ylim([-0.1, 1.1])
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


class TargetRandomForest:
    """
    Random Forest on a rolling or expanding window with EM imputation
    Two-step approach: Lasso selection + RF on selected variables
    """
    
    def __init__(
        self,
        window: int = 36,
        window_type: str = "expanding",
        alpha: float = 0.01,
        n_lags: int = 0,
        n_estimators: int = 100,
        criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = "squared_error",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter_em: int = 50,
        max_iter: int = 1000,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = False,
    ) -> None:
        self.window = window
        self.window_type = window_type
        self.alpha = alpha
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter_em = max_iter_em
        self.max_iter = max_iter
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_: Optional[RollingResult] = None
        
    def _has_missing(self, X: pd.DataFrame) -> bool:
        return X.isna().any().any()
    
    def _impute_median(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(X.median())
    
    def _impute_forward_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.ffill()
        return X_filled.fillna(X_filled.median())
    
    def _impute_em(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        empca = EM_PCA(
            kmax=self.n_factors_imputation,
            criterion='IC2',
            demean=2,
            max_iter=self.max_iter_em,
            verbose=False,
            fallback_n_factors=min(3, self.n_factors_imputation)
        )
        result = empca.fit(X)
        return result.imputed_data, empca
    
    def _impute_data(
        self, 
        X_train: pd.DataFrame, 
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        has_missing_train = self._has_missing(X_train)
        has_missing_test = X_test is not None and self._has_missing(X_test)
        
        if self.imputation_method == 'none':
            if has_missing_train or has_missing_test:
                raise ValueError("Missing values detected but imputation_method='none'")
            return X_train, X_test
        
        if not has_missing_train:
            X_train_imputed = X_train.copy()
            empca_model = None
        else:
            if self.imputation_method == 'em':
                X_train_imputed, empca_model = self._impute_em(X_train)
            elif self.imputation_method == 'median':
                X_train_imputed = self._impute_median(X_train)
                empca_model = None
            elif self.imputation_method == 'forward_fill':
                X_train_imputed = self._impute_forward_fill(X_train)
                empca_model = None
        
        if X_test is None or not has_missing_test:
            X_test_imputed = X_test
        else:
            if self.imputation_method == 'em' and empca_model is not None:
                _, X_test_imp = empca_model.transform(X_test, em=True, return_imputed=True)
                X_test_imputed = X_test_imp
            elif self.imputation_method == 'median':
                X_test_imputed = X_test.fillna(X_train.median())
            elif self.imputation_method == 'forward_fill':
                X_test_imputed = X_test.ffill()
                X_test_imputed = X_test_imputed.fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def fit(self, y: pd.Series, X: pd.DataFrame, threshold: float = 1e-6) -> RollingResult:
        """
        Run RF with Lasso pre-selection on a rolling or expanding window basis
        Lags are handled INSIDE the loop like BaseRollingRegressor
        """
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length: {len(y)} != {len(X)}")
        if self.window >= len(y):
            raise ValueError(f"window ({self.window}) must be < data length ({len(y)})")
    
        if self.window_type not in ["rolling", "expanding"]:
            raise ValueError(f"window_type must be 'rolling' or 'expanding', got {self.window_type}")
    
        # NO lags applied here - keep full data
        df = pd.concat([y.rename("y"), X], axis=1).dropna(subset=['y'])
        y_al = df["y"]
        X_al = df.drop(columns=["y"])
        T = len(df)
    
        y_pred_list = []
        y_true_list = []
        dates_list = []
        has_missing_list = []
        n_nonzero_list = []
        feature_importance_list = [] if self.store_coefs else None
    
        # Store column names for later (will be consistent across all iterations)
        final_col_names = None
    
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
        
            if self.window_type == "rolling":
                X_train = X_al.iloc[t - self.window : t]
                y_train = y_al.iloc[t - self.window : t]
            else:
                X_train = X_al.iloc[0 : t]
                y_train = y_al.iloc[0 : t]

            X_test = X_al.iloc[[t]]
            y_test = y_al.iloc[t]
        
            has_missing = self._has_missing(X_train) or self._has_missing(X_test)
            has_missing_list.append(has_missing)
        
            try:
                X_train_imputed, X_test_imputed = self._impute_data(X_train, X_test)
            except Exception as e:
                if self.verbose:
                    print(f"  WARNING: Imputation failed at t={t}: {e}")
                continue
        
            common_idx = y_train.index.intersection(X_train_imputed.index)
            y_train_al = y_train.loc[common_idx]
            X_train_al = X_train_imputed.loc[common_idx]
        
            if len(y_train_al) < 10:
                if self.verbose:
                    print(f"  WARNING: Only {len(y_train_al)} observations at t={t}, skipping")
                continue
        
        
            if self.n_lags > 0:
                # Combine train and test for lag creation
                X_all = pd.concat([X_train_al, X_test_imputed], axis=0)
                X_all_lag = create_lagged_dataframe(X_all, n_lags=self.n_lags, drop_na=False)
            
                # Store column names on first iteration
                if final_col_names is None:
                    final_col_names = X_all_lag.columns
            
                # Split back into train and test
                X_train_lag = X_all_lag.iloc[:-1].copy()
                X_test_lag = X_all_lag.iloc[[-1]].copy()
            
                # Remove rows with NaN from lags in train
                valid_train_idx = X_train_lag.dropna().index
                X_train_al = X_train_lag.loc[valid_train_idx]
                y_train_al = y_train_al.loc[valid_train_idx]
            
                # For test, forward fill NaN if any (use last known values)
                X_test_imputed = X_test_lag.ffill().bfill()
            else:
                # No lags - store column names on first iteration
                if final_col_names is None:
                    final_col_names = X_train_al.columns
        
            # Check again after adding lags
            if len(y_train_al) < 10:
                if self.verbose:
                    print(f"  WARNING: Only {len(y_train_al)} observations after lags at t={t}, skipping")
                continue
        
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
        
            # Step 1: Lasso variable selection
            lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=True,
                max_iter=self.max_iter,
                random_state=42
            )
            lasso.fit(X_train_scaled, y_train_al.values)
        
            selected = np.abs(lasso.coef_) > threshold
            n_selected = selected.sum()
            n_nonzero_list.append(n_selected)
        
            if n_selected == 0:
                if self.verbose:
                    print(f"  WARNING: No variables selected at t={t}")
                # Still append zeros for feature importances
                if self.store_coefs:
                    importances = np.zeros(len(final_col_names))
                    importance_dict = dict(zip(final_col_names, importances))
                    feature_importance_list.append(importance_dict)
                continue
        
            X_train_selected = X_train_scaled[:, selected]
        
            # Step 2: Random Forest on selected variables
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_selected, y_train_al.values)
        
            # Prediction
            X_test_scaled = scaler.transform(X_test_imputed.values)
            X_test_selected = X_test_scaled[:, selected]
            y_pred = rf.predict(X_test_selected)[0]
        
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
        
            # Store feature importances (using consistent column names)
            if self.store_coefs:
                importances = np.zeros(len(final_col_names))
                importances[selected] = rf.feature_importances_
                importance_dict = dict(zip(final_col_names, importances))
                feature_importance_list.append(importance_dict)
    
        dates_idx = pd.DatetimeIndex(dates_list)
        y_true = pd.Series(y_true_list, index=dates_idx, name="y_true")
        y_pred = pd.Series(y_pred_list, index=dates_idx, name="y_pred")
        err = y_true - y_pred
        err.name = "error"
    
        rmse_cum = np.sqrt((err ** 2).expanding().mean())
        rmse_cum.name = "rmse_cum"
    
        rmse_roll = None
        if self.rmse_window is not None and self.rmse_window >= 2:
            rmse_roll = np.sqrt((err ** 2).rolling(self.rmse_window).mean())
            rmse_roll.name = f"rmse_roll_{self.rmse_window}"
    
        has_missing_series = pd.Series(has_missing_list, index=dates_idx, name="has_missing")
        n_nonzero_series = pd.Series(n_nonzero_list, index=dates_idx, name="n_nonzero_coefs")
    
        feature_importance_df = None
        if self.store_coefs:
            feature_importance_df = pd.DataFrame(feature_importance_list, index=dates_idx)
    
        self.result_ = RollingResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=feature_importance_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=n_nonzero_series
        )
    
        if self.verbose:
            self._print_summary()
    
        return self.result_
        
    def _print_summary(self):
        res = self.result_
        print(f"\n{'='*70}")
        print(f"ROLLING RANDOM FOREST - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Window type          : {self.window_type}")
        print(f"N lags               : {self.n_lags}")
        print(f"Alpha (Lasso select) : {self.alpha}")
        print(f"N estimators (RF)    : {self.n_estimators}")
        print(f"Max depth            : {self.max_depth}")
        print(f"Imputation method    : {self.imputation_method}")
        print(f"N° of predictions    : {len(res.dates)}")
        print(f"\nVariable selection:")
        print(f"  Mean #vars selected: {res.n_nonzero_coefs.mean():.1f}")
        print(f"  Min #vars selected : {res.n_nonzero_coefs.min():.0f}")
        print(f"  Max #vars selected : {res.n_nonzero_coefs.max():.0f}")
        print(f"\nMissing data:")
        pct_missing = 100 * res.has_missing_by_window.sum() / len(res.has_missing_by_window)
        print(f"  Windows with missing: {pct_missing:.1f}%")
        print(f"\nPerformance (out-of-sample):")
        print(f"  Final RMSE (cum)   : {res.rmse_cum.iloc[-1]:.4f}")
        print(f"  Mean absolute error: {res.err.abs().mean():.4f}")
        print(f"  Std of errors      : {res.err.std():.4f}")
        print(f"{'='*70}\n")
    
    def predict(self) -> pd.Series:
        if self.result_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.result_.y_pred 
