from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    """
    Abstract class to store the econometrics and machine learning
    models we will estimate as part of this project
    """

    @abstractmethod
    def model_estimate(self, y, x, window):
        """
        Method to estimate a model using a rolling window
        :param y: variable to estimate
        :param x: regressors
        :param window: window used for the computation
        :return:
        """
        pass
    @abstractmethod
    def predict_model(self, y, x):
        """
        method to predict the results from the model out-sample
        :param y:
        :param x:
        :return:
        """
        pass





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
        fallback_n_factors: int = 3, verbose: bool = True
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

        # -------------------- Checks --------------------
        if self.n_factors_ is None or self.loadings_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if self.feature_names_ is None:
            raise ValueError("feature_names_ missing. Re-fit the model.")

        # -------------------- Align columns --------------------
        missing_cols = set(self.feature_names_) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_names_)
        if missing_cols:
            raise ValueError(f"X missing columns seen in fit(): {sorted(missing_cols)[:10]}")
        if extra_cols:
            raise ValueError(f"X has extra columns not seen in fit(): {sorted(extra_cols)[:10]}")

        X_aligned = X.loc[:, self.feature_names_]

        # -------------------- Prepare arrays --------------------
        X0 = X_aligned.values.astype(float, copy=True)
        miss = np.isnan(X0)

        mu = self.mean_[0, :]
        sd = self.std_[0, :]
        Lambda = self.loadings_.values      # (N × r)
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

        # -------------------- Initial imputation --------------------
        X_imp = X0.copy()
        if miss.any():
            X_imp[miss] = np.take(mu, np.where(miss)[1])

        # ==================== CASE 1: simple projection ====================
        if not em:
            X_tr = _to_transformed(X_imp)
            F = (X_tr @ Lambda) / N
            Xhat_tr = F @ Lambda.T
            Xhat = _to_original(Xhat_tr)

        # ==================== CASE 2: EM with fixed loadings ====================
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

                # convergence on missing entries
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

        # -------------------- Build outputs --------------------
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
        
        # Find minimum
        ic_optimal_idx = np.argmin(IC)
        # Dans _bai_ng(), ajoute un print
        print(f"CT pour r=1: {CT[0]:.6f}")
        print(f"CT pour r={self.kmax}: {CT[-1]:.6f}")
        # Return 0 if r=0 selected, otherwise return r
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  BAI-NG IC{jj} DEBUG")
            print(f"{'='*70}")
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
    Implements rolling linear regression.
    """
    def __init__(self, y: pd.Series, X: pd.DataFrame, window: int, type_window:str = "rolling") -> None:
        """
        Parameters:
        -----------
        y : pd.Series - Dependent variable
        X : pd.DataFrame - Independent variables (factors)
        window : int - Rolling window size
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
        self.result_ = None
        self.type_window: str = type_window
    
    def fit(self) -> RollingRegressionResult:
        """
        Fit rolling OLS regression.
        
        Returns:
        --------
        RollingRegressionResult with dates, coefficients, predictions, R²
        """
        dates, alphas, betas, fitted, r2s = [], [], [], [], []

        # case the window chosen is not rolling or expanding
        if self.type_window != "rolling" and self.type_window != "expanding":
            raise ValueError(f"The rolling regression for a {self.type_window} window is not implemented")
        
        for t in range(self.window, self.T):
            # case for a rolling window
            if self.type_window == "rolling":
                y_window = self.y.iloc[t - self.window : t]
                X_window = self.X.iloc[t - self.window : t]
            else:
                y_window = self.y.iloc[0:t]
                X_window = self.X.iloc[0:t]

            # Add constant and fit
            X_window_c = sm.add_constant(X_window)
            model = sm.OLS(y_window, X_window_c).fit()

            # Predict at time t (OOS)
            # CORRECTION: construire manuellement le vecteur avec constante
            X_oos = self.X.iloc[t].values  # shape (K,)
            X_oos_c = np.concatenate([[1.0], X_oos])  # shape (K+1,) avec constante
            y_pred = np.dot(X_oos_c, model.params)

            # Store results
            dates.append(self.y.index[t])
            alphas.append(model.params.iloc[0])
            betas.append(model.params.iloc[1:].values)
            fitted.append(y_pred)
            r2s.append(model.rsquared)
        
        # Convert to pandas objects with proper index
        dates_idx = pd.DatetimeIndex(dates)
        
        self.result_ = RollingRegressionResult(
            dates=dates_idx,
            alpha=pd.Series(alphas, index=dates_idx, name='alpha'),
            betas=pd.DataFrame(betas, index=dates_idx, columns=self.X.columns),
            fitted=pd.Series(fitted, index=dates_idx, name='indicator_raw'),
            r2=pd.Series(r2s, index=dates_idx, name='r2_train')
        )
        
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
        print(f"\nLast window coefficients:")
        print(f"  Alpha          : {self.result_.alpha.iloc[-1]:.4f}")
        for col, val in zip(self.X.columns, self.result_.betas.iloc[-1]):
            print(f"  {col:12s}   : {val:8.4f}")
        print(f"{'='*60}\n")

    # Class for factor model

    # Class for LASSO


@dataclass
class RollingMLResult:
    dates: pd.DatetimeIndex
    y_true: pd.Series
    y_pred: pd.Series
    err: pd.Series
    rmse_cum: pd.Series
    rmse_roll: Optional[pd.Series] = None


# ============================================================================
# ROLLING LASSO WITH EM IMPUTATION
# ============================================================================

@dataclass
class RollingLassoResult:
    """
    Save Lasso results
    """
    dates: pd.DatetimeIndex
    y_true: pd.Series
    y_pred: pd.Series
    err: pd.Series
    rmse_cum: pd.Series
    rmse_roll: Optional[pd.Series] = None
    coefficients: Optional[pd.DataFrame] = None
    has_missing_by_window: pd.Series = None
    n_nonzero_coefs: pd.Series = None


class RollingLasso:
    """
    LASSO on a rolling window. 
    Missing values are handled via EM algorithm considering only the window.
    The code works in 3 steps:
        - impute missing values in the window if needed using EM-PCA
        - fit LASSO on imputed variables on the window to predict y_t
        -
    """
    
    def __init__(
        self,
        window: int = 36,
        alpha: float = 0.01,
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter_lasso: int = 10000,
        max_iter_em: int = 50,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        - window       : int
            Length of the rolling window in months
        - alpha        : float
            LASSO penalty (hyperparameter)
        - imputation_method : str
            Method to use to replace missing value (should be em)
        - n_factors_imputation : int
            Number of factors to use for EM replacement
        - max_iter_lasso : int
            Maximum number of iterations for the LASSO
        - max_iter_em : int
            Maximum number of iterations for the EM algorithm
        - rmse_window : int 
            Window to use for the computation of the rolling RMSE
        - store_coefs : bool
            True if rolling coefficients have to be stored
        - verbose : bool
            True if extensive user log should be printed  
        """
        self.window = window
        self.alpha = alpha
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter_lasso = max_iter_lasso
        self.max_iter_em = max_iter_em
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_: Optional[RollingLassoResult] = None
        
    def _has_missing(self, X: pd.DataFrame) -> bool:
        """
        Check if a given dataframe contains any missing value
        """
        return X.isna().any().any()
    
    def _impute_median(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(X.median())
    
    def _impute_forward_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.ffill()
        return X_filled.fillna(X_filled.median())
    
    def _impute_em(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        """
        Replace missing values of X by lambda'*F using EM algorithm
        """

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
        Verify if the dataframe contains missing values. If it
        does, replace those missing values following the given procedure.
        """

        has_missing_train = self._has_missing(X_train)
        has_missing_test = X_test is not None and self._has_missing(X_test)
        
        if self.imputation_method == 'none':
            if has_missing_train or has_missing_test:
                raise ValueError("Missing values detected but imputation_method='none'")
            return X_train, X_test
        
        if self.imputation_method == "em":
            X_train_imputed, empca_model = self._impute_em(X_train)
        elif self.imputation_method == "median":
            X_train_imputed = self._impute_median(X_train); empca_model=None
        elif self.imputation_method == "forward_fill":
            X_train_imputed = self._impute_forward_fill(X_train); empca_model=None
        else:
            raise ValueError("Imputation method should be ideally EM")

        
        if X_test is None or not has_missing_test:
            X_test_imputed = X_test
        else:
            if self.imputation_method == 'em' and empca_model is not None:
                F_oos, X_test_imp = empca_model.transform(X_test, em=True, return_imputed=True)
                X_test_imputed = X_test_imp
            elif self.imputation_method == 'median':
                X_test_imputed = X_test.fillna(X_train.median())
            elif self.imputation_method == 'forward_fill':
                X_test_imputed = X_test.ffill()
                X_test_imputed = X_test_imputed.fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def fit(self, y: pd.Series, X: pd.DataFrame) -> RollingLassoResult:
        """
        Run the LASSO regression on a rolling window basis.
        """
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length: {len(y)} != {len(X)}")
        if self.window >= len(y):
            raise ValueError(f"window ({self.window}) must be < data length ({len(y)})")
        
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
        
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
            
            # The window = train ; after it = test
            X_train = X_al.iloc[t - self.window : t]
            y_train = y_al.iloc[t - self.window : t]
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
            
            # Align on window
            common_idx = y_train.index.intersection(X_train_imputed.index)
            y_train_al = y_train.loc[common_idx]
            X_train_al = X_train_imputed.loc[common_idx]
            
            if len(y_train_al) < 10:
                if self.verbose:
                    print(f"  WARNING: Only {len(y_train_al)} observations at t={t}, skipping")
                continue
            # Scale using training set
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
            
            lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=True,
                max_iter=self.max_iter_lasso,
                random_state=42
            )
            lasso.fit(X_train_scaled, y_train_al.values)
            
            # Scale test set using train stats 
            X_test_scaled = scaler.transform(X_test_imputed.values)

            # Predict using the model
            y_pred = lasso.predict(X_test_scaled)[0]
            
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
            
            n_nonzero = np.sum(lasso.coef_ != 0)
            n_nonzero_list.append(n_nonzero)
            
            if self.store_coefs:
                coef_dict = {'intercept': lasso.intercept_}
                coef_dict.update({col: coef for col, coef in zip(X_al.columns, lasso.coef_)})
                coefs_list.append(coef_dict)
        
        dates_idx = pd.DatetimeIndex(dates_list)
        y_true = pd.Series(y_true_list, index=dates_idx, name="y_true")
        y_pred = pd.Series(y_pred_list, index=dates_idx, name="y_pred")

        # Compute OOS error and RMSE
        err = y_true - y_pred
        err.name = "error"
        
        # Cumulative RMSE, i.e. the final one = RMSE on ALL the OOS predictions. 
        rmse_cum = np.sqrt((err ** 2).expanding().mean())
        rmse_cum.name = "rmse_cum"
        
        rmse_roll = None
        if self.rmse_window is not None and self.rmse_window >= 2:
            rmse_roll = np.sqrt((err ** 2).rolling(self.rmse_window).mean())
            rmse_roll.name = f"rmse_roll_{self.rmse_window}"
        
        has_missing_series = pd.Series(has_missing_list, index=dates_idx, name="has_missing")
        n_nonzero_series = pd.Series(n_nonzero_list, index=dates_idx, name="n_nonzero_coefs")
        
        coefs_df = None
        if self.store_coefs:
            coefs_df = pd.DataFrame(coefs_list, index=dates_idx)
        
        # Results
        self.result_ = RollingLassoResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=coefs_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=n_nonzero_series
        )
        
        if self.verbose:
            self._print_summary()
        
        return self.result_
    
    def _print_summary(self):
        res = self.result_
        print(f"\n{'='*70}")
        print(f"ROLLING LASSO WITH EM IMPUTATION - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Alpha (Lasso)        : {self.alpha}")
        print(f"Imputation method    : {self.imputation_method}")
        print(f"N° of predictions    : {len(res.dates)}")
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
    
    def predict(self) -> pd.Series:
        if self.result_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.result_.y_pred
    





# ============================================================================
# ROLLING RIDGE WITH EM IMPUTATION
# ============================================================================

class RollingRidge:
    """
    RIDGE on a rolling window with EM imputation
    Identical to RollingLasso but uses Ridge regression (L2 penalty)
    """
    
    def __init__(
        self,
        window: int = 36,
        alpha: float = 1.0,  # Ridge typically uses larger alpha than Lasso
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter_ridge: int = 10000,
        max_iter_em: int = 50,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        - window       : int
            Length of the rolling window in months
        - alpha        : float
            Ridge penalty (L2 regularization strength)
            Note: Ridge typically uses larger alpha than Lasso (e.g. 1.0 instead of 0.01)
        - imputation_method : str
            Method to use to replace missing value (should be em)
        - n_factors_imputation : int
            Number of factors to use for EM replacement
        - max_iter_ridge : int
            Maximum number of iterations for Ridge
        - max_iter_em : int
            Maximum number of iterations for the EM algorithm
        - rmse_window : int 
            Window to use for the computation of the rolling RMSE
        - store_coefs : bool
            True if rolling coefficients have to be stored
        - verbose : bool
            True if extensive user log should be printed  
        """
        self.window = window
        self.alpha = alpha
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter_ridge = max_iter_ridge
        self.max_iter_em = max_iter_em
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_: Optional[RollingLassoResult] = None
        
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
                F_oos, X_test_imp = empca_model.transform(X_test, em=True, return_imputed=True)
                X_test_imputed = X_test_imp
            elif self.imputation_method == 'median':
                X_test_imputed = X_test.fillna(X_train.median())
            elif self.imputation_method == 'forward_fill':
                X_test_imputed = X_test.ffill()
                X_test_imputed = X_test_imputed.fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def fit(self, y: pd.Series, X: pd.DataFrame) -> RollingLassoResult:
        """
        Run Ridge regression on a rolling window basis.
        """
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length: {len(y)} != {len(X)}")
        if self.window >= len(y):
            raise ValueError(f"window ({self.window}) must be < data length ({len(y)})")
        
        df = pd.concat([y.rename("y"), X], axis=1)
        y_al = df["y"]
        X_al = df.drop(columns=["y"])
        T = len(df)
        
        y_pred_list = []
        y_true_list = []
        dates_list = []
        has_missing_list = []
        n_nonzero_list = []
        coefs_list = [] if self.store_coefs else None
        
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
            
            X_train = X_al.iloc[t - self.window : t]
            y_train = y_al.iloc[t - self.window : t]
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
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
            
            # ============ DIFFÉRENCE : Ridge au lieu de Lasso ============
            ridge = Ridge(
                alpha=self.alpha,
                fit_intercept=True,
                max_iter=self.max_iter_ridge,
                random_state=42
            )
            ridge.fit(X_train_scaled, y_train_al.values)
            # =============================================================
            
            X_test_scaled = scaler.transform(X_test_imputed.values)
            y_pred = ridge.predict(X_test_scaled)[0]
            
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
            
            # Ridge ne fait PAS de sélection de variables (tous les coefs ≠ 0)
            n_nonzero = np.sum(ridge.coef_ != 0)
            n_nonzero_list.append(n_nonzero)
            
            if self.store_coefs:
                coef_dict = {'intercept': ridge.intercept_}
                coef_dict.update({col: coef for col, coef in zip(X_al.columns, ridge.coef_)})
                coefs_list.append(coef_dict)
        
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
        
        coefs_df = None
        if self.store_coefs:
            coefs_df = pd.DataFrame(coefs_list, index=dates_idx)
        
        self.result_ = RollingLassoResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=coefs_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=n_nonzero_series
        )
        
        if self.verbose:
            self._print_summary()
        
        return self.result_
    
    def _print_summary(self):
        res = self.result_
        print(f"\n{'='*70}")
        print(f"ROLLING RIDGE WITH EM IMPUTATION - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Alpha (Ridge)        : {self.alpha}")
        print(f"Imputation method    : {self.imputation_method}")
        print(f"N° of predictions    : {len(res.dates)}")
        print(f"\nCoefficient statistics:")
        print(f"  Mean n° non-zero   : {res.n_nonzero_coefs.mean():.1f}")
        print(f"  (Ridge keeps all variables, but shrinks coefficients)")
        print(f"\nMissing data:")
        pct_missing = 100 * res.has_missing_by_window.sum() / len(res.has_missing_by_window)
        print(f"  Windows with missing: {pct_missing:.1f}% ({res.has_missing_by_window.sum()} / {len(res.dates)})")
        print(f"\nPerformance (out-of-sample):")
        print(f"  Final RMSE (cum)   : {res.rmse_cum.iloc[-1]:.4f}")
        print(f"  Mean absolute error: {res.err.abs().mean():.4f}")
        print(f"  Std of errors      : {res.err.std():.4f}")
        print(f"{'='*70}\n")
    
    def predict(self) -> pd.Series:
        if self.result_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.result_.y_pred


# ============================================================================
# ROLLING ELASTIC NET WITH EM IMPUTATION
# ============================================================================

class RollingElasticNet:
    """
    ELASTIC NET on a rolling window with EM imputation
    Combines L1 (Lasso) and L2 (Ridge) penalties
    """
    
    def __init__(
        self,
        window: int = 36,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,  # 0 = Ridge, 1 = Lasso, 0.5 = 50/50 mix
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter_en: int = 10000,
        max_iter_em: int = 50,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        - window       : int
            Length of the rolling window in months
        - alpha        : float
            Overall regularization strength
        - l1_ratio     : float in [0, 1]
            Mix between L1 and L2 penalty:
            - l1_ratio = 0  → Pure Ridge (L2)
            - l1_ratio = 1  → Pure Lasso (L1)
            - l1_ratio = 0.5 → 50% Lasso + 50% Ridge (RECOMMENDED)
        - imputation_method : str
            Method to use to replace missing value (should be em)
        - n_factors_imputation : int
            Number of factors to use for EM replacement
        - max_iter_en : int
            Maximum number of iterations for Elastic Net
        - max_iter_em : int
            Maximum number of iterations for the EM algorithm
        - rmse_window : int 
            Window to use for the computation of the rolling RMSE
        - store_coefs : bool
            True if rolling coefficients have to be stored
        - verbose : bool
            True if extensive user log should be printed  
        """
        self.window = window
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter_en = max_iter_en
        self.max_iter_em = max_iter_em
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_: Optional[RollingLassoResult] = None
        
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
                F_oos, X_test_imp = empca_model.transform(X_test, em=True, return_imputed=True)
                X_test_imputed = X_test_imp
            elif self.imputation_method == 'median':
                X_test_imputed = X_test.fillna(X_train.median())
            elif self.imputation_method == 'forward_fill':
                X_test_imputed = X_test.ffill()
                X_test_imputed = X_test_imputed.fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def fit(self, y: pd.Series, X: pd.DataFrame) -> RollingLassoResult:
        """
        Run Elastic Net regression on a rolling window basis.
        """
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length: {len(y)} != {len(X)}")
        if self.window >= len(y):
            raise ValueError(f"window ({self.window}) must be < data length ({len(y)})")
        
        df = pd.concat([y.rename("y"), X], axis=1)
        y_al = df["y"]
        X_al = df.drop(columns=["y"])
        T = len(df)
        
        y_pred_list = []
        y_true_list = []
        dates_list = []
        has_missing_list = []
        n_nonzero_list = []
        coefs_list = [] if self.store_coefs else None
        
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
            
            X_train = X_al.iloc[t - self.window : t]
            y_train = y_al.iloc[t - self.window : t]
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
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
            
            # ============ DIFFÉRENCE : Elastic Net ============
            elasticnet = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
                max_iter=self.max_iter_en,
                random_state=42
            )
            elasticnet.fit(X_train_scaled, y_train_al.values)
            # ==================================================
            
            X_test_scaled = scaler.transform(X_test_imputed.values)
            y_pred = elasticnet.predict(X_test_scaled)[0]
            
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
            
            n_nonzero = np.sum(elasticnet.coef_ != 0)
            n_nonzero_list.append(n_nonzero)
            
            if self.store_coefs:
                coef_dict = {'intercept': elasticnet.intercept_}
                coef_dict.update({col: coef for col, coef in zip(X_al.columns, elasticnet.coef_)})
                coefs_list.append(coef_dict)
        
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
        
        coefs_df = None
        if self.store_coefs:
            coefs_df = pd.DataFrame(coefs_list, index=dates_idx)
        
        self.result_ = RollingLassoResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=coefs_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=n_nonzero_series
        )
        
        if self.verbose:
            self._print_summary()
        
        return self.result_
    
    def _print_summary(self):
        res = self.result_
        print(f"\n{'='*70}")
        print(f"ROLLING ELASTIC NET WITH EM IMPUTATION - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Alpha (overall)      : {self.alpha}")
        print(f"L1 ratio             : {self.l1_ratio} ({self.l1_ratio*100:.0f}% Lasso, {(1-self.l1_ratio)*100:.0f}% Ridge)")
        print(f"Imputation method    : {self.imputation_method}")
        print(f"N° of predictions    : {len(res.dates)}")
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
    
    def predict(self) -> pd.Series:
        if self.result_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.result_.y_pred


class RollingAdaptiveLasso:
    """
    Adaptive Lasso on rolling window with EM imputation
    
    Two-step procedure:
    1. Initial estimation (Ridge or Lasso) to get β̂_initial
    2. Adaptive Lasso with weights wⱼ = 1/|β̂ⱼ|^γ
    
    This gives better variable selection and less bias than standard Lasso.
    """
    
    def __init__(
        self,
        window: int = 36,
        alpha: float = 0.01,
        gamma: float = 1.0,  # Adaptive weight power (typically 0.5, 1, or 2)
        initial_estimator: Literal['ridge', 'lasso', 'ols'] = 'ridge',
        alpha_initial: float = 1.0,  # Alpha for initial Ridge/Lasso
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter_lasso: int = 10000,
        max_iter_em: int = 50,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        gamma : float
            Power for adaptive weights. Common values:
            - γ=0.5 : Less aggressive weighting
            - γ=1.0 : Standard (RECOMMENDED)
            - γ=2.0 : More aggressive weighting
        initial_estimator : str
            Method for initial coefficient estimation:
            - 'ridge' : Ridge regression (RECOMMENDED for p > n)
            - 'lasso' : Standard Lasso
            - 'ols'   : OLS (only if n >> p)
        alpha_initial : float
            Regularization for initial estimator (if ridge/lasso)
        """
        self.window = window
        self.alpha = alpha
        self.gamma = gamma
        self.initial_estimator = initial_estimator
        self.alpha_initial = alpha_initial
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter_lasso = max_iter_lasso
        self.max_iter_em = max_iter_em
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_ = None
        
    def _has_missing(self, X: pd.DataFrame) -> bool:
        return X.isna().any().any()
    
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
                X_train_imputed = X_train.fillna(X_train.median())
                empca_model = None
            elif self.imputation_method == 'forward_fill':
                X_train_imputed = X_train.ffill().fillna(X_train.median())
                empca_model = None
        
        if X_test is None or not has_missing_test:
            X_test_imputed = X_test
        else:
            if self.imputation_method == 'em' and empca_model is not None:
                F_oos, X_test_imp = empca_model.transform(X_test, em=True, return_imputed=True)
                X_test_imputed = X_test_imp
            elif self.imputation_method == 'median':
                X_test_imputed = X_test.fillna(X_train.median())
            elif self.imputation_method == 'forward_fill':
                X_test_imputed = X_test.ffill().fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def _get_initial_weights(self, X_scaled: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute adaptive weights from initial estimation
        
        Returns
        -------
        weights : np.ndarray (p,)
            Adaptive penalty weights wⱼ = 1/|β̂ⱼ|^γ
        """
        if self.initial_estimator == 'ridge':
            model = Ridge(alpha=self.alpha_initial, fit_intercept=False)
            model.fit(X_scaled, y)
            beta_init = model.coef_
            
        elif self.initial_estimator == 'lasso':
            model = Lasso(alpha=self.alpha_initial, fit_intercept=False, max_iter=self.max_iter_lasso)
            model.fit(X_scaled, y)
            beta_init = model.coef_
            
        elif self.initial_estimator == 'ols':
            # OLS via normal equations (may be unstable if p > n)
            try:
                beta_init = np.linalg.lstsq(X_scaled, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback to Ridge if OLS fails
                if self.verbose:
                    print("  OLS failed, using Ridge instead")
                model = Ridge(alpha=0.1, fit_intercept=False)
                model.fit(X_scaled, y)
                beta_init = model.coef_
        
        # Compute adaptive weights: wⱼ = 1/|βⱼ|^γ
        # Add small constant to avoid division by zero
        epsilon = 1e-6
        weights = 1.0 / (np.abs(beta_init) + epsilon) ** self.gamma
        
        return weights
    
    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        Rolling Adaptive Lasso estimation
        
        At each date t:
        1. Impute missing values
        2. Get initial coefficients (Ridge/Lasso/OLS)
        3. Compute adaptive weights
        4. Fit weighted Lasso (manually via coordinate descent)
        """
        from sklearn.preprocessing import StandardScaler
        
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length")
        if self.window >= len(y):
            raise ValueError(f"window must be < data length")
        
        df = pd.concat([y.rename("y"), X], axis=1)
        y_al = df["y"]
        X_al = df.drop(columns=["y"])
        T = len(df)
        
        y_pred_list = []
        y_true_list = []
        dates_list = []
        has_missing_list = []
        n_nonzero_list = []
        coefs_list = [] if self.store_coefs else None
        weights_list = [] if self.store_coefs else None
        
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
            
            X_train = X_al.iloc[t - self.window : t]
            y_train = y_al.iloc[t - self.window : t]
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
                    print(f"  WARNING: Only {len(y_train_al)} observations at t={t}")
                continue
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
            
            # STEP 1: Get adaptive weights
            weights = self._get_initial_weights(X_train_scaled, y_train_al.values)
            
            # STEP 2: Fit Adaptive Lasso
            # We use Lasso with scaled penalty: λ_adaptive,j = λ * wⱼ
            # Sklearn doesn't support per-feature penalties directly,
            # so we rescale features: X̃ⱼ = Xⱼ/wⱼ
            X_train_weighted = X_train_scaled / weights
            
            adaptive_lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=True,
                max_iter=self.max_iter_lasso,
                random_state=42
            )
            adaptive_lasso.fit(X_train_weighted, y_train_al.values)
            
            # Recover true coefficients: β̂_true = β̂_weighted / weights
            coef_adaptive = adaptive_lasso.coef_ / weights
            
            # Predict
            X_test_scaled = scaler.transform(X_test_imputed.values)
            y_pred = adaptive_lasso.intercept_ + np.dot(X_test_scaled, coef_adaptive)[0]
            
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
            
            n_nonzero = np.sum(np.abs(coef_adaptive) > 1e-8)
            n_nonzero_list.append(n_nonzero)
            
            if self.store_coefs:
                coef_dict = {'intercept': adaptive_lasso.intercept_}
                coef_dict.update({col: c for col, c in zip(X_al.columns, coef_adaptive)})
                coefs_list.append(coef_dict)
                
                weights_dict = {col: w for col, w in zip(X_al.columns, weights)}
                weights_list.append(weights_dict)
        
        # Build results (same structure as RollingLasso)
        
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
        
        coefs_df = None
        if self.store_coefs:
            coefs_df = pd.DataFrame(coefs_list, index=dates_idx)
        
        self.result_ = RollingLassoResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=coefs_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=n_nonzero_series
        )
        
        # Store weights for analysis
        if self.store_coefs:
            self.weights_ = pd.DataFrame(weights_list, index=dates_idx)
        
        if self.verbose:
            self._print_summary()
        
        return self.result_
    
    def _print_summary(self):
        res = self.result_
        print(f"\n{'='*70}")
        print(f"ROLLING ADAPTIVE LASSO - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Alpha                : {self.alpha}")
        print(f"Gamma (weight power) : {self.gamma}")
        print(f"Initial estimator    : {self.initial_estimator}")
        print(f"N° of predictions    : {len(res.dates)}")
        print(f"\nVariable selection:")
        print(f"  Mean n° non-zero   : {res.n_nonzero_coefs.mean():.1f}")
        print(f"  Std n° non-zero    : {res.n_nonzero_coefs.std():.1f}")
        print(f"  Min - Max          : {res.n_nonzero_coefs.min()} - {res.n_nonzero_coefs.max()}")
        print(f"\nPerformance (out-of-sample):")
        print(f"  Final RMSE (cum)   : {res.rmse_cum.iloc[-1]:.4f}")
        print(f"  Mean absolute error: {res.err.abs().mean():.4f}")
        print(f"  Std of errors      : {res.err.std():.4f}")
        print(f"{'='*70}\n")
    
    def predict(self) -> pd.Series:
        if self.result_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.result_.y_pred

"""
Group Lasso for Rolling Window
================================

Group Lasso pénalise des groupes de variables ensemble.
Utile quand certaines variables doivent être sélectionnées/retirées ensemble.

Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression 
with grouped variables.

Exemples d'utilisation:
- Variables par pays (toutes les vars d'un pays ensemble)
- Variables catégorielles (dummies d'une même catégorie)
- Lags d'une variable (lag1, lag2, lag3 de la même série)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass


class RollingGroupLasso:
    """
    Group Lasso on rolling window with EM imputation
    
    Minimizes: ||y - Xβ||² + λ Σ_g √(|G_g|) ||β_g||₂
    
    where G_g is group g and ||β_g||₂ is the L2 norm of coefficients in group g.
    
    This encourages sparsity at the group level (not individual variables).
    """
    
    def __init__(
        self,
        window: int = 36,
        alpha: float = 0.01,
        groups: Optional[Dict[str, List[str]]] = None,  # {'group_name': ['var1', 'var2']}
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter: int = 1000,
        max_iter_em: int = 50,
        tol: float = 1e-4,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        groups : dict
            Dictionary mapping group names to list of variable names.
            Example: {
                'price_indices': ['IPCAG_EA', 'IPCOG_EA', 'IPICAG_EA'],
                'stock_markets': ['DAX', 'CAC40', 'FTSE_MIB'],
                'monetary': ['M1_EACC', 'M2_EACC']
            }
            If None, each variable is its own group (= standard Lasso)
        alpha : float
            Overall regularization strength
        """
        self.window = window
        self.alpha = alpha
        self.groups = groups
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter = max_iter
        self.max_iter_em = max_iter_em
        self.tol = tol
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_ = None
        
    def _has_missing(self, X: pd.DataFrame) -> bool:
        return X.isna().any().any()
    
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
                raise ValueError("Missing values detected")
            return X_train, X_test
        
        if not has_missing_train:
            X_train_imputed = X_train.copy()
            empca_model = None
        else:
            if self.imputation_method == 'em':
                X_train_imputed, empca_model = self._impute_em(X_train)
            elif self.imputation_method == 'median':
                X_train_imputed = X_train.fillna(X_train.median())
                empca_model = None
            elif self.imputation_method == 'forward_fill':
                X_train_imputed = X_train.ffill().fillna(X_train.median())
                empca_model = None
        
        if X_test is None or not has_missing_test:
            X_test_imputed = X_test
        else:
            if self.imputation_method == 'em' and empca_model is not None:
                F_oos, X_test_imp = empca_model.transform(X_test, em=True, return_imputed=True)
                X_test_imputed = X_test_imp
            elif self.imputation_method == 'median':
                X_test_imputed = X_test.fillna(X_train.median())
            elif self.imputation_method == 'forward_fill':
                X_test_imputed = X_test.ffill().fillna(X_train.median())
        
        return X_train_imputed, X_test_imputed
    
    def _setup_groups(self, feature_names: List[str]) -> Dict[str, List[int]]:
        """
        Convert group definitions to indices
        
        Returns
        -------
        group_indices : dict
            {'group_name': [idx1, idx2, ...]}
        """
        if self.groups is None:
            # Each variable is its own group (standard Lasso)
            return {var: [i] for i, var in enumerate(feature_names)}
        
        # Map variable names to indices
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        group_indices = {}
        
        for group_name, var_names in self.groups.items():
            indices = []
            for var in var_names:
                if var in name_to_idx:
                    indices.append(name_to_idx[var])
                elif self.verbose:
                    print(f"  WARNING: Variable '{var}' in group '{group_name}' not found")
            if indices:
                group_indices[group_name] = indices
        
        # Add ungrouped variables as singleton groups
        grouped_indices = set()
        for indices in group_indices.values():
            grouped_indices.update(indices)
        
        for i, name in enumerate(feature_names):
            if i not in grouped_indices:
                group_indices[f"_singleton_{name}"] = [i]
        
        return group_indices
    
    def _fit_group_lasso(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        group_indices: Dict[str, List[int]]
    ) -> np.ndarray:
        """
        Fit Group Lasso via block coordinate descent
        
        Minimizes: ||y - Xβ||² + λ Σ_g √|G_g| ||β_g||₂
        
        Returns
        -------
        beta : np.ndarray (p,)
        """
        n, p = X.shape
        beta = np.zeros(p)
        
        # Precompute X'X and X'y
        XtX = X.T @ X
        Xty = X.T @ y
        
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            
            # Update each group
            for group_name, group_idx in group_indices.items():
                group_idx = np.array(group_idx)
                
                # Partial residual
                r = y - X @ beta + X[:, group_idx] @ beta[group_idx]
                
                # Group update
                z = X[:, group_idx].T @ r  # Gradient term
                
                # Soft-thresholding for group
                z_norm = np.linalg.norm(z)
                group_size = len(group_idx)
                threshold = self.alpha * np.sqrt(group_size)
                
                if z_norm <= threshold:
                    # Shrink entire group to zero
                    beta[group_idx] = 0
                else:
                    # Block soft-thresholding
                    # Solve: ||y - X_g β_g||² + λ√|G| ||β_g||₂
                    # Solution: β_g = (1 - λ√|G|/||z||) * (X_g'X_g)^{-1} z
                    
                    XgXg = XtX[np.ix_(group_idx, group_idx)]
                    try:
                        beta_g = np.linalg.solve(XgXg + 1e-6 * np.eye(group_size), z)
                        beta_g = beta_g * (1 - threshold / z_norm)
                        beta[group_idx] = beta_g
                    except np.linalg.LinAlgError:
                        beta[group_idx] = 0
            
            # Check convergence
            if np.linalg.norm(beta - beta_old) < self.tol:
                break
        
        return beta
    
    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        Rolling Group Lasso estimation
        """
        if len(y) != len(X):
            raise ValueError("y and X must have same length")
        if self.window >= len(y):
            raise ValueError("window must be < data length")
        
        df = pd.concat([y.rename("y"), X], axis=1)
        y_al = df["y"]
        X_al = df.drop(columns=["y"])
        T = len(df)
        
        y_pred_list = []
        y_true_list = []
        dates_list = []
        has_missing_list = []
        n_nonzero_list = []
        n_active_groups_list = []
        coefs_list = [] if self.store_coefs else None
        
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
            
            X_train = X_al.iloc[t - self.window : t]
            y_train = y_al.iloc[t - self.window : t]
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
                continue
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
            
            # Setup groups
            group_indices = self._setup_groups(list(X_train_al.columns))
            
            # Fit Group Lasso
            beta = self._fit_group_lasso(
                X_train_scaled, 
                y_train_al.values,
                group_indices
            )
            
            # Intercept (mean of y)
            intercept = y_train_al.mean()
            
            # Predict
            X_test_scaled = scaler.transform(X_test_imputed.values)
            y_pred = intercept + np.dot(X_test_scaled, beta)[0]
            
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
            
            n_nonzero = np.sum(np.abs(beta) > 1e-8)
            n_nonzero_list.append(n_nonzero)
            
            # Count active groups
            n_active_groups = sum(
                1 for group_idx in group_indices.values()
                if np.any(np.abs(beta[group_idx]) > 1e-8)
            )
            n_active_groups_list.append(n_active_groups)
            
            if self.store_coefs:
                coef_dict = {'intercept': intercept}
                coef_dict.update({col: c for col, c in zip(X_al.columns, beta)})
                coefs_list.append(coef_dict)
        
        # Build results
        
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
        
        coefs_df = None
        if self.store_coefs:
            coefs_df = pd.DataFrame(coefs_list, index=dates_idx)
        
        self.result_ = RollingLassoResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=coefs_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=n_nonzero_series
        )
        
        # Store group statistics
        self.n_active_groups_ = pd.Series(n_active_groups_list, index=dates_idx, name="n_active_groups")
        
        if self.verbose:
            self._print_summary()
        
        return self.result_
    
    def _print_summary(self):
        res = self.result_
        n_groups = len(self.groups) if self.groups else res.n_nonzero_coefs.mean()
        print(f"\n{'='*70}")
        print(f"ROLLING GROUP LASSO - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Alpha                : {self.alpha}")
        print(f"Number of groups     : {n_groups}")
        print(f"N° of predictions    : {len(res.dates)}")
        print(f"\nVariable selection:")
        print(f"  Mean n° non-zero vars   : {res.n_nonzero_coefs.mean():.1f}")
        print(f"  Mean n° active groups   : {self.n_active_groups_.mean():.1f}")
        print(f"\nPerformance (out-of-sample):")
        print(f"  Final RMSE (cum)   : {res.rmse_cum.iloc[-1]:.4f}")
        print(f"  Mean absolute error: {res.err.abs().mean():.4f}")
        print(f"{'='*70}\n")



def plot_rolling_results(result: RollingLassoResult, figsize=(14, 12)):
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
    """
    
    def __init__(
        self,
        window: int = 36,
        type_window: str = "rolling",
        alpha: float = 1,
        n_estimators: int = 100, # Number of trees for the forest
        criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = "squared_error", # Function to measure the quality of a split
        max_depth: int = None,
        min_sample_split: int = 2,
        min_sample_leaf: int = 1,
        imputation_method: Literal['em', 'median', 'forward_fill', 'none'] = 'em',
        n_factors_imputation: int = 10,
        max_iter_em: int = 50,
        max_iter: int = 1000,
        rmse_window: Optional[int] = 12,
        store_coefs: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        - window       : int
            Length of the rolling window in months
        - imputation_method : str
            Method to use to replace missing value (should be em)
        - n_factors_imputation : int
            Number of factors to use for EM replacement
        - max_iter_em : int
            Maximum number of iterations for the EM algorithm
        - rmse_window : int 
            Window to use for the computation of the rolling RMSE
        - store_coefs : bool
            True if rolling coefficients have to be stored
        - verbose : bool
            True if extensive user log should be printed  
        """
        self.window = window
        self.type_window = type_window
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.imputation_method = imputation_method
        self.n_factors_imputation = n_factors_imputation
        self.max_iter_em = max_iter_em
        self.max_iter = max_iter
        self.rmse_window = rmse_window
        self.store_coefs = store_coefs
        self.verbose = verbose
        self.result_: Optional[RollingLassoResult] = None
        
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
    
    def fit(self, y: pd.Series, X: pd.DataFrame, threshold: float = 1e-6) -> RollingLassoResult:
        """
        Run RF on a rolling or expanding window basis
        """
        if len(y) != len(X):
            raise ValueError(f"y and X must have same length: {len(y)} != {len(X)}")
        if self.window >= len(y):
            raise ValueError(f"window ({self.window}) must be < data length ({len(y)})")
        
        if self.type_window != "rolling" and self.type_window != "expanding":
            raise ValueError(f"The choice of the window {self.type_window} is not implemented for this function")
        
        df = pd.concat([y.rename("y"), X], axis=1)
        y_al = df["y"]
        X_al = df.drop(columns=["y"])
        T = len(df)
        
        y_pred_list = []
        y_true_list = []
        dates_list = []
        has_missing_list = []
        n_nonzero_list = []
        coefs_list = [] if self.store_coefs else None
        
        for t in range(self.window, T):
            if self.verbose and t % 12 == 0:
                print(f"Processing t={t}/{T} ({y_al.index[t].strftime('%Y-%m')})")
            
            if self.type_window == "rolling":
                X_train = X_al.iloc[t - self.window : t]
                y_train = y_al.iloc[t - self.window : t]
            else:
                X_train = X_al.iloc[0 : t]
                y_train = y_al.iloc[0 : t]

            X_test = X_al.iloc[[t]]
            y_test = y_al.iloc[t]
            
            # If there are missing values, we impute them 
            has_missing = self._has_missing(X_train) or self._has_missing(X_test)
            has_missing_list.append(has_missing)
            try:
                X_train_imputed, X_test_imputed = self._impute_data(X_train, X_test)
            except Exception as e:
                if self.verbose:
                    print(f"  WARNING: Imputation failed at t={t}: {e}")
                continue
            
            # We only keep predictors and dependent variables available on the same periods
            common_idx = y_train.index.intersection(X_train_imputed.index)
            y_train_al = y_train.loc[common_idx]
            X_train_al = X_train_imputed.loc[common_idx]
            
            if len(y_train_al) < 10:
                if self.verbose:
                    print(f"  WARNING: Only {len(y_train_al)} observations at t={t}, skipping")
                continue
            
            # Scaling of data before estimation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_al.values)
            
            ### First step - select the subset of regressors using a penalized reggression (Lasso for now, can be extended)
            lasso = Lasso( # A checker: cross-val directement dans la boucle
                alpha = self.alpha,
                fit_intercept=True,
                max_iter=self.max_iter,
                random_state=42
            )

            lasso.fit(X_train_scaled, y_train_al.values)

            # Once the Lasso is fit, we only keep predictors for which the coefficients is large enough
            selected = np.abs(lasso.coef_) > threshold
            X_train_scaled = X_train_scaled[:, selected]
            if X_train_scaled.shape[1] == 0:
                if self.verbose:
                    print(f"No variables selected for time t = {t}")
                continue

            ### Second step - Estimate the Random forest on this subset of predictors
            rf = RandomForestRegressor(
                n_estimators = self.n_estimators, 
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_sample_split,
                min_samples_leaf=self.min_sample_leaf,
                random_state=42
            )

            rf.fit(X_train_scaled, y_train_al.values)

            # We compute predicted values for the set of predictors selected by Lasso / EN, ...
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_imputed.values),
                columns = X_test_imputed.columns,
                index = X_test_imputed.index
            )
            X_test_scaled = X_test_scaled.loc[:,selected].values

            # We make the prediction and retrieve the results
            y_pred = rf.predict(X_test_scaled)[0]
            dates_list.append(y_al.index[t])
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)

            ### Third step - use shape to build the index


            
            # Ridge ne fait PAS de sélection de variables (tous les coefs ≠ 0)
            if self.store_coefs:
                coef_dict = {'intercept': 0}
                #coef_dict.update({col: coef for col, coef in zip(X_al.columns, ridge.coef_)})
                coefs_list.append(coef_dict)
        
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
        #n_nonzero_series = pd.Series(n_nonzero_list, index=dates_idx, name="n_nonzero_coefs")
        
        coefs_df = None
        if self.store_coefs:
            coefs_df = pd.DataFrame(coefs_list, index=dates_idx)
        
        self.result_ = RollingLassoResult(
            dates=dates_idx,
            y_true=y_true,
            y_pred=y_pred,
            err=err,
            rmse_cum=rmse_cum,
            rmse_roll=rmse_roll,
            coefficients=coefs_df,
            has_missing_by_window=has_missing_series,
            n_nonzero_coefs=None
        )
        
        if self.verbose:
            self._print_summary()
        
        return self.result_
    
    def _print_summary(self):
        res = self.result_
        print(f"\n{'='*70}")
        print(f"ROLLING TRF WITH EM IMPUTATION - SUMMARY")
        print(f"{'='*70}")
        print(f"Window size          : {self.window}")
        print(f"Alpha (Ridge)        : {self.alpha}")
        print(f"Imputation method    : {self.imputation_method}")
        print(f"N° of predictions    : {len(res.dates)}")
        print(f"\nCoefficient statistics:")
        print(f"\nMissing data:")
        pct_missing = 100 * res.has_missing_by_window.sum() / len(res.has_missing_by_window)
        print(f"  Windows with missing: {pct_missing:.1f}% ({res.has_missing_by_window.sum()} / {len(res.dates)})")
        print(f"\nPerformance (out-of-sample):")
        print(f"  Final RMSE (cum)   : {res.rmse_cum.iloc[-1]:.4f}")
        print(f"  Mean absolute error: {res.err.abs().mean():.4f}")
        print(f"  Std of errors      : {res.err.std():.4f}")
        print(f"{'='*70}\n")
    
    def predict(self) -> pd.Series:
        if self.result_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.result_.y_pred
