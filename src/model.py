from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import numpy as np
import pandas as pd
import statsmodels.api as sm


class Model(ABC):
    """
    Abstract class to store the econometrics and machine learning
    models we will estimate as part of this project
    """
    @abstractmethod
    def __init__(self, y, x):
        pass

    @abstractmethod
    def model_estimate(self, window):
        """
        Method to estimate a model using a rolling window
        :param y: variable to estimate
        :param x: regressors
        :param window: window used for the computation
        :return:
        """
        pass
    @abstractmethod
    def predict_model(self):
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
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Project new data onto fitted factors (out-of-sample)
        We keep loadings fixed for OOS analysis (to predict without look-ahead bias)
        NEEDS TO BE IMPLEMENTED
        
        Parameters:
        -----------
        X : pd.DataFrame (T_new × N)
            New data with same N variables but new observations
            
        Returns:
        --------
        F_new : pd.DataFrame (T_new × r)
            Projected factors
        """
        if self.n_factors_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Impute missing with training means
        X_array = X.values
        x_missing = np.isnan(X_array)
        X_imputed = X_array.copy()
        
        # Use training means for imputation
        col_mean_train = self.mean_[0, :] 
        X_imputed[x_missing] = np.take(col_mean_train, np.where(x_missing)[1])
        
        # Transform with training parameters
        X_transformed = (X_imputed - self.mean_[0, :]) / self.std_[0, :]
        
        # Project onto loadings: F_new = X_new @ lambda / N
        # (Need to recompute loadings from last fit - store in fit())
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "Out-of-sample projection requires storing loadings. "
            "Use fit() result's loadings and compute F = X @ Λ / N manually."
        )
    
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
    def __init__(self, y: pd.Series, X: pd.DataFrame, window: int) -> None:
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
    
    def fit(self) -> RollingRegressionResult:
        """
        Fit rolling OLS regression.
        
        Returns:
        --------
        RollingRegressionResult with dates, coefficients, predictions, R²
        """
        dates, alphas, betas, fitted, r2s = [], [], [], [], []
        
        for t in range(self.window, self.T):
            # Training window
            y_window = self.y.iloc[t - self.window : t]
            X_window = self.X.iloc[t - self.window : t]

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
            alphas.append(model.params[0])
            betas.append(model.params[1:].values)
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

# Class for ridge model
class RidgeMod:
    def __init__(self, y: pd.Series, x:pd.DataFrame, state:int):
        self.y: pd.Series = y
        self.x: pd.DataFrame = x
        self.state: int = state
        self.results:tuple = None
        
    def model_estimate(self, window: int, alphas: tuple, n_splits: int):
        """
        Estimation of a ridge regression approach following a rolling window
        framework
        :param self: Description
        :param window: Description
         """

        # First: impute missing values using EM algorithm (to adapt)

        # We compute the number of periods of our sample
        t: int = self.y.shape[0]

        # Results to retrieve
        dates, const, betas, fitted = [], [], [], []

        # Model configuration (can be generalized to elastic net and lasso)
        model_pipeline = Pipeline([
            ('scaler',StandardScaler()),
            ('model', RidgeCV(alphas = alphas,
                                  fit_intercept=True,
                                  cv = TimeSeriesSplit(n_splits = n_splits)))
        ])

        # For loop to estimate the model over
        for i in range(window, t):

            
            print(type(self.x), self.x.shape)
            print(type(self.y), self.y.shape)

            # Select all the available information for the forecast
            y_train = self.y.iloc[i-window:i]
            x_train = self.x.iloc[i-window:i, :]

            # Estimation of the model with cross validation respecting the linear dependence (clone such that we estimate a new model on each iteration)
            ridge_model = clone(model_pipeline)
            ridge_model.fit(x_train, y_train)

            # Prediction at time t for y
            x_oos = self.x.iloc[i].values  
            y_pred = ridge_model.named_steps["model"].intercept_ + np.dot(x_oos, ridge_model.named_steps["model"].coef_)

            # We retrieve the coefficients associated with the variables
            dates.append(self.y.index[i])
            const.append(ridge_model.named_steps["model"].intercept_)
            betas.append(ridge_model.named_steps["model"].coef_)
            fitted.append(y_pred)

            # Prediction for a given horizon

            # Store the performance metrics in the result

        # Retrieve the residuals
        self.results = (dates, const, betas, fitted)

        # Method to perform the model prediction

    # Class for random forest model
