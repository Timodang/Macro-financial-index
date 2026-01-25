"""
Time Series Cross-Validation (Dependent / Forward-Chaining)

Stateless expanding-window CV:
- train = past
- test  = future
- optional gap/embargo between train and test
- sklearn-compatible: split(X, y=None, groups=None) + get_n_splits()

Notes:
- Use gap = horizon + max_lag (or more) when you build y_{t+h} and/or lagged features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Generator, Optional, Tuple, Union


IntOrFloat = Union[int, float]


class DependentTimeSeriesCV:
    """
    Expanding-window cross-validation for time series (forward chaining).

    Parameters
    ----------
    n_splits : int
        Number of folds (>= 2)
    test_size : int or float, optional
        If int: number of observations in each test fold.
        If float: proportion of samples in each test fold.
        If None: inferred from n_splits and min_train_size.
    min_train_size : int, optional
        Minimum number of observations in the initial training set.
        If None: inferred from n_splits, test_size and gap.
    gap : int, default=0
        Number of observations to skip between train and test (embargo).
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[IntOrFloat] = None,
        min_train_size: Optional[int] = None,
        gap: int = 0,
    ):
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an int >= 2")
        if test_size is not None and not isinstance(test_size, (int, float)):
            raise TypeError("test_size must be int, float, or None")
        if isinstance(test_size, int) and test_size < 1:
            raise ValueError("test_size as int must be >= 1")
        if isinstance(test_size, float) and not (0.0 < test_size < 1.0):
            raise ValueError("test_size as float must be in (0, 1)")
        if min_train_size is not None and (not isinstance(min_train_size, int) or min_train_size < 1):
            raise ValueError("min_train_size must be an int >= 1 or None")
        if not isinstance(gap, int) or gap < 0:
            raise ValueError("gap must be an int >= 0")

        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap

    def _resolve_test_size(self, n_samples: int) -> int:
        """Return an integer test_size in number of observations."""
        if self.test_size is None:
            # fallback will be computed later once min_train_size is known
            return -1  # sentinel
        if isinstance(self.test_size, float):
            ts = int(np.floor(n_samples * self.test_size))
            return max(ts, 1)
        return int(self.test_size)

    def _resolve_min_train_size(self, n_samples: int, test_size: int) -> int:
        """Return an integer min_train_size in number of observations."""
        if self.min_train_size is not None:
            return int(self.min_train_size)

        if test_size == -1:
            # If both test_size and min_train_size are None:
            # choose a simple heuristic: initial train = n_samples // (n_splits + 1),
            # remaining split equally across folds as test blocks.
            min_train = n_samples // (self.n_splits + 1)
            if min_train < 1:
                raise ValueError("Not enough samples to infer min_train_size.")
            return min_train

        # Otherwise infer min_train so that we can fit all folds
        min_train = n_samples - (self.n_splits * test_size) - (self.n_splits * self.gap)
        if min_train < 1:
            raise ValueError(
                f"Not enough data for n_splits={self.n_splits}, "
                f"test_size={test_size}, gap={self.gap} (computed min_train_size={min_train}). "
                "Reduce n_splits/test_size/gap."
            )
        return min_train

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups=None,  # sklearn compatibility
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to split.")

        test_size = self._resolve_test_size(n_samples)
        min_train_size = self._resolve_min_train_size(n_samples, test_size)

        if test_size == -1:
            # compute test_size now that min_train_size is known
            remaining = n_samples - min_train_size
            test_size = remaining // self.n_splits
            if test_size < 1:
                raise ValueError(
                    f"Not enough remaining samples after min_train_size={min_train_size} "
                    f"to create {self.n_splits} test folds."
                )

        # sanity check again with resolved values
        max_needed = min_train_size + self.n_splits * test_size + self.n_splits * self.gap
        if max_needed > n_samples:
            raise ValueError(
                f"Resolved configuration exceeds n_samples: "
                f"min_train_size={min_train_size}, test_size={test_size}, gap={self.gap}, "
                f"n_splits={self.n_splits}, n_samples={n_samples}."
            )

        for i in range(self.n_splits):
            train_end = min_train_size + i * test_size
            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            train_idx = np.arange(0, train_end, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class TimeSeriesTrainTestSplit:
    """
    Single train/test split preserving time order.

    Parameters
    ----------
    test_size : float or int
        If float: proportion of data used for testing.
        If int: number of observations used for testing.
    gap : int, default=0
        Number of observations to skip between train and test (embargo).
    """

    def __init__(self, test_size: IntOrFloat = 0.2, gap: int = 0):
        if isinstance(test_size, float):
            if not (0.0 < test_size < 1.0):
                raise ValueError("test_size as float must be in (0, 1)")
        elif isinstance(test_size, int):
            if test_size < 1:
                raise ValueError("test_size as int must be >= 1")
        else:
            raise TypeError("test_size must be float or int")

        if not isinstance(gap, int) or gap < 0:
            raise ValueError("gap must be an int >= 0")

        self.test_size = test_size
        self.gap = gap

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to split.")

        if isinstance(self.test_size, float):
            n_test = int(np.floor(n_samples * self.test_size))
            n_test = max(n_test, 1)
        else:
            n_test = int(self.test_size)

        test_start = n_samples - n_test
        train_end = test_start - self.gap

        if train_end < 1:
            raise ValueError(
                f"Not enough data: n_samples={n_samples}, n_test={n_test}, gap={self.gap}. "
                "Reduce test_size or gap."
            )

        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, n_samples, dtype=int)
        return train_idx, test_idx


def plot_cv_indices(cv, X: pd.DataFrame, y: Optional[pd.Series] = None, ax=None, lw: int = 10):
    """
    Quick visualization of CV splits.
    (kept simple; uses matplotlib defaults)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    splits = list(cv.split(X, y))
    for ii, (train_idx, test_idx) in enumerate(splits):
        ax.scatter(train_idx, [ii] * len(train_idx), marker="_", lw=lw, label="Train" if ii == 0 else "")
        ax.scatter(test_idx, [ii] * len(test_idx), marker="_", lw=lw, label="Test" if ii == 0 else "")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("CV iteration")
    ax.set_title("Time Series CV Splits")
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels(range(1, len(splits) + 1))
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return ax
