"""Baseline forecasting models for DNH_total residential water demand."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


class SeasonalNaiveBaseline:
    """Forecast each month using the observation from the same month in the previous year."""

    def __init__(self, seasonal_period: int = 12):
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be positive")
        self.seasonal_period = int(seasonal_period)
        self._reference = None

    def fit(self, y: pd.Series):
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if not y.index.is_monotonic_increasing:
            raise ValueError("y must be sorted chronologically")
        self._reference = y.copy()
        return self

    def predict(self, dates: Iterable[pd.Timestamp] | pd.DatetimeIndex):
        if self._reference is None:
            raise ValueError("Model must be fit before predicting")

        idx = pd.DatetimeIndex(list(dates))
        if len(idx) == 0:
            return pd.Series(dtype=float, index=idx)

        ordered_dates = idx.sort_values()
        generated = {}
        predictions = {}

        for ts in ordered_dates:
            candidate = ts - pd.DateOffset(months=self.seasonal_period)

            if candidate in self._reference.index:
                pred_value = self._reference.loc[candidate]
            elif candidate in generated:
                pred_value = generated[candidate]
            else:
                prior_same_month = self._reference[
                    self._reference.index.month == ts.month
                ]
                if prior_same_month.empty:
                    raise ValueError(
                        f"No seasonal training value available for month {ts.month} in the prior cycle"
                    )
                pred_value = prior_same_month.iloc[-1]

            generated[ts] = pred_value
            predictions[ts] = pred_value

        ordered_pred = pd.Series(
            [predictions[ts] for ts in idx],
            index=idx,
        )
        return ordered_pred
