"""Temporal Purge Gap Validation Splitter for Time-Series CV.

This module implements a custom expanding-window cross-validation splitter
designed specifically for meteorological and renewable energy forecasting.
It enforces a mandatory purge gap between training and validation/testing slices
to eliminate look-ahead bias and data leakage caused by atmospheric persistence.
"""

import logging
from pathlib import Path
from typing import Tuple, Generator, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PurgedExpandingWindowSplitter:
    """TimeSeriesSplit cross-validation generator with an explicit temporal purge gap.

    It yields index splits where each fold consists of an expanding training window,
    followed by a temporal purge gap, followed by a fixed-duration test window.
    """

    def __init__(
        self,
        n_splits: int = 4,
        test_duration: Union[pd.Timedelta, str] = pd.Timedelta(days=365),
        purge_gap: Union[pd.Timedelta, str] = pd.Timedelta(hours=48),
    ) -> None:
        """Initializes the PurgedExpandingWindowSplitter.

        Args:
            n_splits: Number of cross-validation folds.
            test_duration: Duration of each validation/testing window.
            purge_gap: Temporal gap size (embargo) between train and test windows.
        """
        self.n_splits = n_splits
        self.test_duration = pd.to_timedelta(test_duration)
        self.purge_gap = pd.to_timedelta(purge_gap)

    def split(self, X: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates train and test indices for cross-validation folds.

        Args:
            X: Input DataFrame with a temporal index (pd.DatetimeIndex).

        Yields:
            A tuple of (train_indices, test_indices) as integer arrays.

        Raises:
            ValueError: If the index is not a pd.DatetimeIndex or dataset duration
                        is too short for the requested folds and purge gaps.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame index must be a pandas DatetimeIndex to enforce temporal purging."
            )

        dts = X.index
        t_start = dts.min()
        t_end = dts.max()
        total_duration = t_end - t_start

        # Minimum required dataset duration: n_splits * test_duration + purge_gap
        required_duration = self.n_splits * self.test_duration + self.purge_gap
        if total_duration < required_duration:
            raise ValueError(
                f"Dataset duration ({total_duration.days} days) is too short for "
                f"{self.n_splits} splits of {self.test_duration.days} days and a "
                f"{self.purge_gap.total_seconds() / 3600:.1f}-hour purge gap. "
                f"Required minimum duration is {required_duration.days} days."
            )

        logger.info(
            f"Configured PurgedExpandingWindowSplitter: n_splits={self.n_splits}, "
            f"test_duration={self.test_duration.days} days, purge_gap={self.purge_gap.total_seconds() / 3600:.1f} hours."
        )

        for i in range(self.n_splits):
            # Calculate test window boundary dates (moving backwards from the end of the timeline)
            test_end_dt = t_end - (self.n_splits - 1 - i) * self.test_duration
            test_start_dt = test_end_dt - self.test_duration
            
            # The training window ends at test_start minus the purge gap
            train_end_dt = test_start_dt - self.purge_gap

            # Construct boolean masks for training and testing
            train_mask = (dts >= t_start) & (dts <= train_end_dt)
            test_mask = (dts >= test_start_dt) & (dts <= test_end_dt)

            # Convert boolean masks to integer index arrays
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) == 0:
                raise ValueError(
                    f"Fold {i} training set is empty. Increase dataset size or reduce fold sizes."
                )
            if len(test_indices) == 0:
                raise ValueError(
                    f"Fold {i} test set is empty. Check dataset timeline continuity."
                )

            yield train_indices, test_indices


def visualize_splits(X: pd.DataFrame, splitter: PurgedExpandingWindowSplitter) -> None:
    """Logs details and prints a visual timeline representation of each cross-validation fold.

    Args:
        X: Input DataFrame with DatetimeIndex.
        splitter: An instance of PurgedExpandingWindowSplitter.
    """
    dts = X.index
    t_start = dts.min()
    t_end = dts.max()
    total_len = len(X)

    print("\n" + "=" * 80)
    print("Purged Expanding Window Cross-Validation Splitter Visualization")
    print("=" * 80)
    print(f"Dataset start: {t_start}")
    print(f"Dataset end:   {t_end}")
    print(f"Total samples: {total_len}\n")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        train_start = dts[train_idx].min()
        train_end = dts[train_idx].max()
        test_start = dts[test_idx].min()
        test_end = dts[test_idx].max()

        # Calculate actual gap
        gap_start = train_end
        gap_end = test_start
        actual_gap_duration = gap_end - gap_start

        # Confirm no overlap
        overlap_ok = train_end < test_start

        print(f"Fold {fold}:")
        print(f"  Train: {train_start} to {train_end} ({len(train_idx)} samples)")
        print(f"  Gap:   {gap_start} to {gap_end} ({actual_gap_duration})")
        print(f"  Test:  {test_start} to {test_end} ({len(test_idx)} samples)")
        print(f"  Overlap Check Passed: {overlap_ok}")

        # Generate text timeline bar
        bar_chars = []
        for pct in np.linspace(0, 100, 30):
            idx_val = int(pct / 100.0 * (total_len - 1))
            if idx_val in train_idx:
                bar_chars.append("T")
            elif idx_val in test_idx:
                bar_chars.append("V")
            elif len(train_idx) > 0 and len(test_idx) > 0 and train_idx[-1] < idx_val < test_idx[0]:
                bar_chars.append("G")
            else:
                bar_chars.append(".")

        print(f"  Timeline: [{' '.join(bar_chars)}]\n")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)

    # Test the validation splitter using actual consolidated generation data
    actuals_path = Path("data/processed/actual_generation_50hertz.parquet")
    
    if actuals_path.exists():
        logger.info(f"Loading actuals from {actuals_path} for live validation splitter test...")
        df = pd.read_parquet(actuals_path)
        # Set timestamp as DatetimeIndex
        df = df.set_index(pd.DatetimeIndex(df['timestamp_utc']))
        
        # Test with 4 splits, 365-day test size, 72-hour purge gap
        test_splitter = PurgedExpandingWindowSplitter(
            n_splits=4,
            test_duration=pd.Timedelta(days=365),
            purge_gap=pd.Timedelta(hours=72)
        )
        visualize_splits(df, test_splitter)
    else:
        logger.warning(
            f"Could not locate actuals file at {actuals_path} for testing. "
            "Simulating dummy data instead..."
        )
        # Create dummy datetimes
        dummy_idx = pd.date_range(start="2022-01-01", end="2026-06-01", freq="15min")
        dummy_df = pd.DataFrame(index=dummy_idx)
        
        test_splitter = PurgedExpandingWindowSplitter(
            n_splits=4,
            test_duration="365D",
            purge_gap="72H"
        )
        visualize_splits(dummy_df, test_splitter)
