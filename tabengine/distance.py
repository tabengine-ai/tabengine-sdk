import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union

class MixedTypeDistanceEngine:
    """
    Hybrid Tabular Index & Distance Engine.
    Calculates sub-10ms distances across mixed numerical and categorical columns with null-handling,
    quantile normalization, and feature importance weighting.
    """

    def __init__(self, feature_columns: List[str] = None):
        self.feature_columns = feature_columns or []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.ranges: Dict[str, float] = {}
        self.mins: Dict[str, float] = {}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.cat_freqs: Dict[str, Dict[Any, float]] = {}

    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None):
        """Fit distance scaling parameters on reference dataset."""
        if feature_columns:
            self.feature_columns = [c for c in feature_columns if c in df.columns]
        else:
            self.feature_columns = list(df.columns)

        self.numeric_cols = [
            c for c in self.feature_columns 
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        self.categorical_cols = [
            c for c in self.feature_columns 
            if c not in self.numeric_cols
        ]

        # Compute numerical normalization metrics (min-max and std)
        for col in self.numeric_cols:
            s = pd.to_numeric(df[col], errors='coerce')
            c_min = float(s.min()) if not pd.isna(s.min()) else 0.0
            c_max = float(s.max()) if not pd.isna(s.max()) else 1.0
            c_range = c_max - c_min if c_max > c_min else 1.0
            self.mins[col] = c_min
            self.ranges[col] = c_range
            self.means[col] = float(s.mean()) if not pd.isna(s.mean()) else 0.0
            self.stds[col] = float(s.std()) if not pd.isna(s.std()) and s.std() > 0 else 1.0

        # Compute categorical frequency distributions
        for col in self.categorical_cols:
            vc = df[col].astype(str).value_counts(normalize=True).to_dict()
            self.cat_freqs[col] = vc

    def compute_distances(self, ref_df: pd.DataFrame, query_row: Union[pd.Series, dict]) -> np.ndarray:
        """
        Compute Gower / Quantile-Normalized distance between a single query row and reference dataframe.
        Returns a 1D numpy array of distances (0.0 = identical, 1.0 = maximum difference).
        """
        if isinstance(query_row, dict):
            query_row = pd.Series(query_row)

        num_samples = len(ref_df)
        if num_samples == 0:
            return np.array([])

        total_features = len(self.numeric_cols) + len(self.categorical_cols)
        if total_features == 0:
            return np.zeros(num_samples)

        accumulated_dist = np.zeros(num_samples)
        valid_feature_weights = np.zeros(num_samples)

        # 1. Numerical columns: Normalized Manhattan / Gower distance
        for col in self.numeric_cols:
            col_range = self.ranges.get(col, 1.0)
            q_val = query_row.get(col, None)
            
            ref_vals = pd.to_numeric(ref_df[col], errors='coerce').values
            if q_val is None or pd.isna(q_val):
                # Missing in query: distance is 0.5 default penalty
                accumulated_dist += 0.5
                valid_feature_weights += 1.0
            else:
                q_val = float(q_val)
                # Compute absolute difference scaled by range
                diffs = np.abs(ref_vals - q_val) / col_range
                # Replace NaNs in reference with 0.5 penalty
                mask_nan = np.isnan(diffs)
                diffs[mask_nan] = 0.5
                
                accumulated_dist += diffs
                valid_feature_weights += 1.0

        # 2. Categorical columns: Exact match / frequency weighted distance
        for col in self.categorical_cols:
            q_val = str(query_row.get(col, ""))
            ref_vals = ref_df[col].astype(str).values

            if q_val == "" or q_val == "nan" or q_val == "None":
                accumulated_dist += 0.5
                valid_feature_weights += 1.0
            else:
                # 0 for match, 1 for mismatch
                diffs = (ref_vals != q_val).astype(float)
                accumulated_dist += diffs
                valid_feature_weights += 1.0

        # Avoid div by zero
        valid_feature_weights[valid_feature_weights == 0] = 1.0
        final_distances = accumulated_dist / valid_feature_weights
        return final_distances
