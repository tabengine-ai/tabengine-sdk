import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Optional

class MixedTypeDistanceEngine:
    """
    Ultra-Fast Vectorized Hybrid Distance Engine.
    Pre-computes numerical and categorical matrices for sub-millisecond
    distance calculations across mixed tabular data types.
    Handles empty spreadsheet cells ("") safely without exceptions.
    """

    def __init__(self, feature_columns: List[str] = None):
        self.feature_columns = feature_columns or []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        
        self.numeric_matrix: Optional[np.ndarray] = None
        self.categorical_matrix: Optional[np.ndarray] = None
        self.ranges_vec: np.ndarray = np.array([])
        self.mins_vec: np.ndarray = np.array([])
        
        self.mins: Dict[str, float] = {}
        self.ranges: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None):
        """Fit distance scaling parameters and pre-compute vectorized matrices."""
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

        # 1. Pre-compute numerical matrix & scaling vectors
        if self.numeric_cols:
            num_df = df[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
            self.numeric_matrix = num_df.to_numpy(dtype=np.float32)
            
            mins = np.nanmin(self.numeric_matrix, axis=0) if len(self.numeric_matrix) > 0 else np.zeros(len(self.numeric_cols))
            maxs = np.nanmax(self.numeric_matrix, axis=0) if len(self.numeric_matrix) > 0 else np.ones(len(self.numeric_cols))
            
            mins = np.where(np.isnan(mins), 0.0, mins)
            maxs = np.where(np.isnan(maxs), 1.0, maxs)
            ranges = maxs - mins
            ranges[ranges <= 0] = 1.0
            
            self.mins_vec = mins.astype(np.float32)
            self.ranges_vec = ranges.astype(np.float32)

            for i, col in enumerate(self.numeric_cols):
                self.mins[col] = float(mins[i])
                self.ranges[col] = float(ranges[i])
        else:
            self.numeric_matrix = np.empty((len(df), 0), dtype=np.float32)
            self.mins_vec = np.array([], dtype=np.float32)
            self.ranges_vec = np.array([], dtype=np.float32)

        # 2. Pre-compute categorical string matrix
        if self.categorical_cols:
            self.categorical_matrix = df[self.categorical_cols].astype(str).to_numpy()
        else:
            self.categorical_matrix = np.empty((len(df), 0), dtype=str)

    def compute_distances(self, ref_df: pd.DataFrame, query_row: Union[pd.Series, dict]) -> np.ndarray:
        """
        Compute fully vectorized Gower / Manhattan distance in sub-millisecond time.
        Safely converts query row features and handles null / empty values.
        """
        if isinstance(query_row, dict):
            query_row = pd.Series(query_row)

        num_samples = len(ref_df)
        if num_samples == 0:
            return np.array([])

        total_features = len(self.numeric_cols) + len(self.categorical_cols)
        if total_features == 0:
            return np.zeros(num_samples, dtype=np.float32)

        accumulated_dist = np.zeros(num_samples, dtype=np.float32)
        
        # 1. Vectorized Numerical Distance with Safe Float Conversion
        if len(self.numeric_cols) > 0 and self.numeric_matrix is not None:
            q_num_vals = []
            for col in self.numeric_cols:
                val = query_row.get(col, np.nan)
                if val == "" or val is None or pd.isna(val) or val == "nan" or val == "None":
                    q_num_vals.append(np.nan)
                else:
                    try:
                        q_num_vals.append(float(val))
                    except (ValueError, TypeError):
                        q_num_vals.append(np.nan)

            q_num = np.array(q_num_vals, dtype=np.float32)

            # Mask NaNs in query
            nan_query_mask = np.isnan(q_num)
            q_num_filled = np.where(nan_query_mask, 0.0, q_num)

            # Compute absolute differences broadcasted over matrix
            diffs = np.abs(self.numeric_matrix - q_num_filled) / self.ranges_vec
            
            # Handle NaN in matrix or query
            nan_matrix_mask = np.isnan(self.numeric_matrix)
            diffs[nan_matrix_mask | nan_query_mask] = 0.5
            
            accumulated_dist += np.sum(diffs, axis=1)

        # 2. Vectorized Categorical Distance
        if len(self.categorical_cols) > 0 and self.categorical_matrix is not None:
            q_cat = np.array([
                str(query_row.get(col, "")) if query_row.get(col, None) is not None else ""
                for col in self.categorical_cols
            ], dtype=str)

            # Mask missing query categoricals
            missing_q_mask = (q_cat == "") | (q_cat == "nan") | (q_cat == "None")
            
            cat_diffs = (self.categorical_matrix != q_cat).astype(np.float32)
            cat_diffs[:, missing_q_mask] = 0.5
            
            accumulated_dist += np.sum(cat_diffs, axis=1)

        return accumulated_dist / float(total_features)
