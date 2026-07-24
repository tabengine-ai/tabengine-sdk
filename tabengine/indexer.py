import duckdb
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from .distance import MixedTypeDistanceEngine

class TabularIndex:
    """
    Sub-10ms Tabular Vector / Distance Indexer powered by DuckDB & Hybrid Distance Engine.
    Indexes tabular data in-memory or on disk for zero-shot context retrieval.
    """

    def __init__(self, target_column: str = None):
        self.target_column = target_column
        self.feature_columns: List[str] = []
        self.conn = duckdb.connect(database=":memory:")
        self.df_data: pd.DataFrame = None
        self.distance_engine = MixedTypeDistanceEngine()
        self.is_indexed = False

    def index_dataframe(self, df: pd.DataFrame, target_column: str = None) -> "TabularIndex":
        """Index a pandas DataFrame into DuckDB and setup distance engine."""
        if target_column:
            self.target_column = target_column

        self.df_data = df.copy()
        
        # Determine feature columns
        if self.target_column and self.target_column in self.df_data.columns:
            self.feature_columns = [c for c in self.df_data.columns if c != self.target_column]
        else:
            self.feature_columns = list(self.df_data.columns)

        # Register dataframe in DuckDB
        self.conn.register("raw_table", self.df_data)
        
        # Fit distance engine
        self.distance_engine.fit(self.df_data, feature_columns=self.feature_columns)
        self.is_indexed = True
        return self

    def query_context(self, query_row: Union[pd.Series, dict], top_k: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Retrieve the top-K most informative historical context rows and their similarity/distance scores.
        Returns (context_df, distance_scores).
        """
        if not self.is_indexed or self.df_data is None:
            raise ValueError("TabularIndex has not been indexed yet. Call index_dataframe() first.")

        if isinstance(query_row, dict):
            query_row = pd.Series(query_row)

        distances = self.distance_engine.compute_distances(self.df_data, query_row)
        
        # Select top-k smallest distances
        k = min(top_k, len(self.df_data))
        if k == 0:
            return pd.DataFrame(), np.array([])

        top_k_indices = np.argpartition(distances, k - 1)[:k]
        sorted_top_k = top_k_indices[np.argsort(distances[top_k_indices])]

        context_df = self.df_data.iloc[sorted_top_k].copy()
        top_distances = distances[sorted_top_k]

        return context_df, top_distances

    def get_summary(self) -> Dict[str, Any]:
        """Return schema summary of the indexed table."""
        if self.df_data is None:
            return {"indexed": False}
            
        col_types = {col: str(dtype) for col, dtype in self.df_data.dtypes.items()}
        return {
            "indexed": True,
            "num_rows": len(self.df_data),
            "num_columns": len(self.df_data.columns),
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "numeric_features": self.distance_engine.numeric_cols,
            "categorical_features": self.distance_engine.categorical_cols,
            "column_types": col_types
        }
