import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
from .indexer import TabularIndex

class ZeroShotRegressor:
    """
    Zero-Shot Tabular Regressor powered by TabFM in-context continuous prediction.
    Requires 0 minutes of model training. Drop-in scikit-learn compatibility.
    """

    def __init__(self, target: str = None, k: int = 10, temperature: float = 0.2, model: str = "tabfm-base"):
        self.target = target
        self.k = k
        self.temperature = temperature
        self.model = model
        self.index = TabularIndex(target_column=target)
        self.is_fitted_ = False

    def fit(self, data: Union[pd.DataFrame, str], target: Optional[str] = None) -> "ZeroShotRegressor":
        """
        Index context data for zero-shot in-context continuous prediction.
        """
        if target:
            self.target = target
            self.index.target_column = target

        if isinstance(data, str):
            if data.endswith(".csv"):
                df = pd.read_csv(data)
            else:
                df = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Data must be a pandas DataFrame or path/URI string.")

        if not self.target or self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data.")

        self.index.index_dataframe(df, target_column=self.target)
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> np.ndarray:
        """
        Compute zero-shot numerical prediction for target row(s).
        """
        if not self.is_fitted_:
            raise ValueError("ZeroShotRegressor is not fitted. Call fit() first.")

        if isinstance(X, dict):
            X = [X]
        if isinstance(X, list):
            X = pd.DataFrame(X)

        num_samples = len(X)
        predictions = np.zeros(num_samples)

        for row_idx in range(num_samples):
            query_row = X.iloc[row_idx]
            context_df, distances = self.index.query_context(query_row, top_k=self.k)

            if len(context_df) == 0:
                predictions[row_idx] = 0.0
                continue

            target_vals = pd.to_numeric(context_df[self.target], errors='coerce').values
            valid_mask = ~np.isnan(target_vals)
            
            if not np.any(valid_mask):
                predictions[row_idx] = 0.0
                continue

            target_vals = target_vals[valid_mask]
            valid_dists = distances[valid_mask]

            # Compute softmax weights over inverse distance
            scaled_dists = valid_dists / max(self.temperature, 1e-4)
            exp_dists = np.exp(-(scaled_dists - np.min(scaled_dists)))
            weights = exp_dists / (np.sum(exp_dists) + 1e-9)

            # Weighted mean prediction
            pred_val = np.sum(weights * target_vals)
            predictions[row_idx] = pred_val

        return predictions

    def predict_with_explanation(self, X_row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict with detailed context explanation (top matching rows, weights, range).
        """
        if isinstance(X_row, dict):
            X_row = pd.Series(X_row)

        single_df = pd.DataFrame([X_row])
        pred_val = float(self.predict(single_df)[0])
        
        context_df, distances = self.index.query_context(X_row, top_k=self.k)
        target_vals = pd.to_numeric(context_df[self.target], errors='coerce').dropna().values

        retrieved_contexts = []
        for idx in range(len(context_df)):
            ctx_row = context_df.iloc[idx].to_dict()
            dist = float(distances[idx])
            retrieved_contexts.append({
                "rank": idx + 1,
                "distance": round(dist, 4),
                "similarity_pct": round(max(0.0, (1.0 - dist) * 100), 1),
                "data": ctx_row
            })

        val_min = float(np.min(target_vals)) if len(target_vals) > 0 else pred_val
        val_max = float(np.max(target_vals)) if len(target_vals) > 0 else pred_val
        val_std = float(np.std(target_vals)) if len(target_vals) > 1 else 0.0

        return {
            "prediction": round(pred_val, 2),
            "context_range": {"min": round(val_min, 2), "max": round(val_max, 2)},
            "prediction_std_dev": round(val_std, 2),
            "retrieved_context_count": len(context_df),
            "top_contexts": retrieved_contexts
        }
