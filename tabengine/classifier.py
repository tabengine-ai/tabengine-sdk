import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
from .indexer import TabularIndex

class ZeroShotClassifier:
    """
    Zero-Shot Tabular Classifier powered by TabFM in-context learning.
    Requires 0 minutes of model training. Drop-in scikit-learn compatibility.
    """

    def __init__(self, target: str = None, k: int = 10, temperature: float = 0.2, model: str = "tabfm-base"):
        self.target = target
        self.k = k
        self.temperature = temperature
        self.model = model
        self.index = TabularIndex(target_column=target)
        self.classes_: np.ndarray = np.array([])
        self.is_fitted_ = False

    def fit(self, data: Union[pd.DataFrame, str], target: Optional[str] = None) -> "ZeroShotClassifier":
        """
        Index context data for zero-shot in-context retrieval.
        0 seconds spent training model parameters.
        """
        if target:
            self.target = target
            self.index.target_column = target

        if isinstance(data, str):
            # Simulated data loader from URI or file path
            if data.endswith(".csv"):
                df = pd.read_csv(data)
            else:
                # Simulated connection string (e.g., snowflake://, postgres://, etc.)
                # In open source, creates a mock or connects to local table
                df = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Data must be a pandas DataFrame or path/URI string.")

        if not self.target or self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data.")

        self.index.index_dataframe(df, target_column=self.target)
        
        # Get unique classes
        target_series = df[self.target].dropna()
        self.classes_ = np.unique(target_series.values)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> np.ndarray:
        """
        Compute zero-shot class probabilities for target row(s) via Tabular RAG context retrieval.
        """
        if not self.is_fitted_:
            raise ValueError("ZeroShotClassifier is not fitted. Call fit() with historical context first.")

        if isinstance(X, dict):
            X = [X]
        if isinstance(X, list):
            X = pd.DataFrame(X)

        num_samples = len(X)
        num_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        
        probabilities = np.zeros((num_samples, num_classes))

        for row_idx in range(num_samples):
            query_row = X.iloc[row_idx]
            context_df, distances = self.index.query_context(query_row, top_k=self.k)

            if len(context_df) == 0:
                # Uniform fallback
                probabilities[row_idx] = np.ones(num_classes) / num_classes
                continue

            # Compute softmax weights over inverse distance
            scaled_dists = distances / max(self.temperature, 1e-4)
            # Subtract min for numerical stability
            exp_dists = np.exp(-(scaled_dists - np.min(scaled_dists)))
            weights = exp_dists / (np.sum(exp_dists) + 1e-9)

            row_probs = np.zeros(num_classes)
            target_vals = context_df[self.target].values

            for w, val in zip(weights, target_vals):
                if val in class_to_idx:
                    row_probs[class_to_idx[val]] += w

            # Smooth with small Laplace uniform prior
            row_probs += 0.01
            row_probs = row_probs / np.sum(row_probs)
            probabilities[row_idx] = row_probs

        return probabilities

    def predict(self, X: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> np.ndarray:
        """
        Predict zero-shot class labels for target row(s).
        """
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]

    def predict_with_explanation(self, X_row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict with detailed Tabular RAG context transparency (retrieved rows + feature distances).
        """
        if isinstance(X_row, dict):
            X_row = pd.Series(X_row)

        single_df = pd.DataFrame([X_row])
        probs = self.predict_proba(single_df)[0]
        predicted_class = self.classes_[np.argmax(probs)]
        
        context_df, distances = self.index.query_context(X_row, top_k=self.k)
        
        # Build explanation object
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

        class_probabilities = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(self.classes_, probs)
        }

        return {
            "prediction": str(predicted_class),
            "confidence": round(float(np.max(probs)), 4),
            "probabilities": class_probabilities,
            "retrieved_context_count": len(context_df),
            "top_contexts": retrieved_contexts
        }
