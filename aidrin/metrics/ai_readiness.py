import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from scipy.stats import entropy

class AIRreadinessMetrics:
    """Computes AI-specific data readiness metrics such as feature importance, correlations, fairness, and class imbalance."""
    
    @staticmethod
    def feature_correlations(df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculates Pearson correlation matrix for numerical features and Theil's U for categorical features."""
        from dython.nominal import theils_u
        import polars.selectors as cs
        
        numeric_df = df.select(cs.numeric()).fill_nan(None).fill_null(strategy="mean")
        cat_df = df.select(cs.string() | cs.categorical() | cs.boolean())
        
        result = {}
        
        # Pearson for numeric
        if numeric_df.width > 1:
            corr_matrix = numeric_df.corr()
            columns = corr_matrix.columns
            for i, col1 in enumerate(columns):
                result[col1] = {}
                for j, col2 in enumerate(columns):
                    val = corr_matrix[col1][j]
                    if col1 != col2 and val is not None and not np.isnan(val):
                         result[col1][col2] = float(val)
                         
        # Theil's U for categorical
        if cat_df.width > 1:
            cat_pd = cat_df.to_pandas()
            # Drop nulls for valid calculation
            cat_pd = cat_pd.dropna()
            if not cat_pd.empty:
                for col1 in cat_pd.columns:
                    if col1 not in result:
                        result[col1] = {}
                    for col2 in cat_pd.columns:
                        if col1 != col2:
                            val = theils_u(cat_pd[col1], cat_pd[col2])
                            if not np.isnan(val):
                                result[col1][col2] = float(val)
                     
        return result

    @staticmethod
    def class_imbalance(df: pl.DataFrame, target_col: str) -> Dict[str, Any]:
        """Calculates Imbalance Degree (ID) for class imbalance."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
            
        value_counts = df[target_col].value_counts().sort(target_col)
        counts = value_counts.select(pl.col("count")).to_series().to_numpy()
        labels = value_counts[target_col].to_list()
        
        if len(counts) <= 1:
            return {"imbalance_degree_score": 0.0, "is_imbalanced": True, "details": "Only 1 class present."}
            
        # Simplified Imbalance Degree formulation representing deviation from perfectly balanced distribution
        total = np.sum(counts)
        num_classes = len(counts)
        expected = total / num_classes
        
        # A variant of ID uses sum of absolute differences normalized
        id_score = np.sum(np.abs(counts - expected)) / (2 * total * (1 - 1/num_classes)) if num_classes > 1 else 0.0
        
        return {
            "imbalance_degree_score": float(id_score),
            "is_imbalanced": bool(id_score > 0.2), # Threshold can vary
            "class_counts": {str(k): int(v) for k, v in zip(labels, counts)}
        }

    @staticmethod
    def feature_importance(df: pl.DataFrame, target_col: str, task_type: str = "classification") -> Dict[str, float]:
        """Calculates Shapley Values using a Random Forest model."""
        import shap
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
            
        df_clean = df.drop_nulls(subset=[target_col])
        if df_clean.height == 0:
            return {}
            
        y = df_clean[target_col].to_numpy()
        
        if y.dtype.kind in {'U', 'O', 'S', 'b'} and task_type == "classification":
            y = LabelEncoder().fit_transform(y)
            
        import polars.selectors as cs
        X_df = df_clean.select(cs.numeric()).fill_nan(None).fill_null(strategy="mean")
        
        if X_df.width == 0:
             return {}
             
        X = X_df.to_pandas()
        
        if task_type == "classification":
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        else:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            
        rf.fit(X, y)
        
        # Sample for SHAP if dataset is too large
        X_sample = X.sample(n=min(500, len(X)), random_state=42) if len(X) > 500 else X
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        
        # Depending on shap version and task_type, shap_values structure varies
        if isinstance(shap_values, list): # Multi-class classification
            # shap_values is a list of arrays (one per class). We average over all samples and classes.
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3: # Another multiclass format
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else: # Binary classification or Regression
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
             
        # Cast to float natively to avoid numpy 0-d array issues
        importance = {col: float(np.mean(score)) for col, score in zip(X.columns, mean_abs_shap)}
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def fairness_bias(df: pl.DataFrame, target_col: str, protected_attribute: str, positive_label: Any = 1) -> Dict[str, Any]:
        """Calculates Target Standard Deviation (TSD) for non-binary fairness."""
        if target_col not in df.columns or protected_attribute not in df.columns:
             return {"error": f"Target '{target_col}' or protected attribute '{protected_attribute}' missing."}
             
        df_clean = df.drop_nulls(subset=[target_col, protected_attribute])
        if df_clean.height == 0:
             return {"error": "Empty dataset after dropping nulls."}
             
        pdf = df_clean.select([protected_attribute, target_col]).to_pandas()
        
        # Calculate Pr(Y = Y1 | A = n)
        # We find the global probability first
        overall_prob = (pdf[target_col] == positive_label).mean()
        
        groups = pdf[protected_attribute].unique()
        if len(groups) < 2:
             return {"error": "Protected attribute has less than 2 distinct values."}
             
        group_probs = []
        for g in groups:
            g_data = pdf[pdf[protected_attribute] == g]
            if len(g_data) > 0:
                prob = (g_data[target_col] == positive_label).mean()
                group_probs.append(prob)
                
        # TSD Formula
        N = len(group_probs)
        mu = np.mean(group_probs) # or overall_prob
        tsd = np.sqrt(np.sum([(p - mu)**2 for p in group_probs]) / N)
        
        return {
            "target_standard_deviation_tsd": float(tsd),
            "potential_bias_detected": bool(tsd > 0.1), # Define a threshold
            "group_probabilities": {str(g): float(p) for g, p in zip(groups, group_probs)}
        }

    @staticmethod
    def evaluate_all(df: pl.DataFrame, target_col: Optional[str] = None, protected_attribute: Optional[str] = None, task_type: str = "classification", positive_label: Any = 1) -> Dict[str, Any]:
        """Evaluates all AI readiness metrics."""
        results = {
            "feature_correlations": AIRreadinessMetrics.feature_correlations(df)
        }
        
        if target_col:
             try:
                 results["class_imbalance"] = AIRreadinessMetrics.class_imbalance(df, target_col)
                 results["feature_importance"] = AIRreadinessMetrics.feature_importance(df, target_col, task_type)
             except Exception as e:
                 results["target_error"] = str(e)
                 
             if protected_attribute:
                 results["fairness_bias"] = AIRreadinessMetrics.fairness_bias(df, target_col, protected_attribute, positive_label)
                 
        return results
