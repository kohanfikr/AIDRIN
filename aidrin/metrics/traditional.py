import polars as pl
from typing import Dict, Any

class TraditionalMetrics:
    """Computes traditional data quality metrics such as missing values, duplicates, and outliers."""
    
    @staticmethod
    def missing_values(df: pl.DataFrame) -> Dict[str, float]:
        """Calculates the ratio of missing values per column."""
        total_rows = df.height
        if total_rows == 0:
            return {col: 0.0 for col in df.columns}
            
        # Count both true nulls and floating point NaNs
        null_counts = {}
        for col in df.columns:
            if df.schema[col].is_numeric():
                null_counts[col] = df.filter(pl.col(col).is_null() | pl.col(col).is_nan()).height
            else:
                null_counts[col] = df.filter(pl.col(col).is_null()).height
                
        return {col: float(count) / total_rows for col, count in null_counts.items()}
        
    @staticmethod
    def exact_duplicates_ratio(df: pl.DataFrame) -> float:
        """Calculates the ratio of duplicate rows in the dataset."""
        total_rows = df.height
        if total_rows == 0:
            return 0.0
            
        unique_rows = df.n_unique()
        return float(total_rows - unique_rows) / total_rows
        
    @staticmethod
    def outliers_iqr(df: pl.DataFrame) -> Dict[str, float]:
        """Calculates the ratio of outliers for numerical columns using the Interquartile Range (IQR) method."""
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        outlier_ratios = {}
        
        total_rows = df.height
        if total_rows == 0:
            return {col: 0.0 for col in numeric_cols}
            
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            if q1 is None or q3 is None:
                outlier_ratios[col] = 0.0
                continue
                
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_count = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).height
            outlier_ratios[col] = float(outliers_count) / total_rows
            
        return outlier_ratios
        
    @staticmethod
    def summary_statistics(df: pl.DataFrame) -> Dict[str, Any]:
        """Calculates basic summary statistics for numerical columns."""
        import polars.selectors as cs
        numeric_df = df.select(cs.numeric())
        
        stats = {}
        for col in numeric_df.columns:
            s_mean = numeric_df[col].mean()
            s_median = numeric_df[col].median()
            s_std = numeric_df[col].std()
            s_min = numeric_df[col].min()
            s_max = numeric_df[col].max()
            
            # Polars mode returns a series, take the first if exists
            s_modes = numeric_df[col].mode()
            s_mode = s_modes[0] if s_modes.len() > 0 else None
            
            stats[col] = {
                "mean": float(s_mean) if s_mean is not None else None,
                "median": float(s_median) if s_median is not None else None,
                "std": float(s_std) if s_std is not None else None,
                "min": float(s_min) if s_min is not None else None,
                "max": float(s_max) if s_max is not None else None,
                "mode": float(s_mode) if s_mode is not None else None,
            }
        return stats
        
    @staticmethod
    def evaluate_all(df: pl.DataFrame) -> Dict[str, Any]:
        """Runs all traditional data quality checks and returns a summary."""
        return {
            "summary_statistics": TraditionalMetrics.summary_statistics(df),
            "missing_values_ratio": TraditionalMetrics.missing_values(df),
            "exact_duplicates_ratio": TraditionalMetrics.exact_duplicates_ratio(df),
            "outliers_ratio": TraditionalMetrics.outliers_iqr(df)
        }
