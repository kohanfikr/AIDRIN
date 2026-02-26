import polars as pl
from typing import Dict, Any, List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIDetector:
    """Uses Microsoft Presidio to detect PII in string columns."""
    
    def __init__(self):
        # We handle initialization gracefully in case models aren't downloaded
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.is_ready = True
        except Exception as e:
            print(f"PIIDetector failed to initialize (run 'python -m spacy download en_core_web_lg'): {e}")
            self.is_ready = False
            
    def detect_pii(self, df: pl.DataFrame, sample_size: int = 100) -> Dict[str, List[str]]:
        """Scans string columns for PII entities on a sample of data."""
        if not self.is_ready:
            return {"error": ["Presidio not initialized properly."]}
            
        string_cols = df.select(pl.col(pl.String)).columns
        if not string_cols:
             return {}
             
        sample_df = df.head(sample_size)
        pii_report = {}
        
        for col in string_cols:
            entities_found = set()
            for text in sample_df[col].drop_nulls().to_list():
                if not isinstance(text, str):
                    continue
                results = self.analyzer.analyze(text=text, language='en')
                for result in results:
                    entities_found.add(result.entity_type)
            
            if entities_found:
                pii_report[col] = list(entities_found)
                
        return pii_report
