from typing import Dict, Any, Optional
from aidrin.ingestors.base import BaseIngestor
from aidrin.metrics.traditional import TraditionalMetrics
from aidrin.metrics.ai_readiness import AIRreadinessMetrics
from aidrin.metrics.fair import FAIRCompliance
from aidrin.intelligence.pii_detector import PIIDetector
from aidrin.intelligence.llm_profiler import LLMProfiler

class AIDRINProfiler:
    """Core orchestrator for the AIDRIN evaluation process."""
    
    def __init__(self, ingestor: BaseIngestor):
        self.ingestor = ingestor
        self.pii_detector = PIIDetector()
        self.llm_profiler = LLMProfiler()
        
    def profile(self, source: str, target_col: Optional[str] = None, protected_attribute: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Runs the entire readiness suite on a dataset."""
        print(f"Loading data from {source}...")
        df = self.ingestor.load_data(source, **kwargs)
        
        print("Running traditional data quality metrics...")
        traditional_results = TraditionalMetrics.evaluate_all(df)
        
        print("Running AI readiness metrics...")
        ai_results = AIRreadinessMetrics.evaluate_all(
            df, 
            target_col=target_col, 
            protected_attribute=protected_attribute,
            task_type=kwargs.get("task_type", "classification"),
            positive_label=kwargs.get("positive_label", 1)
        )
        
        print("Running privacy and PII checks...")
        pii_results = self.pii_detector.detect_pii(df)
        
        print("Running FAIR principle compliance check...")
        metadata = kwargs.get("metadata", {})
        fair_results = FAIRCompliance.evaluate(metadata) if metadata else {"message": "No metadata dictionary provided for FAIR analysis."}
        
        report = {
            "source": source,
            "row_count": df.height,
            "column_count": df.width,
            "columns": df.columns,
            "traditional_metrics": traditional_results,
            "ai_readiness_metrics": ai_results,
            "fair_compliance": fair_results,
            "privacy_metrics": {"pii_detected": pii_results}
        }
        
        if self.llm_profiler.client:
            print("Generating LLM insights...")
            insights = self.llm_profiler.generate_actionable_insights(traditional_results, ai_results)
            report["llm_insights"] = insights
        else:
            report["llm_insights"] = "Skipped - OPENAI_API_KEY not found."
            
        print("Profiling complete.")
        return report
