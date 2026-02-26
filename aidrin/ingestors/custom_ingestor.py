import importlib.util
from typing import Any
import polars as pl
from aidrin.ingestors.base import BaseIngestor

class CustomIngestor(BaseIngestor):
    """Ingestor that dynamically loads user-provided python scripts."""
    
    def load_data(self, source: str, **kwargs) -> pl.DataFrame:
        function_name = kwargs.get("function_name", "load_custom_data")
        
        spec = importlib.util.spec_from_file_location("custom_module", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {source}")
            
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        
        load_func = getattr(custom_module, function_name, None)
        if not load_func or not callable(load_func):
            raise ValueError(f"Function {function_name} not found or not callable in {source}")
            
        result = load_func(source, **kwargs)
        if not isinstance(result, pl.DataFrame):
            try:
                result = pl.DataFrame(result)
            except Exception as e:
                raise TypeError(f"Custom function returned type that cannot be converted to Polars DataFrame: {e}")
                
        return result
