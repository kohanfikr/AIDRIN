from abc import ABC, abstractmethod
from typing import Any
import polars as pl

class BaseIngestor(ABC):
    """
    Abstract Base Class for AIDRIN data ingestors.
    All custom and format-specific ingestors must inherit from this class.
    """
    
    @abstractmethod
    def load_data(self, source: str, **kwargs: Any) -> pl.DataFrame:
        """
        Load data from the source and return a Polars DataFrame.
        """
        pass
