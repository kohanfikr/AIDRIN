import uproot
import polars as pl
import awkward as ak
from aidrin.ingestors.base import BaseIngestor

class ROOTIngestor(BaseIngestor):
    """Ingestor for High-Energy Physics ROOT files via uproot."""
    
    def load_data(self, source: str, **kwargs) -> pl.DataFrame:
        tree_name = kwargs.get("tree_name")
        if not tree_name:
             raise ValueError("tree_name must be provided for ROOTIngestor")
             
        with uproot.open(source) as file:
            tree = file[tree_name]
            arrays = tree.arrays(library="ak")
            arrow_table = ak.to_arrow_table(arrays)
            df = pl.from_arrow(arrow_table)
            
        if df is None:
             raise ValueError(f"Failed to load table from ROOT file tree {tree_name}")
             
        return df
