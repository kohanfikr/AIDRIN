import zarr
import polars as pl
from aidrin.ingestors.base import BaseIngestor

class ZarrIngestor(BaseIngestor):
    """Ingestor for chunked, compressed Zarr multidimensional arrays."""
    
    def load_data(self, source: str, **kwargs) -> pl.DataFrame:
        group_path = kwargs.get("group_path", "/")
        dataset_name = kwargs.get("dataset_name")
        
        root = zarr.open(source, mode='r')
        if dataset_name:
            group = root[group_path] if group_path != "/" else root
            data = group[dataset_name][:]
        else:
            data = root[:]
            
        if data.dtype.names:
            df = pl.DataFrame({name: data[name] for name in data.dtype.names})
        elif len(data.shape) == 2:
            num_cols = data.shape[1]
            df = pl.DataFrame({f"col_{i}": data[:, i] for i in range(num_cols)})
        elif len(data.shape) == 1:
            df = pl.DataFrame({"value": data})
        else:
            raise ValueError(f"Unsupported dataset shape: {data.shape}")
                
        return df
