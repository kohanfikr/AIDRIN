import h5py
import polars as pl
from aidrin.ingestors.base import BaseIngestor

class HDF5Ingestor(BaseIngestor):
    """Ingestor for HDF5 hierarchical files."""
    
    def load_data(self, source: str, **kwargs) -> pl.DataFrame:
        dataset_name = kwargs.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name must be provided for HDF5Ingestor")
            
        with h5py.File(source, 'r') as f:
            data = f[dataset_name][:]
            
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
