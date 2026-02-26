import pytest
import polars as pl
import numpy as np
import os
import h5py
import zarr
import awkward as ak
import uproot
from aidrin.core import AIDRINProfiler
from aidrin.ingestors.hdf5_ingestor import HDF5Ingestor
from aidrin.ingestors.zarr_ingestor import ZarrIngestor
from aidrin.ingestors.root_ingestor import ROOTIngestor
from aidrin.metrics.traditional import TraditionalMetrics
from aidrin.metrics.ai_readiness import AIRreadinessMetrics
from aidrin.metrics.fair import FAIRCompliance

@pytest.fixture(scope="module")
def sample_dataset():
    """Generates an in-memory polars dataframe with specific issues."""
    np.random.seed(42)
    n = 100
    
    # Feature 1: Complete normal data
    f1 = np.random.normal(0, 1, n)
    
    # Feature 2: Missing values (completeness check)
    f2 = np.random.normal(5, 2, n)
    f2[np.random.choice(n, 10, replace=False)] = np.nan
    
    # Feature 3: Outliers
    f3 = np.random.normal(0, 1, n)
    f3[0] = 100.0  # Big outlier
    
    # Protected Attribute (for fairness)
    protected = np.random.choice([0, 1], n)
    
    # Target (imbalanced and biased)
    # Target is mostly 0, biased slightly by protected attribute
    target = (f1 + protected * 1.5 > 1).astype(int)
    
    df = pl.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "protected": protected,
        "target": target
    })
    
    # Add exact duplicates (duplicate check)
    duplicate_rows = df.head(5)
    df = pl.concat([df, duplicate_rows])
    return df

class TestTraditionalMetrics:
    def test_completeness_missing_values(self, sample_dataset):
        # The project description asks for 'completeness'
        stats = TraditionalMetrics.missing_values(sample_dataset)
        assert stats["f1"] == 0.0
        assert stats["f2"] > 0.0  # Should have missing values

    def test_outliers(self, sample_dataset):
        # The project description asks for 'outliers'
        stats = TraditionalMetrics.outliers_iqr(sample_dataset)
        assert stats["f3"] > 0.0  # Should detect the outlier

    def test_duplicates(self, sample_dataset):
        # The project description asks for 'duplicates'
        ratio = TraditionalMetrics.exact_duplicates_ratio(sample_dataset)
        assert ratio > 0.0  # Should detect the exactly duplicated 5 rows

class TestAIRreadinessMetrics:
    def test_feature_importance(self, sample_dataset):
        # The project description asks for 'feature importance'
        importance = AIRreadinessMetrics.feature_importance(sample_dataset, "target")
        assert "f1" in importance
        assert "protected" in importance
        
    def test_feature_correlations(self, sample_dataset):
        # The project description asks for 'feature correlations'
        corr = AIRreadinessMetrics.feature_correlations(sample_dataset)
        assert "f1" in corr
        assert "f2" in corr["f1"]
        
    def test_class_imbalance(self, sample_dataset):
        # The project description asks for 'class imbalance'
        imbalance = AIRreadinessMetrics.class_imbalance(sample_dataset, "target")
        assert "imbalance_degree_score" in imbalance
        assert "is_imbalanced" in imbalance
        
    def test_fairness(self, sample_dataset):
         # The project description asks for 'fairness'
         fairness = AIRreadinessMetrics.fairness_bias(sample_dataset, "target", "protected")
         assert "target_standard_deviation_tsd" in fairness

class TestMultipleFileFormats:
    # Project explicitly asks for: Zarr, ROOT, HDF5
    def test_hdf5_ingestion(self, tmp_path):
        f_path = os.path.join(tmp_path, "test.h5")
        data = np.zeros(10, dtype=[('x', 'i4'), ('y', 'f4')])
        data['x'] = np.arange(10)
        data['y'] = np.random.rand(10)
        
        with h5py.File(f_path, "w") as f:
            f.create_dataset("ds1", data=data)
            
        ingestor = HDF5Ingestor()
        df = ingestor.load_data(f_path, dataset_name="ds1")
        assert df.height == 10
        assert "x" in df.columns

    def test_zarr_ingestion(self, tmp_path):
        f_path = os.path.join(tmp_path, "test.zarr")
        z = zarr.open(f_path, mode='w', shape=(10, 2), chunks=(5, 2), dtype='i4')
        z[:] = 42
        
        ingestor = ZarrIngestor()
        df = ingestor.load_data(f_path, group_path="/")
        assert df.height == 10
        
    def test_root_ingestion(self, tmp_path):
        f_path = os.path.join(tmp_path, "test.root")
        
        # Test file creation:
        with uproot.recreate(f_path) as file:
            file["tree1"] = {"branch1": np.array([1, 2, 3]), "branch2": np.array([4.1, 5.2, 6.3])}
            
        ingestor = ROOTIngestor()
        df = ingestor.load_data(f_path, tree_name="tree1")
        assert df.height == 3
        assert "branch1" in df.columns

class TestFAIRCompliance:
    def test_fair_evaluation(self):
        # The project description asks for FAIR principle compliance checking
        dummy_metadata = {
             "dataset": {
                  "identifier": "123",
                  "title": "Test Title",
                  "distribution": {
                       "format": "csv",
                       "downloadURL": "http://example.com"
                  },
                  "license": "CC-BY",
                  "programCode": "000",
                  "bureauCode": "000",
                  "conformsTo": "standards",
                  "accessLevel": "public",
                  "publisher": { "name": "Test Pub" }
             }
        }
        fair_results = FAIRCompliance.evaluate(dummy_metadata)
        assert "overall_compliance_score" in fair_results
        assert fair_results["overall_compliance_score"] > 0
        
        # Verify specific category scoring
        assert fair_results["category_breakdown"]["findable"]["score"] > 0
