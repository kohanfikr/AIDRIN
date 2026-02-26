import argparse
import sys
import json
import subprocess
from aidrin.core import AIDRINProfiler
from aidrin.ingestors.hdf5_ingestor import HDF5Ingestor
from aidrin.ingestors.zarr_ingestor import ZarrIngestor
from aidrin.ingestors.root_ingestor import ROOTIngestor
from aidrin.ingestors.custom_ingestor import CustomIngestor
from aidrin.report.html import HTMLReporter

def main():
    parser = argparse.ArgumentParser(description="AIDRIN: AI Data Readiness Inspector CLI")
    parser.add_argument("source", help="Path to the dataset")
    parser.add_argument("--format", choices=["hdf5", "zarr", "root", "custom"], help="Format of the dataset")
    parser.add_argument("--dataset-name", help="Name of the dataset (for HDF5/Zarr)")
    parser.add_argument("--tree-name", help="Tree name (for ROOT files)")
    parser.add_argument("--function-name", help="Function name for custom ingestor", default="load_custom_data")
    parser.add_argument("--target-col", help="Target column for AI metrics (Feature Importance, Imbalance)")
    parser.add_argument("--protected-attr", help="Protected attribute for Fairness/Bias metrics")
    parser.add_argument("--output-json", help="Path to save JSON report", default="aidrin_report.json")
    parser.add_argument("--output-html", help="Path to save HTML report", default="aidrin_report.html")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit dashboard after profiling")
    
    args = parser.parse_args()
    
    if args.format == "hdf5":
        ingestor = HDF5Ingestor()
    elif args.format == "zarr":
        ingestor = ZarrIngestor()
    elif args.format == "root":
        ingestor = ROOTIngestor()
    elif args.format == "custom":
        ingestor = CustomIngestor()
    else:
        # Default to Custom if format isn't specified but we can infer or raise
        print("Format not specified. Please provide --format")
        sys.exit(1)
        
    profiler = AIDRINProfiler(ingestor)
    
    kwargs = {}
    if args.dataset_name: kwargs["dataset_name"] = args.dataset_name
    if args.tree_name: kwargs["tree_name"] = args.tree_name
    if args.function_name: kwargs["function_name"] = args.function_name
    
    try:
        report = profiler.profile(args.source, target_col=args.target_col, protected_attribute=args.protected_attr, **kwargs)
    except Exception as e:
        print(f"Error during profiling: {e}")
        sys.exit(1)
        
    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=4)
    print(f"JSON Report saved to {args.output_json}")
    
    # Save HTML
    HTMLReporter.generate(report, args.output_html)
    
    # Launch Streamlit
    if args.streamlit:
        print("Launching Streamlit Dashboard...")
        subprocess.run(["streamlit", "run", "aidrin/report/streamlit_app.py", args.output_json])

if __name__ == "__main__":
    main()
