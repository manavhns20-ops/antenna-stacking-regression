"""
Master Execution Script: EC-101 Stacking Regression Project
Antenna Performance Prediction System
Student: Manav Hans | Roll Number: 26BCS063 | Institute: NIT Hamirpur
Course: EC-101 | Category: Multi-Classical Regression
"""

import os
import sys
import argparse

# Ensure local imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import run_training_pipeline
from src.predict import AntennaPredictor


def main():
    parser = argparse.ArgumentParser(
        description="EC-101 Stacking Regression for Antenna Design — Master Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "train", "predict"],
        default="all",
        help="Pipeline execution mode: 'all' (train, evaluate, plot, predict), 'train', or 'predict'."
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      NIT HAMIRPUR — DEPARTMENT OF ELECTRONICS & COMMUNICATION ENGINEERING")
    print("                     EC-101 B.TECH SEMESTER 1 PROJECT")
    print("=" * 80)
    print("Project Title    : Stacking Regression for Antenna Design")
    print("Student Name     : Manav Hans")
    print("Roll Number      : 26BCS063")
    print("Official Course  : EC-101")
    print("Allotment Group  : Multi-Classical Regression (Item #63)")
    print("Assigned Method  : Stacking Regression with Meta-Model Study")
    print("Target Task      : Forward Regression [DV1...DV10] -> [P1_F1...P7_F5] (10 -> 35)")
    print("=" * 80 + "\n")

    if args.mode in ["all", "train"]:
        print(">>> Starting Training, Meta-Model Study, Evaluation and Plot Generation...\n")
        final_model, preprocessor, summary_df, meta_df = run_training_pipeline(
            data_path=os.path.join(PROJECT_ROOT, "data/dataset.csv"),
            models_dir=os.path.join(PROJECT_ROOT, "models/saved_model"),
            metrics_dir=os.path.join(PROJECT_ROOT, "results/metrics"),
            plots_dir=os.path.join(PROJECT_ROOT, "results/plots"),
            random_state=42
        )

    if args.mode in ["all", "predict"]:
        print("\n>>> Verifying Final Model Inference Engine...")
        predictor = AntennaPredictor(
            model_path=os.path.join(PROJECT_ROOT, "models/saved_model/final_stacking_model.joblib"),
            preprocessor_path=os.path.join(PROJECT_ROOT, "models/saved_model/preprocessor.joblib")
        )

        sample_inputs = {
            "DV1": 58.3862, "DV2": 0.58145, "DV3": 60.0, "DV4": 2.5636, "DV5": 17.4506,
            "DV6": 65.3378, "DV7": 58.2795, "DV8": 36.6794, "DV9": 44.5441, "DV10": 3807.854
        }
        print("Sample Antenna Design Variables:")
        for k, v in sample_inputs.items():
            print(f"  {k}: {v}")

        formatted_preds = predictor.predict_formatted_by_frequency(sample_inputs)
        print("\nPredicted 35 Performance Outputs Matrix (F1-F5 x P1-P7):")
        print(formatted_preds.to_string(index=False))

    print("\n" + "=" * 80)
    print("  PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("  Artifacts available in:")
    print("    - Models:  models/saved_model/")
    print("    - Metrics: results/metrics/")
    print("    - Plots:   results/plots/")
    print("    - Web App: app.py (Launch via: streamlit run app.py)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
