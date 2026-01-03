#!/usr/bin/env python3
"""
Simple Model Test - Test models with correct input format
"""

import joblib
import numpy as np
from pathlib import Path

def test_models():
    """Test models with correct input format."""
    
    results_dir = Path("../Data/Results")
    
    # Load recovery model
    recovery_file = results_dir / "real_recovery_time_model.pkl"
    if recovery_file.exists():
        recovery_model = joblib.load(recovery_file)
        print(f"✅ Recovery Model: {recovery_model.n_features_in_} features")
        
        # Test with ages (single feature)
        test_ages = np.array([[18], [20], [22], [24]])
        predictions = recovery_model.predict(test_ages)
        
        print("Recovery Predictions:")
        for age, pred in zip([18, 20, 22, 24], predictions):
            print(f"  Age {age}: {pred:.1f} days recovery")
    
    # Load risk model
    risk_file = results_dir / "real_injury_risk_model.pkl"
    if risk_file.exists():
        risk_model = joblib.load(risk_file)
        print(f"\n✅ Risk Model: {risk_model.n_features_in_} features")
        
        # Test with ages
        test_ages = np.array([[18], [20], [22], [24]])
        predictions = risk_model.predict(test_ages)
        
        risk_labels = ["Low", "Medium", "High"]
        print("Risk Predictions:")
        for age, pred in zip([18, 20, 22, 24], predictions):
            risk_level = risk_labels[pred] if pred < len(risk_labels) else f"Level {pred}"
            print(f"  Age {age}: {risk_level} risk")

if __name__ == "__main__":
    test_models()