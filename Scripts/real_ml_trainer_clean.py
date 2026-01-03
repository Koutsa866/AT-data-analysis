#!/usr/bin/env python3
"""
Real Data ML Trainer - Clean Version
Trains ML models on real Kaggle injury dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

class RealDataMLTrainer:
    """Train ML models on real injury data."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def load_real_data(self):
        """Load the real Kaggle injury data."""
        
        injury_file = Path("../Data/Master/Injury_Master.xlsx")
        
        if not injury_file.exists():
            print("❌ No injury data found. Run data pipeline first.")
            return None
        
        injury_df = pd.read_excel(injury_file)
        print(f"✅ Loaded {len(injury_df)} injury records")
        print(f"Columns: {list(injury_df.columns)}")
        
        return injury_df
    
    def train_recovery_model(self, injury_df):
        """Train recovery time prediction model."""
        
        df = injury_df.copy()
        
        # Clean Days Out column
        if 'Days Out' not in df.columns:
            print("❌ No 'Days Out' column found")
            return None
        
        df['Days_Out_Clean'] = pd.to_numeric(df['Days Out'], errors='coerce')
        df = df[df['Days_Out_Clean'].notna()].copy()
        df['Days_Out_Clean'] = df['Days_Out_Clean'].clip(0, 180)
        
        if len(df) < 10:
            print("❌ Insufficient data for training")
            return None
        
        # Prepare features
        features = []
        
        # Age
        if 'Age' in df.columns:
            df['Age_Clean'] = pd.to_numeric(df['Age'], errors='coerce').fillna(20)
            features.append('Age_Clean')
        
        # Body Part
        if 'Body Part' in df.columns:
            le = LabelEncoder()
            df['Body_Part_Encoded'] = le.fit_transform(df['Body Part'].fillna('Unknown'))
            features.append('Body_Part_Encoded')
        
        # Sport
        if 'Sport' in df.columns:
            le = LabelEncoder()
            df['Sport_Encoded'] = le.fit_transform(df['Sport'].fillna('Unknown'))
            features.append('Sport_Encoded')
        
        if len(features) == 0:
            print("❌ No usable features found")
            return None
        
        # Train model
        X = df[features].fillna(0)
        y = df['Days_Out_Clean']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['recovery_time'] = model
        self.results['recovery_time'] = {
            'mae': mae,
            'r2': r2,
            'n_samples': len(X),
            'features': features
        }
        
        print(f"✅ Recovery Model: MAE={mae:.2f} days, R²={r2:.3f}, N={len(X)}")
        return model
    
    def train_risk_model(self, injury_df):
        """Train injury risk classification model."""
        
        df = injury_df.copy()
        
        if 'Days Out' not in df.columns:
            return None
        
        df['Days_Out_Clean'] = pd.to_numeric(df['Days Out'], errors='coerce')
        df = df[df['Days_Out_Clean'].notna()].copy()
        
        # Create risk categories
        df['Risk'] = pd.cut(df['Days_Out_Clean'], 
                           bins=[0, 7, 30, 180], 
                           labels=['Low', 'Medium', 'High'])
        df = df[df['Risk'].notna()].copy()
        
        if len(df) < 10:
            return None
        
        # Features
        features = []
        
        if 'Age' in df.columns:
            df['Age_Clean'] = pd.to_numeric(df['Age'], errors='coerce').fillna(20)
            features.append('Age_Clean')
        
        if 'Body Part' in df.columns:
            le = LabelEncoder()
            df['Body_Part_Encoded'] = le.fit_transform(df['Body Part'].fillna('Unknown'))
            features.append('Body_Part_Encoded')
        
        if len(features) == 0:
            return None
        
        # Train
        X = df[features].fillna(0)
        y = LabelEncoder().fit_transform(df['Risk'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        self.models['injury_risk'] = model
        self.results['injury_risk'] = {
            'accuracy': acc,
            'n_samples': len(X),
            'features': features
        }
        
        print(f"✅ Risk Model: Accuracy={acc:.3f}, N={len(X)}")
        return model
    
    def save_models(self):
        """Save trained models."""
        
        output_dir = Path("../Data/Results")
        output_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_file = output_dir / f"real_{name}_model.pkl"
            joblib.dump(model, model_file)
            print(f"💾 Saved: {model_file}")
        
        return len(self.models)

def main():
    """Train ML models on real data."""
    
    trainer = RealDataMLTrainer()
    
    print("🎯 Real Data ML Trainer")
    print("=" * 40)
    
    # Load data
    injury_df = trainer.load_real_data()
    
    if injury_df is None:
        return
    
    # Train models
    print("\n🤖 Training Models...")
    
    recovery_model = trainer.train_recovery_model(injury_df)
    risk_model = trainer.train_risk_model(injury_df)
    
    # Save models
    if trainer.models:
        print("\n💾 Saving Models...")
        saved_count = trainer.save_models()
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Trained {len(trainer.models)} models on REAL data")
        print(f"💾 Saved {saved_count} model files")
        print(f"🚀 Models ready for prediction!")
    else:
        print("\n❌ No models trained. Check data quality.")

if __name__ == "__main__":
    main()