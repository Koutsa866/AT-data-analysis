#!/usr/bin/env python3
"""
Time Series Injury Prediction System
Build advanced LSTM models for injury prediction with time series data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

class TimeSeriesInjuryPredictor:
    """Advanced time series injury prediction system."""
    
    def __init__(self):
        self.output_dir = Path("../Data/TimeSeries")
        self.output_dir.mkdir(exist_ok=True)
        self.scaler = MinMaxScaler()
        self.models = {}
    
    def create_realistic_time_series_data(self, n_athletes=50, weeks=52):
        """Create realistic time series training data."""
        
        print("🏃♂️ Creating Realistic Time Series Training Data...")
        
        time_series_data = []
        
        for athlete_id in range(n_athletes):
            # Athlete characteristics
            age = np.random.randint(18, 35)
            gender = np.random.choice(['M', 'F'])
            fitness_level = np.random.uniform(0.3, 1.0)
            
            # Initialize weekly metrics
            base_mileage = np.random.uniform(20, 60)  # Weekly miles
            injury_risk = 0.0
            
            for week in range(weeks):
                # Training load progression with seasonality
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * week / 52)
                
                # Weekly training metrics
                weekly_mileage = base_mileage * seasonal_factor * np.random.uniform(0.8, 1.2)
                avg_pace = np.random.uniform(6.5, 9.0)  # Minutes per mile
                elevation_gain = weekly_mileage * np.random.uniform(50, 200)  # Feet
                
                # Training intensity
                easy_miles = weekly_mileage * np.random.uniform(0.6, 0.8)
                tempo_miles = weekly_mileage * np.random.uniform(0.1, 0.2)
                interval_miles = weekly_mileage * np.random.uniform(0.05, 0.15)
                
                # Recovery metrics
                sleep_hours = np.random.uniform(6.5, 9.0)
                resting_hr = np.random.uniform(45, 70)
                
                # Injury risk factors
                mileage_increase = (weekly_mileage - base_mileage) / base_mileage
                training_stress = (tempo_miles + interval_miles * 2) / weekly_mileage
                
                # Calculate injury probability
                injury_prob = (
                    0.1 * max(0, mileage_increase - 0.1) +  # Rapid mileage increase
                    0.15 * training_stress +                # High intensity
                    0.05 * (1 - fitness_level) +           # Low fitness
                    0.02 * max(0, age - 30) +              # Age factor
                    0.03 * max(0, 8 - sleep_hours) +       # Poor sleep
                    injury_risk * 0.3                      # Previous injury risk
                )
                
                # Injury occurrence
                injured = np.random.random() < injury_prob
                
                if injured:
                    injury_risk = min(1.0, injury_risk + 0.5)
                    injury_severity = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])  # 1=minor, 3=severe
                else:
                    injury_risk = max(0.0, injury_risk - 0.1)
                    injury_severity = 0
                
                # Record data
                record = {
                    'athlete_id': athlete_id,
                    'week': week,
                    'date': datetime(2024, 1, 1) + timedelta(weeks=week),
                    'age': age,
                    'gender': gender,
                    'weekly_mileage': weekly_mileage,
                    'avg_pace': avg_pace,
                    'elevation_gain': elevation_gain,
                    'easy_miles': easy_miles,
                    'tempo_miles': tempo_miles,
                    'interval_miles': interval_miles,
                    'sleep_hours': sleep_hours,
                    'resting_hr': resting_hr,
                    'mileage_increase': mileage_increase,
                    'training_stress': training_stress,
                    'injury_risk_score': injury_risk,
                    'injured': int(injured),
                    'injury_severity': injury_severity,
                    'fitness_level': fitness_level
                }
                
                time_series_data.append(record)
                
                # Update base mileage gradually
                base_mileage = base_mileage * 0.95 + weekly_mileage * 0.05
        
        df = pd.DataFrame(time_series_data)
        
        # Save dataset
        output_file = self.output_dir / "realistic_running_timeseries.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Created time series data: {len(df)} records")
        print(f"   Athletes: {n_athletes}")
        print(f"   Weeks per athlete: {weeks}")
        print(f"   Injury rate: {df['injured'].mean():.1%}")
        print(f"   Saved to: {output_file}")
        
        return df
    
    def prepare_sequences(self, df, sequence_length=4, target_col='injured'):
        """Prepare sequences for LSTM training."""
        
        print(f"🔧 Preparing sequences (length={sequence_length})...")
        
        # Features for prediction
        feature_cols = [
            'weekly_mileage', 'avg_pace', 'elevation_gain',
            'easy_miles', 'tempo_miles', 'interval_miles',
            'sleep_hours', 'resting_hr', 'mileage_increase',
            'training_stress', 'injury_risk_score'
        ]
        
        sequences = []
        targets = []
        
        # Group by athlete
        for athlete_id in df['athlete_id'].unique():
            athlete_data = df[df['athlete_id'] == athlete_id].sort_values('week')
            
            if len(athlete_data) < sequence_length + 1:
                continue
            
            # Create sequences
            for i in range(len(athlete_data) - sequence_length):
                # Input sequence (past weeks)
                seq = athlete_data[feature_cols].iloc[i:i+sequence_length].values
                # Target (next week injury)
                target = athlete_data[target_col].iloc[i+sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"✅ Created {len(X)} sequences")
        print(f"   Shape: {X.shape}")
        print(f"   Injury rate: {y.mean():.1%}")
        
        return X, y
    
    def train_simple_ml_model(self, df):
        """Train simple ML model for comparison."""
        
        print("🤖 Training Simple ML Model...")
        
        # Features
        feature_cols = [
            'weekly_mileage', 'avg_pace', 'training_stress',
            'mileage_increase', 'sleep_hours', 'injury_risk_score'
        ]
        
        X = df[feature_cols].fillna(0)
        y = df['injured']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Simple Random Forest for comparison
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"✅ Simple ML Model Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        # Feature importance
        importance = model.feature_importances_
        for feature, imp in zip(feature_cols, importance):
            print(f"   {feature}: {imp:.3f}")
        
        # Save model
        model_file = self.output_dir / "injury_prediction_rf.pkl"
        joblib.dump(model, model_file)
        
        self.models['random_forest'] = model
        return model
    
    def create_weekly_risk_predictor(self, df):
        """Create weekly injury risk prediction system."""
        
        print("📊 Creating Weekly Risk Prediction System...")
        
        # Calculate risk scores for each athlete-week
        risk_data = []
        
        for athlete_id in df['athlete_id'].unique():
            athlete_data = df[df['athlete_id'] == athlete_id].sort_values('week')
            
            for idx, row in athlete_data.iterrows():
                # Calculate risk factors
                mileage_risk = min(1.0, max(0, row['mileage_increase']) * 5)
                intensity_risk = min(1.0, row['training_stress'] * 2)
                recovery_risk = min(1.0, max(0, 8 - row['sleep_hours']) / 2)
                
                # Combined risk score
                total_risk = (mileage_risk + intensity_risk + recovery_risk) / 3
                
                # Risk category
                if total_risk < 0.3:
                    risk_category = 'Low'
                elif total_risk < 0.6:
                    risk_category = 'Medium'
                else:
                    risk_category = 'High'
                
                risk_record = {
                    'athlete_id': athlete_id,
                    'week': row['week'],
                    'date': row['date'],
                    'mileage_risk': mileage_risk,
                    'intensity_risk': intensity_risk,
                    'recovery_risk': recovery_risk,
                    'total_risk_score': total_risk,
                    'risk_category': risk_category,
                    'actual_injury': row['injured'],
                    'weekly_mileage': row['weekly_mileage'],
                    'training_stress': row['training_stress']
                }
                
                risk_data.append(risk_record)
        
        risk_df = pd.DataFrame(risk_data)
        
        # Save risk predictions
        risk_file = self.output_dir / "weekly_risk_predictions.csv"
        risk_df.to_csv(risk_file, index=False)
        
        # Evaluate risk prediction
        risk_accuracy = {}
        for category in ['Low', 'Medium', 'High']:
            category_data = risk_df[risk_df['risk_category'] == category]
            if len(category_data) > 0:
                injury_rate = category_data['actual_injury'].mean()
                risk_accuracy[category] = injury_rate
        
        print(f"✅ Weekly Risk System Created:")
        print(f"   Total predictions: {len(risk_df)}")
        for category, rate in risk_accuracy.items():
            print(f"   {category} risk injury rate: {rate:.1%}")
        
        return risk_df

def main():
    """Build time series injury prediction system."""
    
    predictor = TimeSeriesInjuryPredictor()
    
    print("🏃♂️ Time Series Injury Prediction System")
    print("=" * 60)
    
    # Create realistic time series data
    df = predictor.create_realistic_time_series_data(n_athletes=100, weeks=52)
    
    # Train simple ML model
    ml_model = predictor.train_simple_ml_model(df)
    
    # Create weekly risk system
    risk_df = predictor.create_weekly_risk_predictor(df)
    
    print(f"\n🎉 SUCCESS! Advanced injury prediction system ready!")
    print(f"\n📋 Capabilities:")
    print(f"✅ Time series data with 100 athletes × 52 weeks")
    print(f"✅ ML model for injury prediction")
    print(f"✅ Weekly risk scoring system")
    print(f"✅ Multi-factor risk assessment")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Integrate with Streamlit dashboard")
    print(f"2. Add LSTM models for sequence prediction")
    print(f"3. Create injury prevention recommendations")
    print(f"4. Build athlete monitoring system")

if __name__ == "__main__":
    main()