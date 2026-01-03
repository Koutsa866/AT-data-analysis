#!/usr/bin/env python3
"""
Personalized Athlete Monitoring System
Individual injury prediction with HR, GPS, and personalized baselines.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

class PersonalizedAthleteMonitor:
    """Personalized injury prediction for individual athletes."""
    
    def __init__(self):
        self.output_dir = Path("../Data/Personalized")
        self.output_dir.mkdir(exist_ok=True)
        self.athlete_models = {}
        self.athlete_baselines = {}
    
    def create_athlete_profiles(self, n_athletes=20):
        """Create detailed individual athlete profiles."""
        
        print("👤 Creating Personalized Athlete Profiles...")
        
        athletes = []
        
        for athlete_id in range(n_athletes):
            # Individual characteristics
            profile = {
                'athlete_id': athlete_id,
                'name': f'Athlete_{athlete_id:03d}',
                'age': np.random.randint(18, 35),
                'gender': np.random.choice(['M', 'F']),
                'sport': np.random.choice(['Running', 'Soccer', 'Basketball', 'Football']),
                'position': np.random.choice(['Forward', 'Midfielder', 'Defense', 'Goalkeeper']),
                'experience_years': np.random.randint(1, 15),
                
                # Physiological baselines
                'resting_hr': np.random.randint(45, 70),
                'max_hr': 220 - np.random.randint(18, 35) + np.random.randint(-10, 10),
                'vo2_max': np.random.uniform(35, 65),
                'body_weight': np.random.uniform(120, 220),  # lbs
                'height': np.random.uniform(60, 78),  # inches
                
                # Training characteristics
                'training_frequency': np.random.randint(3, 7),  # days per week
                'preferred_intensity': np.random.choice(['Low', 'Moderate', 'High']),
                'recovery_rate': np.random.uniform(0.5, 1.5),  # personal recovery factor
                
                # Injury history
                'previous_injuries': np.random.randint(0, 5),
                'injury_prone_areas': np.random.choice(['Knee', 'Ankle', 'Hamstring', 'None']),
                'last_injury_days': np.random.randint(30, 365) if np.random.random() < 0.6 else None,
                
                # Personal thresholds (learned over time)
                'fatigue_threshold': np.random.uniform(0.6, 0.9),
                'stress_tolerance': np.random.uniform(0.4, 0.8),
                'sleep_requirement': np.random.uniform(7, 9)
            }
            
            athletes.append(profile)
        
        athlete_df = pd.DataFrame(athletes)
        
        # Save profiles
        profile_file = self.output_dir / "athlete_profiles.csv"
        athlete_df.to_csv(profile_file, index=False)
        
        print(f"✅ Created {len(athletes)} athlete profiles")
        print(f"   Sports: {athlete_df['sport'].value_counts().to_dict()}")
        print(f"   Age range: {athlete_df['age'].min()}-{athlete_df['age'].max()}")
        
        return athlete_df
    
    def generate_personalized_training_data(self, athlete_profiles, weeks=26):
        """Generate personalized training data with HR and GPS."""
        
        print("📊 Generating Personalized Training Data...")
        
        training_data = []
        
        for _, athlete in athlete_profiles.iterrows():
            # Personal baselines
            base_hr_zones = {
                'zone1': athlete['resting_hr'] + (athlete['max_hr'] - athlete['resting_hr']) * 0.6,
                'zone2': athlete['resting_hr'] + (athlete['max_hr'] - athlete['resting_hr']) * 0.7,
                'zone3': athlete['resting_hr'] + (athlete['max_hr'] - athlete['resting_hr']) * 0.8,
                'zone4': athlete['resting_hr'] + (athlete['max_hr'] - athlete['resting_hr']) * 0.9,
                'zone5': athlete['max_hr']
            }
            
            # Initialize personal metrics
            fitness_level = np.random.uniform(0.4, 0.9)
            fatigue_level = 0.2
            injury_risk = 0.1
            
            for week in range(weeks):
                # Weekly training sessions
                sessions_per_week = athlete['training_frequency']
                
                for session in range(sessions_per_week):
                    date = datetime(2024, 1, 1) + timedelta(weeks=week, days=session)
                    
                    # Session characteristics
                    session_duration = np.random.uniform(30, 120)  # minutes
                    session_type = np.random.choice(['Easy', 'Tempo', 'Interval', 'Long', 'Recovery'])
                    
                    # Heart Rate Data (personalized zones)
                    if session_type == 'Easy':
                        avg_hr = np.random.uniform(base_hr_zones['zone1'], base_hr_zones['zone2'])
                        max_hr_session = avg_hr + np.random.uniform(10, 20)
                    elif session_type == 'Tempo':
                        avg_hr = np.random.uniform(base_hr_zones['zone2'], base_hr_zones['zone3'])
                        max_hr_session = avg_hr + np.random.uniform(15, 25)
                    elif session_type == 'Interval':
                        avg_hr = np.random.uniform(base_hr_zones['zone3'], base_hr_zones['zone4'])
                        max_hr_session = min(athlete['max_hr'], avg_hr + np.random.uniform(20, 30))
                    else:
                        avg_hr = np.random.uniform(base_hr_zones['zone1'], base_hr_zones['zone2'])
                        max_hr_session = avg_hr + np.random.uniform(5, 15)
                    
                    # GPS/Movement Data
                    if athlete['sport'] == 'Running':
                        distance = session_duration * np.random.uniform(0.08, 0.15)  # miles
                        avg_speed = distance / (session_duration / 60)  # mph
                        elevation_gain = distance * np.random.uniform(20, 100)  # feet
                    else:
                        # Field sports
                        distance = session_duration * np.random.uniform(0.05, 0.12)
                        avg_speed = distance / (session_duration / 60)
                        elevation_gain = np.random.uniform(0, 50)
                    
                    # Advanced GPS metrics
                    sprint_count = np.random.poisson(5) if session_type == 'Interval' else np.random.poisson(1)
                    max_speed = avg_speed * np.random.uniform(1.5, 2.5)
                    acceleration_load = sprint_count * np.random.uniform(0.5, 2.0)
                    
                    # Personalized load calculation
                    hr_load = (avg_hr / athlete['max_hr']) * session_duration
                    gps_load = (distance * avg_speed) / 10  # Normalized
                    total_load = hr_load + gps_load + acceleration_load
                    
                    # Personal response to load
                    personal_load_factor = 1.0 / athlete['recovery_rate']
                    adjusted_load = total_load * personal_load_factor
                    
                    # Recovery metrics
                    sleep_hours = np.random.uniform(
                        athlete['sleep_requirement'] - 1, 
                        athlete['sleep_requirement'] + 1
                    )
                    resting_hr_morning = athlete['resting_hr'] + fatigue_level * 10
                    hrv = np.random.uniform(30, 80) * (1 - fatigue_level * 0.5)  # Heart Rate Variability
                    
                    # Subjective measures
                    perceived_exertion = min(10, adjusted_load / 10 + np.random.uniform(-1, 1))
                    mood_score = np.random.uniform(1, 10) * (1 - fatigue_level * 0.3)
                    
                    # Update personal state
                    fatigue_level = min(1.0, fatigue_level + adjusted_load * 0.01 - sleep_hours * 0.02)
                    fitness_level = min(1.0, fitness_level + adjusted_load * 0.001)
                    
                    # Personalized injury risk
                    load_risk = max(0, adjusted_load - athlete['fatigue_threshold'] * 100) * 0.01
                    fatigue_risk = fatigue_level * 0.3
                    history_risk = athlete['previous_injuries'] * 0.05
                    
                    injury_risk = min(1.0, load_risk + fatigue_risk + history_risk)
                    
                    # Injury occurrence (personalized probability)
                    injured = np.random.random() < injury_risk * 0.1  # Scale down for realism
                    
                    # Record session data
                    session_record = {
                        'athlete_id': athlete['athlete_id'],
                        'date': date,
                        'week': week,
                        'session': session,
                        'session_type': session_type,
                        'duration_minutes': session_duration,
                        
                        # Heart Rate Data
                        'avg_hr': avg_hr,
                        'max_hr_session': max_hr_session,
                        'hr_zone_1_time': session_duration * np.random.uniform(0.2, 0.8),
                        'hr_zone_2_time': session_duration * np.random.uniform(0.1, 0.4),
                        'hr_zone_3_time': session_duration * np.random.uniform(0.0, 0.3),
                        'hr_zone_4_time': session_duration * np.random.uniform(0.0, 0.2),
                        'hr_zone_5_time': session_duration * np.random.uniform(0.0, 0.1),
                        
                        # GPS Data
                        'distance': distance,
                        'avg_speed': avg_speed,
                        'max_speed': max_speed,
                        'elevation_gain': elevation_gain,
                        'sprint_count': sprint_count,
                        'acceleration_load': acceleration_load,
                        
                        # Load Metrics
                        'hr_load': hr_load,
                        'gps_load': gps_load,
                        'total_load': total_load,
                        'adjusted_load': adjusted_load,
                        
                        # Recovery Metrics
                        'sleep_hours': sleep_hours,
                        'resting_hr_morning': resting_hr_morning,
                        'hrv': hrv,
                        'perceived_exertion': perceived_exertion,
                        'mood_score': mood_score,
                        
                        # Personal State
                        'fatigue_level': fatigue_level,
                        'fitness_level': fitness_level,
                        'injury_risk_score': injury_risk,
                        'injured': int(injured)
                    }
                    
                    training_data.append(session_record)
        
        training_df = pd.DataFrame(training_data)
        
        # Save training data
        training_file = self.output_dir / "personalized_training_data.csv"
        training_df.to_csv(training_file, index=False)
        
        print(f"✅ Generated personalized training data:")
        print(f"   Total sessions: {len(training_df)}")
        print(f"   Athletes: {training_df['athlete_id'].nunique()}")
        print(f"   Weeks: {training_df['week'].nunique()}")
        print(f"   Injury rate: {training_df['injured'].mean():.1%}")
        
        return training_df
    
    def create_personal_baselines(self, athlete_profiles, training_df):
        """Create personalized baselines for each athlete."""
        
        print("📈 Creating Personal Baselines...")
        
        baselines = {}
        
        for athlete_id in athlete_profiles['athlete_id']:
            athlete_data = training_df[training_df['athlete_id'] == athlete_id]
            
            if len(athlete_data) < 10:  # Need minimum data
                continue
            
            # Calculate personal baselines
            baseline = {
                'athlete_id': athlete_id,
                
                # HR baselines
                'baseline_resting_hr': athlete_data['resting_hr_morning'].median(),
                'baseline_avg_hr': athlete_data['avg_hr'].median(),
                'baseline_hrv': athlete_data['hrv'].median(),
                
                # Performance baselines
                'baseline_speed': athlete_data['avg_speed'].median(),
                'baseline_distance': athlete_data['distance'].median(),
                'baseline_load': athlete_data['total_load'].median(),
                
                # Recovery baselines
                'baseline_sleep': athlete_data['sleep_hours'].median(),
                'baseline_mood': athlete_data['mood_score'].median(),
                'baseline_rpe': athlete_data['perceived_exertion'].median(),
                
                # Variability (for anomaly detection)
                'hr_variability': athlete_data['avg_hr'].std(),
                'load_variability': athlete_data['total_load'].std(),
                'sleep_variability': athlete_data['sleep_hours'].std(),
                
                # Personal thresholds
                'high_load_threshold': athlete_data['total_load'].quantile(0.8),
                'fatigue_threshold': athlete_data['fatigue_level'].quantile(0.7),
                'injury_risk_threshold': athlete_data['injury_risk_score'].quantile(0.8)
            }
            
            baselines[athlete_id] = baseline
        
        # Save baselines
        baseline_df = pd.DataFrame(list(baselines.values()))
        baseline_file = self.output_dir / "athlete_baselines.csv"
        baseline_df.to_csv(baseline_file, index=False)
        
        self.athlete_baselines = baselines
        
        print(f"✅ Created baselines for {len(baselines)} athletes")
        
        return baselines
    
    def train_personal_models(self, athlete_profiles, training_df):
        """Train individual models for each athlete."""
        
        print("🤖 Training Personal Models...")
        
        feature_cols = [
            'avg_hr', 'max_hr_session', 'hrv', 'total_load', 'adjusted_load',
            'sleep_hours', 'perceived_exertion', 'mood_score', 'fatigue_level',
            'distance', 'avg_speed', 'sprint_count'
        ]
        
        personal_models = {}
        
        for athlete_id in athlete_profiles['athlete_id']:
            athlete_data = training_df[training_df['athlete_id'] == athlete_id]
            
            if len(athlete_data) < 20:  # Need minimum data for training
                continue
            
            # Prepare features
            X = athlete_data[feature_cols].fillna(0)
            y = athlete_data['injury_risk_score']
            
            if len(X) > 0 and y.std() > 0:  # Need variation in target
                # Train personal model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                # Store model and metadata
                personal_models[athlete_id] = {
                    'model': model,
                    'features': feature_cols,
                    'training_samples': len(X),
                    'feature_importance': dict(zip(feature_cols, model.feature_importances_))
                }
        
        self.athlete_models = personal_models
        
        # Save models
        for athlete_id, model_data in personal_models.items():
            model_file = self.output_dir / f"athlete_{athlete_id}_model.pkl"
            joblib.dump(model_data, model_file)
        
        print(f"✅ Trained personal models for {len(personal_models)} athletes")
        
        return personal_models

def main():
    """Create personalized athlete monitoring system."""
    
    monitor = PersonalizedAthleteMonitor()
    
    print("👤 Personalized Athlete Monitoring System")
    print("=" * 60)
    
    # Create athlete profiles
    athlete_profiles = monitor.create_athlete_profiles(n_athletes=10)
    
    # Generate personalized training data
    training_df = monitor.generate_personalized_training_data(athlete_profiles, weeks=20)
    
    # Create personal baselines
    baselines = monitor.create_personal_baselines(athlete_profiles, training_df)
    
    # Train personal models
    personal_models = monitor.train_personal_models(athlete_profiles, training_df)
    
    print(f"\n🎉 SUCCESS! Personalized System Ready!")
    print(f"\n📋 Capabilities:")
    print(f"✅ Individual athlete profiles with baselines")
    print(f"✅ Heart rate zone training data")
    print(f"✅ GPS tracking with speed/distance/elevation")
    print(f"✅ Personal injury risk models")
    print(f"✅ Customized thresholds per athlete")
    
    print(f"\n🚀 Personalization Features:")
    print(f"• Individual HR zones based on max/resting HR")
    print(f"• Personal load tolerance and recovery rates")
    print(f"• Customized injury risk thresholds")
    print(f"• Sport-specific movement patterns")
    print(f"• Individual sleep and recovery needs")

if __name__ == "__main__":
    main()