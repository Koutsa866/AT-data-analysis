# Athletic Training Analytics Workflow Guide

This guide provides step-by-step workflows for using the comprehensive athletic training analytics system across all three tiers of functionality.

## System Architecture Overview

### Tier 1: Traditional Athletic Training
- Basic injury and treatment tracking
- Recovery time prediction
- Provider workload analysis

### Tier 2: Time Series Analytics  
- Longitudinal athlete monitoring
- Weekly injury risk assessment
- Population-level trend analysis

### Tier 3: Personalized Monitoring
- Individual athlete profiles
- Heart rate zone training
- GPS movement analysis
- Personal ML models

## Quick Start Workflows

### 🚀 Basic Athletic Training Workflow

#### Step 1: Data Preparation
```bash
cd Scripts
python data_preparation.py
```
**What it does:**
- Cleans raw Excel files
- Standardizes column names and data types
- Handles missing values and date conversions
- Creates binary flags for analysis

**Expected Output:**
- `Treatment_Cleaned.xlsx`
- `Injury_Cleaned.xlsx`
- Console summary statistics

#### Step 2: Data Integration
```bash
python data_merger.py
```
**What it does:**
- Merges new data with historical records
- Removes duplicates automatically
- Sorts chronologically
- Updates master files

**Expected Output:**
- `Treatment_Master.xlsx` (updated)
- `Injury_Master.xlsx` (updated)
- Merge summary statistics

#### Step 3: Model Training
```bash
python real_ml_trainer_clean.py
```
**What it does:**
- Trains ML models on real injury data
- Validates model performance
- Saves trained models

**Expected Output:**
- `real_recovery_time_model.pkl` (0.75 day MAE)
- `real_injury_risk_model.pkl`
- Model performance metrics

#### Step 4: Interactive Analysis
```bash
streamlit run streamlit_dashboard.py
```
**What it provides:**
- Web-based interactive dashboard
- Time range filtering
- Provider and body part analysis
- D-Tale integration for advanced exploration

---

### 📊 Time Series Analytics Workflow

#### Step 1: Generate Time Series Data
```bash
python time_series_injury_predictor.py
```
**What it creates:**
- 5,200 athlete-week records (100 athletes × 52 weeks)
- Multi-factor injury prediction dataset
- Weekly risk scoring system
- Feature engineering for advanced models

**Expected Output:**
- `realistic_running_timeseries.csv`
- `weekly_risk_predictions.csv`
- `injury_prediction_rf.pkl` (96.5% accuracy)

#### Step 2: Analyze Time Series Patterns
```bash
# Launch dashboard with time series data
streamlit run streamlit_dashboard.py
```
**Analysis Capabilities:**
- Longitudinal trend analysis
- Seasonal pattern identification
- Population-level risk assessment
- Multi-week injury prediction

#### Step 3: Advanced Model Development
```python
# Example: LSTM model development
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load time series data
df = pd.read_csv("Data/TimeSeries/realistic_running_timeseries.csv")

# Prepare sequences for LSTM training
# (Framework ready for deep learning extension)
```

---

### 👤 Personalized Monitoring Workflow

#### Step 1: Create Athlete Profiles
```bash
python personalized_athlete_monitor.py
```
**What it generates:**
- Individual athlete profiles with baselines
- Heart rate zone training data
- GPS tracking with movement analysis
- Personal ML models per athlete

**Expected Output:**
- `athlete_profiles.csv` (10 athletes)
- `personalized_training_data.csv` (760 sessions)
- `athlete_baselines.csv` (personal thresholds)
- `athlete_XXX_model.pkl` (individual models)

#### Step 2: Monitor Individual Athletes
```python
# Example: Real-time athlete monitoring
import joblib
import pandas as pd

# Load athlete's personal model
athlete_model = joblib.load("Data/Personalized/athlete_001_model.pkl")

# Current session data
current_session = {
    'avg_hr': 155,
    'sleep_hours': 7.5,
    'total_load': 180,
    'fatigue_level': 0.4
}

# Predict injury risk
risk_score = athlete_model['model'].predict([list(current_session.values())])
print(f"Injury risk: {risk_score[0]:.2f}")
```

#### Step 3: Team Management Dashboard
```bash
# Launch comprehensive dashboard
streamlit run streamlit_dashboard.py
```
**Team Management Features:**
- Individual athlete status overview
- Risk stratification across team
- Personalized training recommendations
- Alert system for high-risk athletes

---

## Advanced Workflows

### 🔄 Continuous Monitoring Setup

#### Daily Workflow
1. **Data Collection**: Import new session data (HR, GPS, subjective measures)
2. **Risk Assessment**: Run personalized models for each athlete
3. **Alert Generation**: Identify athletes exceeding risk thresholds
4. **Intervention Planning**: Generate personalized recommendations

#### Weekly Workflow
1. **Model Updates**: Retrain personal models with new data
2. **Baseline Adjustment**: Update personal thresholds based on recent performance
3. **Trend Analysis**: Identify multi-week patterns and progressions
4. **Team Reporting**: Generate comprehensive team status reports

#### Monthly Workflow
1. **Performance Review**: Analyze model accuracy and athlete outcomes
2. **System Optimization**: Adjust algorithms based on real-world results
3. **Research Analysis**: Identify population-level trends and insights
4. **Documentation Updates**: Update athlete profiles and system parameters

### 🔬 Research and Validation Workflows

#### Model Validation Workflow
```bash
# Test model performance
python simple_model_test.py
```
**Validation Steps:**
1. Load trained models
2. Test with sample data
3. Compare predictions to actual outcomes
4. Generate performance reports

#### Research Data Export
```python
# Export data for research analysis
import pandas as pd

# Combine all datasets for comprehensive analysis
injury_data = pd.read_excel("Data/Master/Injury_Master.xlsx")
time_series_data = pd.read_csv("Data/TimeSeries/realistic_running_timeseries.csv")
personal_data = pd.read_csv("Data/Personalized/personalized_training_data.csv")

# Create research dataset
research_df = pd.concat([injury_data, time_series_data, personal_data], ignore_index=True)
research_df.to_csv("Data/Results/comprehensive_research_dataset.csv", index=False)
```

### 📱 Integration Workflows

#### Wearable Device Integration (Framework Ready)
```python
# Example: Garmin Connect API integration
class GarminIntegration:
    def fetch_athlete_data(self, athlete_id, date_range):
        # Fetch HR, GPS, sleep data from Garmin
        pass
    
    def update_athlete_profile(self, athlete_id, new_data):
        # Update personal baselines with real device data
        pass
```

#### Real-time Alert System
```python
# Example: Automated alert system
class AlertSystem:
    def check_athlete_risk(self, athlete_id):
        # Load personal model and current data
        # Calculate risk score
        # Send alerts if thresholds exceeded
        pass
    
    def generate_team_alerts(self):
        # Check all athletes
        # Generate coach notifications
        pass
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Data Loading Issues
```bash
# Check data file existence
ls -la Data/Master/
ls -la Data/TimeSeries/
ls -la Data/Personalized/

# Verify file formats
python -c "import pandas as pd; print(pd.read_excel('Data/Master/Injury_Master.xlsx').shape)"
```

#### Model Loading Issues
```bash
# Check model files
ls -la Data/Results/*.pkl

# Test model loading
python -c "import joblib; model = joblib.load('Data/Results/real_recovery_time_model.pkl'); print(type(model))"
```

#### Performance Optimization
```python
# For large datasets, use chunking
chunk_size = 1000
for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
    # Process chunk
    pass

# Use data caching for repeated analysis
@st.cache_data
def load_data():
    return pd.read_excel("Data/Master/Injury_Master.xlsx")
```

## Best Practices

### Data Management
1. **Regular Backups**: Backup master files before major updates
2. **Version Control**: Track changes to models and configurations
3. **Data Validation**: Verify data quality before analysis
4. **Documentation**: Maintain clear records of data sources and processing

### Model Management
1. **Regular Retraining**: Update models with new data weekly
2. **Performance Monitoring**: Track model accuracy over time
3. **A/B Testing**: Compare different model approaches
4. **Validation Studies**: Conduct prospective validation when possible

### System Maintenance
1. **Dependency Updates**: Keep Python packages current
2. **Security Reviews**: Regular security assessments for production use
3. **Performance Monitoring**: Track system response times
4. **User Training**: Ensure proper system usage by all stakeholders

## Getting Help

### Documentation Resources
- `README.md` - Complete system overview
- `DATA_DICTIONARY.md` - Comprehensive data documentation
- Script docstrings - Detailed function documentation

### Support Workflow
1. **Check Documentation**: Review relevant documentation first
2. **Test with Sample Data**: Verify issue with known good data
3. **Check Logs**: Review console output for error messages
4. **Isolate Issue**: Test individual components separately

### Research and Development
- **Academic Partnerships**: Collaborate with sports medicine researchers
- **Industry Connections**: Partner with wearable device manufacturers
- **Open Source Contributions**: Share improvements with community
- **Validation Studies**: Conduct prospective clinical validation