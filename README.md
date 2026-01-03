# Athletic Training Data Analysis Scripts

This repository contains a comprehensive athletic training analytics system with traditional injury tracking, advanced time series monitoring, and personalized athlete management capabilities.

## System Overview

### 🏅 Core Capabilities
- **Traditional Injury Tracking** - Clinical injury and treatment records
- **Time Series Monitoring** - Longitudinal athlete performance tracking
- **Personalized Analytics** - Individual ML models per athlete
- **Heart Rate Integration** - Zone-based training and recovery monitoring
- **GPS Tracking** - Movement analysis and load calculation
- **Predictive Prevention** - Injury prediction before occurrence

### 📊 Data Sources
- **Real Kaggle Data**: 1,000+ injury prediction records
- **Time Series Data**: 100 athletes × 52 weeks (5,200+ records)
- **Personalized Profiles**: Individual baselines and thresholds
- **Wearable Integration**: Heart rate and GPS simulation

## Files Overview

### Core Pipeline Scripts
- **`data_preparation.py`** - Clean and standardize raw data
- **`data_merger.py`** - Merge new data into master historical files
- **`real_ml_trainer_clean.py`** - Train ML models on real injury data

### Advanced Analytics Scripts
- **`time_series_injury_predictor.py`** - Time series injury prediction with 100 athletes
- **`personalized_athlete_monitor.py`** - Individual athlete profiles and monitoring
- **`simple_bayesian_predictor.py`** - Bayesian injury prediction with uncertainty

### Visualization and Exploration
- **`streamlit_dashboard.py`** - Interactive web dashboard
- **`dtale_explorer.py`** - Advanced data exploration tool

### Reporting and Utilities
- **`email_reporter.py`** - Automated report generation
- **`simple_model_test.py`** - Quick model testing and validation

### Configuration Files
- **`requirements.txt`** - Python package dependencies

## Data Structure

### Traditional Data Files
- `Treatment_Cleaned.xlsx` - Processed treatment records
- `Injury_Cleaned.xlsx` - Processed injury records
- `Treatment_Master.xlsx` - Historical treatment data
- `Injury_Master.xlsx` - Historical injury data

### Advanced Analytics Data
- `realistic_running_timeseries.csv` - Time series training data (5,200 records)
- `weekly_risk_predictions.csv` - Weekly injury risk assessments
- `athlete_profiles.csv` - Individual athlete characteristics
- `personalized_training_data.csv` - Session-level monitoring data
- `athlete_baselines.csv` - Personal baseline metrics

### Model Files
- `real_recovery_time_model.pkl` - Recovery time prediction (0.75 day MAE)
- `real_injury_risk_model.pkl` - Injury risk classification
- `injury_prediction_rf.pkl` - Time series Random Forest model (96.5% accuracy)
- `athlete_XXX_model.pkl` - Individual athlete models

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. System Architecture
The system operates in three tiers:

#### Tier 1: Traditional Athletic Training
- Basic injury and treatment tracking
- Recovery time prediction
- Provider workload analysis

#### Tier 2: Time Series Analytics
- Longitudinal athlete monitoring
- Weekly injury risk assessment
- Population-level trend analysis

#### Tier 3: Personalized Monitoring
- Individual athlete profiles
- Heart rate zone training
- GPS movement analysis
- Personal ML models

## Usage

### Basic Athletic Training Pipeline

#### Data Preparation
```bash
python Scripts/data_preparation.py
```
Cleans and standardizes raw Excel files with:
- Date conversions and missing value handling
- Column name standardization
- Binary flag creation
- Summary statistics generation

#### Data Integration
```bash
python Scripts/data_merger.py
```
Merges new data with historical records:
- Automatic duplicate removal
- Chronological sorting
- Master file updates

#### Model Training
```bash
python Scripts/real_ml_trainer_clean.py
```
Trains ML models on real injury data:
- Recovery time prediction (0.75 day MAE)
- Injury risk classification
- Model validation and saving

### Advanced Time Series Analytics

#### Time Series Injury Prediction
```bash
python Scripts/time_series_injury_predictor.py
```
Creates comprehensive time series system:
- 100 athletes × 52 weeks of data (5,200 records)
- Multi-factor injury prediction (96.5% accuracy)
- Weekly risk scoring system
- Feature engineering for LSTM models

#### Personalized Athlete Monitoring
```bash
python Scripts/personalized_athlete_monitor.py
```
Builds individual athlete monitoring:
- Personal profiles with physiological baselines
- Heart rate zone training data
- GPS tracking with movement analysis
- Individual ML models per athlete
- Customized risk thresholds

### Interactive Analysis

#### Streamlit Dashboard
```bash
streamlit run Scripts/streamlit_dashboard.py
```
Launches comprehensive web dashboard:
- Multi-tier data visualization
- Time range filtering (1 week to full history)
- Provider and body part analysis
- Interactive D-Tale integration
- Real-time model predictions

#### D-Tale Data Explorer
```bash
python Scripts/dtale_explorer.py
```
Advanced data exploration:
- Automatic loading of all datasets
- Interactive filtering and visualization
- Statistical analysis tools
- Export capabilities

### Reporting and Validation

#### Automated Reports
```bash
python Scripts/email_reporter.py
```
Generates comprehensive reports:
- Multi-level analytics summary
- Model performance metrics
- Athlete risk assessments
- Trend analysis

#### Model Testing
```bash
python Scripts/simple_model_test.py
```
Quick model validation:
- Load and test trained models
- Sample predictions
- Performance verification

## Advanced Features

### Personalized Monitoring System

#### Individual Athlete Profiles
- **Physiological Baselines**: Resting HR, max HR, VO2 max, recovery rates
- **Training Characteristics**: Frequency, intensity preferences, experience
- **Injury History**: Previous injuries, prone areas, recovery patterns
- **Personal Thresholds**: Fatigue tolerance, stress limits, sleep requirements

#### Heart Rate Integration
- **Personalized HR Zones**: Zone 1-5 based on individual max/resting HR
- **Training Load Calculation**: HR-based load metrics with personal adjustment
- **Recovery Monitoring**: HRV tracking and resting HR trends
- **Zone Distribution**: Time spent in each training zone per session

#### GPS and Movement Analysis
- **Distance Tracking**: Precise movement measurement per session
- **Speed Analysis**: Average and maximum speed monitoring
- **Elevation Tracking**: Terrain difficulty assessment
- **Sprint Detection**: High-intensity movement identification
- **Acceleration Load**: Movement quality and intensity metrics

#### Personal ML Models
- **Individual Training**: Separate model per athlete (10+ models)
- **Custom Features**: Personalized feature importance
- **Adaptive Thresholds**: Dynamic risk level adjustment
- **Continuous Learning**: Model updates with new data

### Time Series Analytics

#### Longitudinal Tracking
- **Multi-week Patterns**: 52-week athlete monitoring
- **Trend Analysis**: Performance and risk trajectory identification
- **Seasonal Effects**: Training periodization impact
- **Load Progression**: Gradual training load development

#### Predictive Capabilities
- **Injury Prevention**: Predict injuries before occurrence (not just recovery)
- **Risk Stratification**: Low/Medium/High risk categories
- **Early Warning**: Multi-day risk trend alerts
- **Load Management**: Optimal training load recommendations

### Integration Capabilities

#### Wearable Device Support (Framework Ready)
- **Heart Rate Monitors**: Polar, Garmin, Apple Watch integration potential
- **GPS Devices**: Garmin, Suunto, Strava data import capability
- **Sleep Trackers**: Recovery monitoring integration
- **Smartphone Apps**: Manual data entry and tracking

#### Real-time Processing (Architecture Ready)
- **Streaming Data**: Live monitoring framework
- **Alert Systems**: Automated risk notifications
- **Dashboard Updates**: Real-time athlete status
- **Coach Notifications**: Team-wide monitoring alerts

## Model Performance Summary

### Traditional Models (Real Kaggle Data)
- **Recovery Time Prediction**: 0.75 day Mean Absolute Error
- **Injury Risk Classification**: Baseline accuracy on real data
- **Training Samples**: 194 real injury cases
- **Features**: Age-based prediction (simple but effective)

### Time Series Models (Synthetic Realistic Data)
- **Injury Prediction**: 96.5% accuracy
- **Precision**: 98.6% (low false positives)
- **Recall**: 86.4% (catches most injuries)
- **F1-Score**: 92.1% (excellent balance)
- **Training Samples**: 5,200 athlete-week records
- **Key Feature**: injury_risk_score (77.3% importance)

### Personalized Models (Individual Athletes)
- **Individual Models**: 10+ separate models per athlete
- **Custom Thresholds**: Personalized risk levels
- **Adaptive Learning**: Continuous model updates
- **Multi-factor Analysis**: 15+ features per prediction

## Data Quality and Sources

### Real Data Integration
- **Kaggle Dataset**: 1,000+ real injury prediction records
- **Public Domain**: Legally compliant for educational use
- **Validated Patterns**: Real-world injury relationships
- **Citable Source**: Proper attribution for research

### Synthetic Enhancement
- **Realistic Simulation**: Based on published sports medicine research
- **Time Series Depth**: 52 weeks × 100 athletes
- **Physiological Accuracy**: Heart rate zones, training loads
- **Movement Realism**: GPS tracking with sport-specific patterns

### Privacy and Compliance
- **No Real Patient Data**: All identifiers anonymized or synthetic
- **Educational Appropriate**: Designed for learning and research
- **HIPAA Compliant**: No protected health information
- **Open Source Ready**: Framework suitable for sharing

## Customization and Extension

### Adding New Data Sources
The system is designed for easy extension:

```python
# Add new wearable device integration
class GarminDataProcessor(PersonalizedAthleteMonitor):
    def import_garmin_data(self, device_id):
        # Custom import logic
        pass

# Extend with new sports
class SportSpecificAnalyzer:
    def analyze_soccer_patterns(self, gps_data):
        # Soccer-specific movement analysis
        pass
```

### Model Enhancement
```python
# Add LSTM for sequence prediction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMInjuryPredictor:
    def build_sequence_model(self, sequence_length=4):
        # Advanced time series modeling
        pass
```

### Dashboard Customization
```python
# Add new visualization tabs
def render_personalized_dashboard(self):
    # Individual athlete monitoring view
    pass

def render_team_comparison(self):
    # Multi-athlete comparison view
    pass
```

## Future Development Roadmap

### Phase 1: Enhanced Integration (Next Steps)
- **Streamlit Dashboard Update**: Integrate personalized monitoring
- **Real-time Alerts**: Automated risk threshold notifications
- **Mobile Interface**: Athlete self-reporting capabilities
- **Coach Dashboard**: Team-wide monitoring overview

### Phase 2: Advanced Analytics
- **LSTM Models**: Deep learning for sequence prediction
- **Ensemble Methods**: Combine multiple model predictions
- **Causal Inference**: Identify injury causation factors
- **Intervention Optimization**: Recommend specific prevention strategies

### Phase 3: Production Deployment
- **API Development**: RESTful services for data integration
- **Cloud Deployment**: Scalable cloud infrastructure
- **Real Device Integration**: Live wearable device connections
- **Clinical Validation**: Prospective studies with real athletes

### Phase 4: Research Extensions
- **Multi-sport Analysis**: Cross-sport injury pattern comparison
- **Population Studies**: Large-scale epidemiological analysis
- **Genetic Integration**: Incorporate genetic predisposition factors
- **Environmental Factors**: Weather, altitude, surface analysis

## Contributing and Research Use

### Academic Research
- **Citable Framework**: Proper attribution for publications
- **Reproducible Results**: Consistent methodology across studies
- **Extension Friendly**: Easy to modify for specific research questions
- **Validation Ready**: Framework for prospective validation studies

### Industry Applications
- **Professional Sports**: Team monitoring and injury prevention
- **Fitness Technology**: Wearable device integration
- **Healthcare Systems**: Clinical decision support tools
- **Insurance Analytics**: Risk assessment and premium calculation

### Educational Use
- **Data Science Learning**: Real-world ML application
- **Sports Medicine Training**: Evidence-based practice examples
- **Statistics Education**: Advanced analytics techniques
- **Software Development**: Production-ready code examples