# Athletic Training Data Analysis System

Comprehensive athletic training analytics system with injury recovery prediction model, StreamLit Interface, and Exploratory Analysis using D-Tale

## System Overview

###  Core Capabilities
- **Traditional Injury Tracking** - Clinical injury and treatment records
- **Time Series Monitoring** - Longitudinal athlete performance tracking
- **Personalized Analytics** - Individual ML models per athlete
- **Heart Rate Integration** - Zone-based training and recovery monitoring
- **GPS Tracking** - Movement analysis and load calculation
- **Predictive Prevention** - Injury prediction before occurrence

###  Data Sources
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

#### Data Integration
```bash
python Scripts/data_merger.py
```

#### Model Training
```bash
python Scripts/real_ml_trainer_clean.py
```

### Interactive Analysis

#### Streamlit Dashboard
```bash
streamlit run Scripts/streamlit_dashboard.py
```

#### D-Tale Data Explorer
```bash
python Scripts/dtale_explorer.py
```

## Model Performance Summary

### Traditional Models (Real Kaggle Data)
- **Recovery Time Prediction**: 0.75 day Mean Absolute Error
- **Injury Risk Classification**: Baseline accuracy on real data
- **Training Samples**: 194 real injury cases

### Time Series Models (Synthetic Realistic Data)
- **Injury Prediction**: 96.5% accuracy
- **Precision**: 98.6% (low false positives)
- **Recall**: 86.4% (catches most injuries)
- **F1-Score**: 92.1% (excellent balance)
- **Training Samples**: 5,200 athlete-week records

### Personalized Models (Individual Athletes)
- **Individual Models**: 10+ separate models per athlete
- **Custom Thresholds**: Personalized risk levels
- **Adaptive Learning**: Continuous model updates

## Privacy and Compliance
- **No Real Patient Data**: All identifiers anonymized or synthetic
- **Educational Appropriate**: Designed for learning and research
- **HIPAA Compliant**: No protected health information
- **Open Source Ready**: Framework suitable for sharing

## Contributing and Research Use

### Academic Research
- **Citable Framework**: Proper attribution for publications
- **Reproducible Results**: Consistent methodology across studies
- **Extension Friendly**: Easy to modify for specific research questions

### Industry Applications
- **Professional Sports**: Team monitoring and injury prevention
- **Fitness Technology**: Wearable device integration
- **Healthcare Systems**: Clinical decision support tools

### Educational Use
- **Data Science Learning**: Real-world ML application
- **Sports Medicine Training**: Evidence-based practice examples
- **Statistics Education**: Advanced analytics techniques
