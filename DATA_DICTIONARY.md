# Athletic Training Data Dictionary

This document describes all column headers, their meanings, and variable types in the comprehensive athletic training and personalized monitoring datasets.

⚠️ **Note**: This data dictionary reflects the complete system including traditional injury data, time series monitoring, and personalized athlete tracking.

## Core Athletic Training Data

### Treatment Data Columns

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `Date` | DateTime | Date and time of treatment encounter | 2025-08-15 14:30:00 |
| `Patient DOB` | Text/Object | Patient date of birth (anonymized identifier) | Various dates |
| `Service` | Categorical | Type of treatment service provided | "Evaluation", "Treatment", "Rehab" |
| `Code` | Numeric (Float) | Treatment procedure code | 97110, 97112, etc. |
| `Body Part` | Categorical | Anatomical area treated | "Knee", "Ankle", "Shoulder" |
| `Affected Area` | Categorical | General body region | "Lower Extremity", "Upper Extremity" |
| `Side` | Categorical | Laterality of treatment | "Left", "Right", "Bilateral" |
| `Treating Provider` | Categorical | Healthcare provider name | Provider names |
| `Missed` | Categorical | Whether appointment was missed | "Yes", "No" |
| `Scheduled` | Categorical | Whether appointment was scheduled | "Yes", "No" |
| `Date_only` | Date | Date portion only (derived) | 2025-08-15 |
| `Missed_Flag` | Boolean | Binary flag for missed appointments | True/False |

### Injury Data Columns

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `Unique Id` | Text/Object | Unique identifier for injury record | Alphanumeric codes |
| `Graduation Year` | Numeric (Float) | Student's expected graduation year | 2026, 2027, 2028, 2029 |
| `Teams` | Numeric (Float) | Team identifier or code | Various numeric codes |
| `Problem Created Date Time` | DateTime | When injury record was created in system | 2025-08-15 09:00:00 |
| `Problem Reported By` | Categorical | Who reported the injury | "Student", "Coach", "Trainer" |
| `Problem Reported By Provider Identifier` | Numeric (Float) | Provider ID who reported injury | Numeric codes |
| `Problem Date` | DateTime | Actual date/time when injury occurred | 2025-08-14 16:45:00 |
| `Expected Return Date` | DateTime | Projected return to activity date | 2025-08-28 |
| `Actual Return Date` | DateTime | Actual return to activity date | 2025-08-30 |
| `Days Out` | Numeric (Integer) | Number of days out of activity (calculated) | 0, 5, 14, 30 |
| `Reported Date` | DateTime | Date injury was reported | 2025-08-15 |
| `Privacy Level` | Categorical | Data privacy classification | "Standard", "Restricted" |
| `Body Part` | Categorical | Anatomical area injured | "Knee", "Ankle", "Shoulder" |
| `Affected Area` | Categorical | General body region | "Lower Extremity", "Upper Extremity" |
| `Problem Side` | Categorical | Laterality of injury | "Left", "Right", "Bilateral" |
| `Condition` | Categorical | Type of injury or condition | "Sprain", "Strain", "Contusion" |
| `Problem Sites` | Categorical | Specific anatomical sites | Detailed anatomical locations |
| `Problem Status` | Categorical | Current status of injury | "Active", "Returned To Play", "Closed" |
| `Diagnosis Code` | Text/Object | Medical diagnosis code | ICD-10 or similar codes |
| `Datalys Injury Type` | Categorical | Standardized injury classification | Datalys taxonomy categories |

## Personalized Athlete Monitoring Data

### Athlete Profile Columns

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `athlete_id` | Numeric (Integer) | Unique athlete identifier | 0, 1, 2, 3... |
| `name` | Text/Object | Athlete identifier name | "Athlete_001", "Athlete_002" |
| `age` | Numeric (Integer) | Athlete age in years | 18, 19, 20... 35 |
| `gender` | Categorical | Athlete gender | "M", "F" |
| `sport` | Categorical | Primary sport | "Running", "Soccer", "Basketball", "Football" |
| `position` | Categorical | Playing position | "Forward", "Midfielder", "Defense", "Goalkeeper" |
| `experience_years` | Numeric (Integer) | Years of competitive experience | 1, 2, 3... 15 |
| `resting_hr` | Numeric (Integer) | Baseline resting heart rate (bpm) | 45, 50, 55... 70 |
| `max_hr` | Numeric (Integer) | Maximum heart rate (bpm) | 180, 190, 200... 210 |
| `vo2_max` | Numeric (Float) | VO2 max fitness measure | 35.5, 45.2, 55.8... 65.0 |
| `body_weight` | Numeric (Float) | Body weight in pounds | 120.5, 150.0, 180.2... 220.0 |
| `height` | Numeric (Float) | Height in inches | 60.0, 65.5, 70.0... 78.0 |
| `training_frequency` | Numeric (Integer) | Training days per week | 3, 4, 5, 6, 7 |
| `preferred_intensity` | Categorical | Preferred training intensity | "Low", "Moderate", "High" |
| `recovery_rate` | Numeric (Float) | Personal recovery factor | 0.5, 0.8, 1.0, 1.2, 1.5 |
| `previous_injuries` | Numeric (Integer) | Number of previous injuries | 0, 1, 2, 3, 4, 5 |
| `injury_prone_areas` | Categorical | Areas prone to injury | "Knee", "Ankle", "Hamstring", "None" |
| `last_injury_days` | Numeric (Integer) | Days since last injury | 30, 60, 90... 365 |
| `fatigue_threshold` | Numeric (Float) | Personal fatigue tolerance | 0.6, 0.7, 0.8, 0.9 |
| `stress_tolerance` | Numeric (Float) | Stress handling capacity | 0.4, 0.5, 0.6, 0.7, 0.8 |
| `sleep_requirement` | Numeric (Float) | Required sleep hours | 7.0, 7.5, 8.0, 8.5, 9.0 |

### Training Session Data Columns

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `athlete_id` | Numeric (Integer) | Athlete identifier | 0, 1, 2, 3... |
| `date` | DateTime | Session date and time | 2024-01-15 08:00:00 |
| `week` | Numeric (Integer) | Training week number | 0, 1, 2... 52 |
| `session` | Numeric (Integer) | Session number within week | 0, 1, 2... 6 |
| `session_type` | Categorical | Type of training session | "Easy", "Tempo", "Interval", "Long", "Recovery" |
| `duration_minutes` | Numeric (Float) | Session duration in minutes | 30.0, 45.5, 60.0... 120.0 |

### Heart Rate Data Columns

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `avg_hr` | Numeric (Float) | Average heart rate during session (bpm) | 120.5, 145.2, 165.8 |
| `max_hr_session` | Numeric (Float) | Maximum heart rate during session (bpm) | 150.0, 175.5, 190.2 |
| `hr_zone_1_time` | Numeric (Float) | Time in HR Zone 1 (minutes) | 15.0, 20.5, 30.0 |
| `hr_zone_2_time` | Numeric (Float) | Time in HR Zone 2 (minutes) | 10.0, 15.5, 25.0 |
| `hr_zone_3_time` | Numeric (Float) | Time in HR Zone 3 (minutes) | 5.0, 10.0, 15.0 |
| `hr_zone_4_time` | Numeric (Float) | Time in HR Zone 4 (minutes) | 2.0, 5.0, 8.0 |
| `hr_zone_5_time` | Numeric (Float) | Time in HR Zone 5 (minutes) | 0.0, 1.0, 3.0 |
| `resting_hr_morning` | Numeric (Float) | Morning resting heart rate (bpm) | 45.0, 50.5, 55.0 |
| `hrv` | Numeric (Float) | Heart Rate Variability | 30.0, 45.5, 60.0... 80.0 |

### GPS and Movement Data Columns

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `distance` | Numeric (Float) | Total distance covered (miles) | 2.5, 5.0, 8.2, 12.1 |
| `avg_speed` | Numeric (Float) | Average speed (mph) | 6.5, 8.0, 10.2, 12.5 |
| `max_speed` | Numeric (Float) | Maximum speed reached (mph) | 12.0, 15.5, 18.2, 22.0 |
| `elevation_gain` | Numeric (Float) | Total elevation gained (feet) | 50.0, 150.5, 300.0, 500.0 |
| `sprint_count` | Numeric (Integer) | Number of sprints performed | 0, 2, 5, 8, 12 |
| `acceleration_load` | Numeric (Float) | Acceleration/deceleration load | 0.5, 1.2, 2.0, 3.5 |

### Load and Recovery Metrics

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `hr_load` | Numeric (Float) | Heart rate-based training load | 50.0, 100.5, 150.0, 200.0 |
| `gps_load` | Numeric (Float) | GPS-based movement load | 10.0, 25.5, 40.0, 60.0 |
| `total_load` | Numeric (Float) | Combined training load | 75.0, 150.0, 225.0, 300.0 |
| `adjusted_load` | Numeric (Float) | Personalized adjusted load | 80.0, 160.0, 240.0, 320.0 |
| `sleep_hours` | Numeric (Float) | Hours of sleep | 6.5, 7.0, 7.5, 8.0, 8.5 |
| `perceived_exertion` | Numeric (Float) | RPE scale (1-10) | 3.0, 5.5, 7.0, 8.5, 10.0 |
| `mood_score` | Numeric (Float) | Mood rating (1-10) | 4.0, 6.5, 8.0, 9.5 |
| `fatigue_level` | Numeric (Float) | Fatigue level (0-1) | 0.2, 0.4, 0.6, 0.8 |
| `fitness_level` | Numeric (Float) | Current fitness level (0-1) | 0.4, 0.6, 0.8, 0.9 |
| `injury_risk_score` | Numeric (Float) | Calculated injury risk (0-1) | 0.1, 0.3, 0.5, 0.7, 0.9 |
| `injured` | Boolean | Injury occurrence flag | 0, 1 |

### Personal Baseline Data Columns

| Column Name | Variable Type | Description | Example Values |
|-------------|---------------|-------------|----------------|
| `baseline_resting_hr` | Numeric (Float) | Personal resting HR baseline | 48.5, 52.0, 58.2 |
| `baseline_avg_hr` | Numeric (Float) | Personal average HR baseline | 140.0, 155.5, 170.0 |
| `baseline_hrv` | Numeric (Float) | Personal HRV baseline | 45.0, 55.5, 65.0 |
| `baseline_speed` | Numeric (Float) | Personal speed baseline | 7.5, 9.0, 11.2 |
| `baseline_distance` | Numeric (Float) | Personal distance baseline | 4.0, 6.5, 8.0 |
| `baseline_load` | Numeric (Float) | Personal load baseline | 120.0, 180.0, 240.0 |
| `baseline_sleep` | Numeric (Float) | Personal sleep baseline | 7.5, 8.0, 8.5 |
| `baseline_mood` | Numeric (Float) | Personal mood baseline | 6.5, 7.5, 8.5 |
| `baseline_rpe` | Numeric (Float) | Personal RPE baseline | 4.0, 5.5, 7.0 |
| `hr_variability` | Numeric (Float) | HR standard deviation | 8.0, 12.5, 18.0 |
| `load_variability` | Numeric (Float) | Load standard deviation | 25.0, 45.0, 65.0 |
| `sleep_variability` | Numeric (Float) | Sleep standard deviation | 0.5, 0.8, 1.2 |
| `high_load_threshold` | Numeric (Float) | 80th percentile load threshold | 200.0, 280.0, 360.0 |
| `fatigue_threshold` | Numeric (Float) | 70th percentile fatigue threshold | 0.5, 0.6, 0.7 |
| `injury_risk_threshold` | Numeric (Float) | 80th percentile risk threshold | 0.4, 0.6, 0.8 |

## Current Project Status

✅ **Complete System**: Traditional injury tracking + Advanced personalized monitoring

### System Components:
1. **Core Athletic Training Data** - Traditional injury and treatment records
2. **Time Series Monitoring** - Longitudinal athlete tracking (5,200+ records)
3. **Personalized Models** - Individual ML models per athlete
4. **Heart Rate Integration** - Zone-based training and recovery monitoring
5. **GPS Tracking** - Movement analysis and load calculation
6. **Predictive Analytics** - Injury prevention (not just recovery)

### Data Sources:
- **Real Kaggle Data**: 1,000+ injury prediction records
- **Time Series Data**: 100 athletes × 52 weeks of monitoring
- **Personalized Data**: Individual profiles and baselines
- **Wearable Integration**: Heart rate and GPS tracking simulation

### Model Capabilities:
- **Recovery Prediction**: 0.75 day MAE on real data
- **Injury Risk Prediction**: 96.5% accuracy with personalized models
- **Weekly Risk Assessment**: Individual athlete monitoring
- **Prevention Focus**: Predict injuries before they occur

## Advanced Analytics Capabilities

### Personalized Monitoring
- **Individual Baselines**: Each athlete has personalized thresholds
- **Adaptive Models**: ML models trained per athlete
- **Real-time Risk**: Continuous injury risk assessment
- **Custom Alerts**: Personalized warning systems

### Heart Rate Analytics
- **Zone Training**: Personalized HR zones (Zone 1-5)
- **Load Calculation**: HR-based training load metrics
- **Recovery Monitoring**: HRV and resting HR tracking
- **Fatigue Detection**: Early warning systems

### GPS and Movement Analysis
- **Distance Tracking**: Precise movement measurement
- **Speed Analysis**: Average and maximum speed monitoring
- **Elevation Tracking**: Terrain difficulty assessment
- **Sprint Analysis**: High-intensity movement detection
- **Load Integration**: GPS + HR combined load metrics

### Predictive Models
- **Time Series Forecasting**: LSTM-ready sequential data
- **Risk Stratification**: Low/Medium/High risk categories
- **Prevention Focus**: Predict injuries before occurrence
- **Multi-factor Analysis**: 15+ variables per prediction

## Integration Capabilities

### Wearable Device Support
- **Heart Rate Monitors**: Polar, Garmin, Apple Watch ready
- **GPS Devices**: Garmin, Suunto, Strava integration potential
- **Sleep Trackers**: Recovery monitoring integration
- **Smartphone Apps**: Manual data entry and tracking

### Real-time Processing
- **Streaming Data**: Live monitoring capabilities
- **Alert Systems**: Automated risk notifications
- **Dashboard Integration**: Real-time athlete status
- **Coach Notifications**: Team-wide monitoring alerts

## Key Relationships and Analysis

### Multi-level Data Integration
- **Athlete Profiles** ↔ **Training Sessions**: Individual characteristics linked to performance
- **Heart Rate Data** ↔ **GPS Data**: Physiological and movement correlation
- **Training Load** ↔ **Recovery Metrics**: Load-recovery balance analysis
- **Personal Baselines** ↔ **Current Metrics**: Deviation detection and alerts

### Advanced Derived Metrics
- **Personalized Load**: Individual training load adjusted for recovery capacity
- **Risk Trajectory**: Time-based injury risk progression
- **Performance Efficiency**: Speed/HR ratio and movement economy
- **Recovery Quality**: Sleep, HRV, and subjective measure integration
- **Training Stress Balance**: Acute vs chronic load ratios

### Predictive Analytics
- **Individual Models**: Separate ML models per athlete (10+ models)
- **Feature Importance**: Personalized risk factor identification
- **Threshold Adaptation**: Dynamic adjustment of warning levels
- **Trend Analysis**: Multi-week pattern recognition

## Usage Recommendations

### For Personalized Monitoring
1. **Establish Baselines**: Minimum 4-6 weeks of data per athlete
2. **Regular Updates**: Weekly model retraining with new data
3. **Threshold Calibration**: Adjust alerts based on individual responses
4. **Multi-factor Analysis**: Consider HR + GPS + subjective measures

### For Team Management
1. **Population Analysis**: Compare individual athletes to team norms
2. **Risk Stratification**: Identify high-risk athletes proactively
3. **Load Distribution**: Balance training across team members
4. **Recovery Optimization**: Personalized recovery recommendations

### For Injury Prevention
1. **Early Warning**: Monitor risk score trends, not just absolute values
2. **Multi-day Patterns**: Look for sustained risk elevation
3. **Individual Thresholds**: Use personal baselines, not population averages
4. **Intervention Triggers**: Automated alerts for coach/trainer action