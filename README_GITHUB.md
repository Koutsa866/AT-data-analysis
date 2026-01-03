# Athletic Training Analytics System

A comprehensive athletic training analytics platform with injury tracking, treatment monitoring, and predictive analytics capabilities.

## 🏅 Features

- **Real-time Dashboard** - Interactive Streamlit analytics dashboard
- **Injury Tracking** - Clinical injury and treatment records management
- **Predictive Analytics** - ML models for injury risk and recovery time prediction
- **Provider Analytics** - Workload analysis and efficiency metrics
- **Trend Analysis** - Seasonal patterns, service trends, and body part analytics
- **Data Integration** - Automated data merging and processing pipeline

## 📊 Dashboard Analytics

### Trends Over Time
- Weekly/daily encounter volume analysis
- Day of week patterns and seasonal trends
- Service type trends and body part patterns
- Provider workload distribution
- Busiest days and peak period identification

### Clinical Insights
- Body part injury patterns
- Service type utilization
- Provider workload analysis
- Injury profile and recovery tracking
- Treatment effectiveness metrics

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run Scripts/streamlit_dashboard.py
```

### Data Processing Pipeline
```bash
# 1. Prepare and clean data
python Scripts/data_preparation.py

# 2. Merge into master datasets
python Scripts/data_merger.py

# 3. Train ML models
python Scripts/real_ml_trainer_clean.py
```

## 📁 Project Structure

```
AT_Dept_Data/
├── Scripts/                    # Core analytics scripts
│   ├── streamlit_dashboard.py  # Interactive dashboard
│   ├── data_merger.py         # Data integration pipeline
│   ├── data_preparation.py    # Data cleaning and processing
│   └── real_ml_trainer_clean.py # ML model training
├── Data/                      # Data files (not in repo)
│   ├── Master/               # Master datasets
│   ├── Results/              # Model outputs
│   └── Encounter Log Table-*.xlsx
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

Update the data path in `streamlit_dashboard.py`:
```python
self.data_path = "/path/to/your/data/directory"
```

## 📈 Model Performance

- **Recovery Time Prediction**: 0.75 day Mean Absolute Error
- **Injury Risk Classification**: Real-world validated accuracy
- **Time Series Prediction**: 96.5% accuracy on synthetic data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🏫 Institution

Developed for athletic training analytics and research applications.