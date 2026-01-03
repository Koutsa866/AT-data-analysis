#!/usr/bin/env python3
"""
Simplified Bayesian Athletic Training Injury Prediction Model

Uses Bayesian statistics with scipy to predict injury recovery times incorporating:
- Sport type, Gender, Position, Age, Body part, Injury type

Author: Advanced Data Analytics Student
Date: 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SimpleBayesianPredictor:
    """Simplified Bayesian model for injury prediction using conjugate priors."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.priors = {}
        self.posteriors = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.group_stats = {}
    
    def prepare_features(self, injury_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for Bayesian modeling."""
        df = injury_df.copy()
        
        # Filter for records with actual return dates
        df = df[df['Actual Return Date'].notna()].copy()
        
        if len(df) == 0:
            raise ValueError("No records with actual return dates found")
        
        # Target variable: Days out
        df['Days_Out'] = (df['Actual Return Date'] - df['Problem Date']).dt.days
        df['Days_Out'] = df['Days_Out'].clip(lower=0)  # No negative recovery times
        
        # Age calculation
        if 'Patient DOB' in df.columns:
            df['Patient DOB'] = pd.to_datetime(df['Patient DOB'], errors='coerce')
            df['Age'] = (df['Problem Date'] - df['Patient DOB']).dt.days / 365.25
            df['Age'] = df['Age'].fillna(20)
        else:
            df['Age'] = 22 - (df['Graduation Year'] - df['Problem Date'].dt.year)
            df['Age'] = df['Age'].fillna(20)
        
        # Create categorical features with defaults
        features = []
        
        # Demographics
        if 'Sport' not in df.columns:
            df['Sport'] = 'General'
        features.append('Sport')
        
        if 'Gender' not in df.columns:
            df['Gender'] = 'Unknown'
        features.append('Gender')
        
        if 'Position' not in df.columns:
            df['Position'] = 'General'
        features.append('Position')
        
        # Injury characteristics
        if 'Body Part' in df.columns:
            features.append('Body Part')
        if 'Condition' in df.columns:
            features.append('Condition')
        
        # Age groups for hierarchical structure
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 19, 21, 23, 100], 
                                labels=['Freshman', 'Sophomore', 'Junior', 'Senior+'])
        features.append('Age_Group')
        
        # Injury severity proxy
        if 'Days_to_Report' in df.columns:
            df['Days_to_Report'] = df['Days_to_Report'].fillna(0)
            df['Severity_Proxy'] = pd.cut(df['Days_to_Report'], 
                                         bins=[-1, 0, 1, 7, 100],
                                         labels=['Immediate', 'Next_Day', 'Week', 'Delayed'])
        else:
            df['Severity_Proxy'] = 'Unknown'
        features.append('Severity_Proxy')
        
        # Seasonal effects
        df['Season'] = df['Problem Date'].dt.month.map({
            8: 'Fall', 9: 'Fall', 10: 'Fall', 11: 'Fall',
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer'
        })
        features.append('Season')
        
        self.feature_columns = features
        return df[features + ['Age', 'Days_Out']].copy()
    
    def fit_bayesian_groups(self, df: pd.DataFrame):
        """Fit Bayesian models for each categorical group."""
        
        # Overall prior (weakly informative)
        overall_mean = df['Days_Out'].mean()
        overall_std = df['Days_Out'].std()
        
        # Prior parameters (normal-inverse-gamma conjugate prior)
        prior_mu = overall_mean
        prior_kappa = 1.0  # Weak prior on mean
        prior_alpha = 2.0  # Weak prior on variance
        prior_beta = prior_alpha * (overall_std ** 2)
        
        self.priors = {
            'mu': prior_mu,
            'kappa': prior_kappa,
            'alpha': prior_alpha,
            'beta': prior_beta
        }
        
        # Fit posterior for each group
        for feature in self.feature_columns:
            if feature in df.columns and df[feature].dtype == 'object':
                self.posteriors[feature] = {}
                
                for group in df[feature].unique():
                    if pd.isna(group):
                        continue
                        
                    group_data = df[df[feature] == group]['Days_Out']
                    n = len(group_data)
                    
                    if n > 0:
                        # Update posterior parameters
                        sample_mean = group_data.mean()
                        sample_var = group_data.var() if n > 1 else overall_std ** 2
                        
                        # Posterior parameters (conjugate update)
                        post_kappa = prior_kappa + n
                        post_mu = (prior_kappa * prior_mu + n * sample_mean) / post_kappa
                        post_alpha = prior_alpha + n / 2
                        post_beta = (prior_beta + 
                                   0.5 * (n - 1) * sample_var + 
                                   (prior_kappa * n * (sample_mean - prior_mu) ** 2) / (2 * post_kappa))
                        
                        # Store posterior parameters
                        self.posteriors[feature][group] = {
                            'n': n,
                            'mu': post_mu,
                            'kappa': post_kappa,
                            'alpha': post_alpha,
                            'beta': post_beta,
                            'sample_mean': sample_mean,
                            'sample_std': np.sqrt(sample_var)
                        }
        
        # Store overall statistics
        self.group_stats = {
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'n_total': len(df)
        }
    
    def train(self, injury_df: pd.DataFrame) -> dict:
        """Train the Bayesian model."""
        
        # Prepare data
        feature_df = self.prepare_features(injury_df)
        
        if len(feature_df) < 10:
            raise ValueError(f"Insufficient data. Need at least 10 records, got {len(feature_df)}")
        
        # Fit Bayesian groups
        self.fit_bayesian_groups(feature_df)
        
        self.is_trained = True
        
        # Calculate model performance metrics
        results = {
            'n_samples': len(feature_df),
            'overall_mean': self.group_stats['overall_mean'],
            'overall_std': self.group_stats['overall_std'],
            'n_groups': sum(len(groups) for groups in self.posteriors.values()),
            'group_effects': self._calculate_group_effects()
        }
        
        return results
    
    def _calculate_group_effects(self) -> dict:
        """Calculate effect sizes for each group."""
        effects = {}
        overall_mean = self.group_stats['overall_mean']
        
        for feature, groups in self.posteriors.items():
            effects[feature] = {}
            for group, params in groups.items():
                # Effect size as difference from overall mean
                effect = params['mu'] - overall_mean
                effects[feature][group] = {
                    'effect_size': effect,
                    'posterior_mean': params['mu'],
                    'n_observations': params['n']
                }
        
        return effects
    
    def predict(self, injury_data: pd.DataFrame, n_samples: int = 1000) -> dict:
        """Make Bayesian predictions with uncertainty."""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare prediction data
        df = injury_data.copy()
        
        # Add missing columns with defaults
        for col in self.feature_columns:
            if col not in df.columns:
                if col == 'Age':
                    df[col] = 20
                else:
                    df[col] = 'Unknown'
        
        predictions = []
        
        for idx, row in df.iterrows():
            # Collect relevant group posteriors
            relevant_posteriors = []
            
            for feature in self.feature_columns:
                if feature in self.posteriors and row[feature] in self.posteriors[feature]:
                    relevant_posteriors.append(self.posteriors[feature][row[feature]])
            
            if relevant_posteriors:
                # Combine posteriors (simple average for demonstration)
                combined_mu = np.mean([p['mu'] for p in relevant_posteriors])
                combined_alpha = np.mean([p['alpha'] for p in relevant_posteriors])
                combined_beta = np.mean([p['beta'] for p in relevant_posteriors])
                
                # Sample from posterior predictive distribution
                # First sample variance from inverse-gamma
                variance_samples = stats.invgamma.rvs(combined_alpha, scale=combined_beta, size=n_samples)
                
                # Then sample means from normal given variance
                prediction_samples = []
                for var in variance_samples:
                    pred = stats.norm.rvs(combined_mu, np.sqrt(var))
                    prediction_samples.append(max(0, pred))  # No negative recovery times
                
                prediction_samples = np.array(prediction_samples)
            else:
                # Use overall prior if no specific group information
                variance_samples = stats.invgamma.rvs(self.priors['alpha'], 
                                                    scale=self.priors['beta'], size=n_samples)
                prediction_samples = []
                for var in variance_samples:
                    pred = stats.norm.rvs(self.priors['mu'], np.sqrt(var))
                    prediction_samples.append(max(0, pred))
                
                prediction_samples = np.array(prediction_samples)
            
            # Calculate prediction statistics
            pred_stats = {
                'mean_days': float(np.mean(prediction_samples)),
                'median_days': float(np.median(prediction_samples)),
                'std_days': float(np.std(prediction_samples)),
                'ci_lower': float(np.percentile(prediction_samples, 2.5)),
                'ci_upper': float(np.percentile(prediction_samples, 97.5)),
                'probability_gt_14_days': float(np.mean(prediction_samples > 14)),
                'probability_gt_30_days': float(np.mean(prediction_samples > 30)),
                'probability_gt_60_days': float(np.mean(prediction_samples > 60))
            }
            
            predictions.append(pred_stats)
        
        # Return single prediction if only one row, otherwise list
        return predictions[0] if len(predictions) == 1 else predictions
    
    def plot_group_effects(self, save_path: str = None):
        """Plot posterior group effects."""
        
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (feature, groups) in enumerate(self.posteriors.items()):
            if i >= len(axes):
                break
                
            group_names = list(groups.keys())
            group_means = [groups[g]['mu'] for g in group_names]
            group_stds = [np.sqrt(groups[g]['beta'] / groups[g]['alpha']) for g in group_names]
            
            axes[i].errorbar(range(len(group_names)), group_means, yerr=group_stds, 
                           fmt='o', capsize=5)
            axes[i].set_title(f'{feature} Effects')
            axes[i].set_xticks(range(len(group_names)))
            axes[i].set_xticklabels(group_names, rotation=45)
            axes[i].set_ylabel('Recovery Days')
            axes[i].axhline(y=self.group_stats['overall_mean'], color='r', linestyle='--', 
                          label='Overall Mean')
            axes[i].legend()
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        import joblib
        
        model_data = {
            'priors': self.priors,
            'posteriors': self.posteriors,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'group_stats': self.group_stats
        }
        
        joblib.dump(model_data, f"{filepath}.pkl")
        print(f"Bayesian model saved to {filepath}.pkl")


def main():
    """Main execution function."""
    BASE_PATH = "/Users/philip_koutsaftis/Library/CloudStorage/GoogleDrive-philipkoutsaftis@gmail.com/My Drive/AT_Dept_Data"
    INJURY_FILE = f"{BASE_PATH}/Data/Master/Injury_Master.xlsx"
    MODEL_PATH = f"{BASE_PATH}/Data/Results/simple_bayesian_model"
    
    try:
        print("Loading data...")
        injury_df = pd.read_excel(INJURY_FILE)
        print(f"Loaded {len(injury_df)} injury records")
        
        # Initialize predictor
        predictor = SimpleBayesianPredictor()
        
        print("Training Bayesian model...")
        results = predictor.train(injury_df)
        
        print("\n" + "="*50)
        print("BAYESIAN MODEL RESULTS")
        print("="*50)
        print(f"Training samples: {results['n_samples']}")
        print(f"Overall mean recovery: {results['overall_mean']:.1f} days")
        print(f"Overall std: {results['overall_std']:.1f} days")
        print(f"Number of groups: {results['n_groups']}")
        
        print("\nGroup Effects (difference from overall mean):")
        for feature, groups in results['group_effects'].items():
            print(f"\n{feature}:")
            for group, stats in groups.items():
                effect = stats['effect_size']
                n_obs = stats['n_observations']
                print(f"  {group}: {effect:+.1f} days (n={n_obs})")
        
        # Save model
        Path(f"{BASE_PATH}/Data/Results").mkdir(parents=True, exist_ok=True)
        predictor.save_model(MODEL_PATH)
        
        # Example prediction
        print("\n" + "="*50)
        print("BAYESIAN PREDICTION EXAMPLE")
        print("="*50)
        
        sample_injury = injury_df.iloc[:1].copy()
        sample_injury['Actual Return Date'] = pd.NaT
        
        pred_results = predictor.predict(sample_injury)
        
        print(f"Predicted recovery time:")
        print(f"  Mean: {pred_results['mean_days']:.1f} days")
        print(f"  Median: {pred_results['median_days']:.1f} days")
        print(f"  95% CI: [{pred_results['ci_lower']:.1f}, {pred_results['ci_upper']:.1f}] days")
        print(f"  Probability > 14 days: {pred_results['probability_gt_14_days']:.1%}")
        print(f"  Probability > 30 days: {pred_results['probability_gt_30_days']:.1%}")
        
        # Plot group effects
        predictor.plot_group_effects(f"{BASE_PATH}/Data/Results/bayesian_group_effects.png")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Run data_preparation.py to create enhanced variables")
        print("2. Have sufficient data with actual return dates")


if __name__ == "__main__":
    main()