#!/usr/bin/env python3
"""
Athletic Training Clinic - Automated Email Report

Generates and sends weekly/monthly summary reports via email.
Includes key statistics, trends, and visualizations.

Author: Advanced Data Analytics Student
Date: 2025
"""

import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.io as pio


class ATEmailReporter:
    """Automated email reporter for clinic statistics."""
    
    def __init__(self, treatment_file: str, injury_file: str):
        """Initialize reporter with data files."""
        self.treatment_df = pd.read_excel(treatment_file)
        self.injury_df = pd.read_excel(injury_file)
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis."""
        # Convert dates
        if 'Date' in self.treatment_df.columns:
            self.treatment_df['Date'] = pd.to_datetime(self.treatment_df['Date'], errors='coerce')
        
        if 'Problem Date' in self.injury_df.columns:
            self.injury_df['Problem Date'] = pd.to_datetime(self.injury_df['Problem Date'], errors='coerce')
    
    def generate_summary_stats(self, days_back: int = 7) -> dict:
        """Generate summary statistics for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter recent data
        recent_treatments = self.treatment_df[
            self.treatment_df['Date'] >= cutoff_date
        ] if 'Date' in self.treatment_df.columns else self.treatment_df
        
        recent_injuries = self.injury_df[
            self.injury_df['Problem Date'] >= cutoff_date
        ] if 'Problem Date' in self.injury_df.columns else self.injury_df
        
        stats = {
            'period_days': days_back,
            'total_encounters': len(recent_treatments),
            'total_injuries': len(recent_injuries),
            'avg_per_day': len(recent_treatments) / days_back if days_back > 0 else 0,
            'date_range': (cutoff_date.date(), datetime.now().date())
        }
        
        # Top providers
        if 'Treating Provider' in recent_treatments.columns:
            stats['top_providers'] = recent_treatments['Treating Provider'].value_counts().head(3).to_dict()
        
        # Top services
        if 'Service' in recent_treatments.columns:
            stats['top_services'] = recent_treatments['Service'].value_counts().head(5).to_dict()
        
        # Top body parts
        if 'Body Part' in recent_treatments.columns:
            stats['top_body_parts'] = recent_treatments['Body Part'].value_counts().head(5).to_dict()
        
        # Busiest day
        if 'Date' in recent_treatments.columns:
            daily_counts = recent_treatments.groupby(recent_treatments['Date'].dt.date).size()
            if not daily_counts.empty:
                busiest_day = daily_counts.idxmax()
                stats['busiest_day'] = (busiest_day, daily_counts.max())
        
        return stats
    
    def create_html_report(self, stats: dict) -> str:
        """Create HTML email body with statistics."""
        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px; padding: 15px; background-color: #f4f4f4; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .footer {{ text-align: center; padding: 20px; color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 Athletic Training Clinic Report</h1>
                <p>Period: {stats['date_range'][0]} to {stats['date_range'][1]}</p>
                <p><small>Generated: {generation_time}</small></p>
            </div>
            
            <div class="section">
                <h2>📊 Key Statistics ({stats['period_days']} Days)</h2>
                <div class="metric">
                    <div class="metric-value">{stats['total_encounters']}</div>
                    <div class="metric-label">Total Encounters</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['total_injuries']}</div>
                    <div class="metric-label">New Injuries</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats['avg_per_day']:.1f}</div>
                    <div class="metric-label">Avg Encounters/Day</div>
                </div>
            </div>
        """
        
        # Busiest day
        if 'busiest_day' in stats:
            html += f"""
            <div class="section">
                <h2>📅 Busiest Day</h2>
                <p><strong>{stats['busiest_day'][0]}</strong> with <strong>{stats['busiest_day'][1]}</strong> encounters</p>
            </div>
            """
        
        # Top providers
        if 'top_providers' in stats:
            html += """
            <div class="section">
                <h2>🧍 Top Athletic Trainers</h2>
                <table>
                    <tr><th>Provider</th><th>Encounters</th></tr>
            """
            for provider, count in stats['top_providers'].items():
                html += f"<tr><td>{provider}</td><td>{count}</td></tr>"
            html += "</table></div>"
        
        # Top services
        if 'top_services' in stats:
            html += """
            <div class="section">
                <h2>💉 Most Common Services</h2>
                <table>
                    <tr><th>Service</th><th>Count</th></tr>
            """
            for service, count in stats['top_services'].items():
                html += f"<tr><td>{service}</td><td>{count}</td></tr>"
            html += "</table></div>"
        
        # Top body parts
        if 'top_body_parts' in stats:
            html += """
            <div class="section">
                <h2>🦵 Most Treated Body Parts</h2>
                <table>
                    <tr><th>Body Part</th><th>Treatments</th></tr>
            """
            for part, count in stats['top_body_parts'].items():
                html += f"<tr><td>{part}</td><td>{count}</td></tr>"
            html += "</table></div>"
        
        html += """
            <div class="footer">
                <p>This is an automated report from the Athletic Training Clinic Analytics System</p>
                <p>For detailed analysis, access the interactive dashboard</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def send_email(self, 
                   to_emails: list,
                   subject: str,
                   html_body: str,
                   results_dir: str,
                   smtp_server: str = "smtp.gmail.com",
                   smtp_port: int = 587,
                   sender_email: str = None,
                   sender_password: str = None):
        """Send email report and save to Results folder."""
        
        # Create Results directory if it doesn't exist
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save report with date in filename
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_filename = results_path / f"Clinic_Report_{date_str}.html"
        
        with open(report_filename, 'w') as f:
            f.write(html_body)
        print(f"📄 Report saved to: {report_filename}")
        
        if not sender_email or not sender_password:
            print("⚠️  Email credentials not provided. Report saved but not sent.")
            return
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = ', '.join(to_emails)
        
        # Attach HTML
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"✅ Email sent successfully to {', '.join(to_emails)}")
        except Exception as e:
            print(f"❌ Error sending email: {e}")
    
    def generate_and_send_report(self,
                                 to_emails: list,
                                 days_back: int = 7,
                                 results_dir: str = None,
                                 sender_email: str = None,
                                 sender_password: str = None):
        """Generate and send complete report."""
        
        # Generate stats
        print(f"Generating report for last {days_back} days...")
        stats = self.generate_summary_stats(days_back)
        
        # Create HTML
        html_body = self.create_html_report(stats)
        
        # Create subject
        period = "Weekly" if days_back == 7 else f"{days_back}-Day"
        subject = f"🏥 {period} Athletic Training Clinic Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Send email and save
        self.send_email(
            to_emails=to_emails,
            subject=subject,
            html_body=html_body,
            results_dir=results_dir,
            sender_email=sender_email,
            sender_password=sender_password
        )


def main():
    """Main execution function."""
    
    # Configuration
    BASE_PATH = "/Users/philip_koutsaftis/Library/CloudStorage/GoogleDrive-philipkoutsaftis@gmail.com/My Drive/AT_Dept_Data"
    DATA_PATH = f"{BASE_PATH}/Data/Master"
    RESULTS_DIR = f"{BASE_PATH}/Data/Results"
    
    TREATMENT_FILE = f"{DATA_PATH}/Treatment_Master.xlsx"
    INJURY_FILE = f"{DATA_PATH}/Injury_Master.xlsx"
    
    # Email settings
    TO_EMAILS = [
        "athletic.director@example.com",
        "head.trainer@example.com"
    ]
    
    # Optional: Set these as environment variables for security
    # import os
    # SENDER_EMAIL = os.getenv('SENDER_EMAIL')
    # SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
    
    SENDER_EMAIL = None  # Set your email
    SENDER_PASSWORD = None  # Set your app password
    
    # Report period (7 = weekly, 30 = monthly)
    DAYS_BACK = 7
    
    # Generate and send report
    try:
        reporter = ATEmailReporter(TREATMENT_FILE, INJURY_FILE)
        reporter.generate_and_send_report(
            to_emails=TO_EMAILS,
            days_back=DAYS_BACK,
            results_dir=RESULTS_DIR,
            sender_email=SENDER_EMAIL,
            sender_password=SENDER_PASSWORD
        )
        print("\n✅ Report generation completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
