#!/usr/bin/env python3
"""
Denison University Athletic Training Clinic Dashboard

Interactive Streamlit dashboard for exploring athletic training clinic data.
Provides visualizations and analysis of treatment encounters and injury patterns.

Author: Advanced Data Analytics Student
Date: 2025

Usage:
    streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any
import dtale
from streamlit.components.v1 import html


class ATDashboard:
    """Denison University Athletic Training Dashboard for data visualization and analysis."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.injury_df = None
        self.treatment_df = None
        self.encounter_df = None
        self.summary = {}
        self.data_path = "/Users/philip_koutsaftis/Library/CloudStorage/GoogleDrive-philipkoutsaftis@gmail.com/My Drive/AT_Dept_Data/Data"
    
    @st.cache_data
    def load_injury_data(_self, file) -> pd.DataFrame:
        """
        Load and process injury data with caching.
        
        Args:
            file: File path or uploaded file object
            
        Returns:
            Processed injury DataFrame
        """
        df = pd.read_excel(file)
        
        # Convert date columns
        date_cols = [
            "Problem Date",
            "Expected Return Date", 
            "Actual Return Date",
            "Reported Date",
            "Problem Created Date Time"
        ]
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Calculate days out
        if "Problem Date" in df.columns and "Actual Return Date" in df.columns:
            df["Days Out"] = (df["Actual Return Date"] - df["Problem Date"]).dt.days
        
        # Clean categorical columns
        categorical_cols = ["Body Part", "Condition", "Problem Status", "Datalys Injury Type"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Create clearance flag
        if "Problem Status" in df.columns:
            cleared_statuses = [
                "Returned To Play As Tolerated",
                "Returned To Play and Closed", 
                "Returned To Play and Closed, Returned To Play As Tolerated"
            ]
            df["Is Cleared"] = df["Problem Status"].isin(cleared_statuses)
        
        return df
    
    @st.cache_data
    def load_treatment_data(_self, file) -> pd.DataFrame:
        """
        Load and process treatment data with caching.
        
        Args:
            file: File path or uploaded file object
            
        Returns:
            Processed treatment DataFrame
        """
        df = pd.read_excel(file)
        
        # Convert date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["DateOnly"] = df["Date"].dt.date
            df["Week"] = df["Date"].dt.to_period("W-MON").dt.start_time
            df["DayOfWeek"] = df["Date"].dt.day_name()
        
        # Clean categorical columns
        categorical_cols = ["Body Part", "Service", "Treating Provider", "Missed", "Scheduled"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Create flags
        if "Missed" in df.columns:
            df["Missed_Flag"] = df["Missed"].str.lower().eq("yes")
        
        return df
    
    @st.cache_data
    def load_encounter_data(_self, file_path: str = None) -> pd.DataFrame:
        """
        Load and process encounter log data.
        
        Args:
            file_path: Path to encounter log file
            
        Returns:
            Processed encounter DataFrame
        """
        if file_path is None:
            file_path = f"{_self.data_path}/Encounter Log Table-12_1_2025.xlsx"
        
        try:
            df = pd.read_excel(file_path)
            
            # Convert date column
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df["DateOnly"] = df["Date"].dt.date
                df["Week"] = df["Date"].dt.to_period("W-MON").dt.start_time
                df["DayOfWeek"] = df["Date"].dt.day_name()
            
            # Clean categorical columns
            categorical_cols = ["Service", "Body Part", "Treating Provider", "Missed", "Scheduled"]
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            
            return df
        except Exception as e:
            st.warning(f"Could not load encounter data: {e}")
            return pd.DataFrame()
    
    def compute_summary(self, treatment_df: pd.DataFrame, injury_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics for the dashboard.
        
        Args:
            treatment_df: Treatment data
            injury_df: Injury data
            
        Returns:
            Dictionary of summary statistics
        """
        summary = {}
        
        # Treatment summary
        if not treatment_df.empty and "Date" in treatment_df.columns:
            summary["treat_start"] = treatment_df["Date"].min()
            summary["treat_end"] = treatment_df["Date"].max()
            summary["total_encounters"] = len(treatment_df)
            
            daily_counts = treatment_df.groupby("DateOnly").size()
            summary["num_days"] = len(daily_counts)
            summary["avg_per_day"] = daily_counts.mean() if len(daily_counts) > 0 else np.nan
            
            weekly_counts = treatment_df.groupby("Week").size()
            summary["num_weeks"] = len(weekly_counts)
            summary["avg_per_week"] = weekly_counts.mean() if len(weekly_counts) > 0 else np.nan
            
            summary["daily_counts"] = daily_counts
            summary["weekly_counts"] = weekly_counts
            summary["busiest_days"] = daily_counts.sort_values(ascending=False).head(10)
        else:
            summary.update({
                "treat_start": None, "treat_end": None, "total_encounters": 0,
                "num_days": 0, "avg_per_day": np.nan, "num_weeks": 0, "avg_per_week": np.nan,
                "daily_counts": pd.Series(dtype=int), "weekly_counts": pd.Series(dtype=int),
                "busiest_days": pd.Series(dtype=int)
            })
        
        # Injury summary
        if not injury_df.empty and "Problem Date" in injury_df.columns:
            summary["inj_start"] = injury_df["Problem Date"].min()
            summary["inj_end"] = injury_df["Problem Date"].max()
            summary["total_injuries"] = len(injury_df)
        else:
            summary.update({
                "inj_start": None, "inj_end": None, "total_injuries": 0
            })
        
        return summary
    
    def render_header(self):
        """Render the dashboard header."""
        st.set_page_config(
            page_title="Athletic Training Clinic Dashboard",
            layout="wide"
        )
        
        st.title("🏥 Athletic Training Clinic – Analytics Dashboard")
        st.markdown(
            "**Real-time analysis** of injury and treatment data from your athletic training clinic. "
            "Data automatically loaded from master files."
        )
        
        # Load data automatically
        try:
            encounter_df = self.load_encounter_data()
            injury_df = self.load_injury_data(f"{self.data_path}/Master/Injury_Master.xlsx")
            treatment_df = self.load_treatment_data(f"{self.data_path}/Master/Treatment_Master.xlsx")
            
            # Show data source info
            with st.expander("📊 Data Sources", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Encounter Records", f"{len(encounter_df):,}")
                    st.caption("From: Encounter Log Table")
                with col2:
                    st.metric("Treatment Records", f"{len(treatment_df):,}")
                    st.caption("From: Treatment Master")
                with col3:
                    st.metric("Injury Records", f"{len(injury_df):,}")
                    st.caption("From: Injury Master")
            
            return injury_df, treatment_df, encounter_df
            
        except Exception as e:
            st.error(f"Error loading data files: {e}")
            st.info("Make sure to run the data merger first: `python data_merger.py`")
            st.stop()
    
    def render_kpis(self, summary: Dict[str, Any]):
        """Render key performance indicators."""
        st.subheader("📊 Key Clinic Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Encounters", f"{summary['total_encounters']:,}")
        
        with col2:
            st.metric("Total Injuries", f"{summary['total_injuries']:,}")
        
        with col3:
            if not np.isnan(summary["avg_per_day"]):
                st.metric("Avg Encounters / Day", f"{summary['avg_per_day']:.1f}")
            else:
                st.metric("Avg Encounters / Day", "—")
        
        with col4:
            if not np.isnan(summary["avg_per_week"]):
                st.metric("Avg Encounters / Week", f"{summary['avg_per_week']:.1f}")
            else:
                st.metric("Avg Encounters / Week", "—")
        
        # Date ranges
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            st.write(
                f"**Treatment data range:** "
                f"{summary['treat_start'].date() if summary['treat_start'] else '—'} "
                f"→ {summary['treat_end'].date() if summary['treat_end'] else '—'}"
            )
        with date_col2:
            st.write(
                f"**Injury data range:** "
                f"{summary['inj_start'].date() if summary['inj_start'] else '—'} "
                f"→ {summary['inj_end'].date() if summary['inj_end'] else '—'}"
            )
        
        st.markdown("---")
    
    def render_trends_tab(self, summary: Dict[str, Any], treatment_df: pd.DataFrame):
        """Render trends over time tab with flexible time range slider."""
        st.subheader("📈 Encounter Volume Analysis")
        
        # Time range selector
        st.markdown("### 🕐 Select Analysis Time Window")
        
        col1, col2 = st.columns(2)
        with col1:
            time_window = st.selectbox(
                "Quick Select:",
                ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months", "1 Year", "All Time", "Custom Range"],
                index=6  # Default to "All Time"
            )
        
        # Calculate date range based on selection
        if 'Date' in treatment_df.columns:
            max_date = treatment_df['Date'].max()
            min_date = treatment_df['Date'].min()
            
            if time_window == "1 Week":
                start_date = max_date - pd.Timedelta(days=7)
            elif time_window == "2 Weeks":
                start_date = max_date - pd.Timedelta(days=14)
            elif time_window == "1 Month":
                start_date = max_date - pd.Timedelta(days=30)
            elif time_window == "3 Months":
                start_date = max_date - pd.Timedelta(days=90)
            elif time_window == "6 Months":
                start_date = max_date - pd.Timedelta(days=180)
            elif time_window == "1 Year":
                start_date = max_date - pd.Timedelta(days=365)
            elif time_window == "Custom Range":
                with col2:
                    date_range = st.date_input(
                        "Select date range:",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                    if len(date_range) == 2:
                        start_date = pd.to_datetime(date_range[0])
                        max_date = pd.to_datetime(date_range[1])
                    else:
                        start_date = min_date
            else:  # All Time
                start_date = min_date
            
            # Filter data
            filtered_df = treatment_df[
                (treatment_df['Date'] >= start_date) & 
                (treatment_df['Date'] <= max_date)
            ]
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Total Encounters", f"{len(filtered_df):,}")
            with metric_col2:
                days_span = (max_date - start_date).days
                st.metric("Days in Range", f"{days_span}")
            with metric_col3:
                avg_per_day = len(filtered_df) / max(days_span, 1)
                st.metric("Avg/Day", f"{avg_per_day:.1f}")
            
            st.markdown("---")
        
        # Weekly trend
        st.subheader("📊 Weekly Encounter Volume")
        weekly_counts = summary["weekly_counts"]
        if weekly_counts.empty:
            st.warning("No weekly data available.")
        else:
            weekly_df = weekly_counts.reset_index()
            weekly_df.columns = ["Week", "Encounters"]
            
            # Filter weekly data
            if time_window != "All Time" and 'Date' in treatment_df.columns:
                weekly_df = weekly_df[
                    (weekly_df["Week"] >= start_date) & 
                    (weekly_df["Week"] <= max_date)
                ]
            
            fig = px.line(
                weekly_df,
                x="Week",
                y="Encounters", 
                markers=True,
                title=f"Weekly Treatment Encounters ({time_window})"
            )
            fig.update_layout(xaxis_title="Week", yaxis_title="Number of Encounters")
            st.plotly_chart(fig, use_container_width=True)
        
        # Daily trend (only for shorter time windows)
        if time_window in ["1 Week", "2 Weeks", "1 Month", "Custom Range"]:
            st.subheader("📅 Daily Encounter Volume")
            daily_counts = summary["daily_counts"]
            if not daily_counts.empty:
                daily_df = daily_counts.reset_index()
                daily_df.columns = ["Date", "Encounters"]
                daily_df['Date'] = pd.to_datetime(daily_df['Date'])
                
                # Filter daily data
                daily_df = daily_df[
                    (daily_df["Date"] >= start_date) & 
                    (daily_df["Date"] <= max_date)
                ]
                
                fig_daily = px.bar(
                    daily_df,
                    x="Date",
                    y="Encounters",
                    title=f"Daily Treatment Encounters ({time_window})"
                )
                fig_daily.update_layout(xaxis_title="Date", yaxis_title="Encounters")
                st.plotly_chart(fig_daily, use_container_width=True)
        
        st.subheader("📅 Busiest Days in Selected Range")
        busiest = summary["busiest_days"]
        if not busiest.empty:
            busiest_df = busiest.reset_index()
            busiest_df.columns = ["Date", "Encounters"]
            busiest_df['Date'] = pd.to_datetime(busiest_df['Date'])
            
            # Filter busiest days
            if time_window != "All Time" and 'Date' in treatment_df.columns:
                busiest_df = busiest_df[
                    (busiest_df["Date"] >= start_date) & 
                    (busiest_df["Date"] <= max_date)
                ].head(10)
            
            st.dataframe(busiest_df)
        else:
            st.info("No busiest day data available.")
        
        # Additional Trend Analysis
        st.markdown("---")
        st.subheader("📊 Additional Trend Analysis")
        
        # 1. Day of Week Patterns
        if "DayOfWeek" in treatment_df.columns:
            st.markdown("### 📅 Day of Week Patterns")
            dow_counts = treatment_df["DayOfWeek"].value_counts().reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            ).fillna(0)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_dow = px.bar(
                    x=dow_counts.index, y=dow_counts.values,
                    title="Encounters by Day of Week",
                    labels={"x": "Day", "y": "Encounters"}
                )
                st.plotly_chart(fig_dow, use_container_width=True)
            
            with col2:
                # Weekday vs Weekend
                weekday_total = dow_counts[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].sum()
                weekend_total = dow_counts[["Saturday", "Sunday"]].sum()
                
                fig_wknd = px.pie(
                    values=[weekday_total, weekend_total],
                    names=["Weekdays", "Weekend"],
                    title="Weekday vs Weekend Distribution"
                )
                st.plotly_chart(fig_wknd, use_container_width=True)
        
        # 2. Service Type Trends
        if "Service" in treatment_df.columns and "Date" in treatment_df.columns:
            st.markdown("### 💉 Service Type Trends Over Time")
            
            # Get top 5 services
            top_services = treatment_df["Service"].value_counts().head(5).index
            service_trends = treatment_df[treatment_df["Service"].isin(top_services)]
            
            if not service_trends.empty:
                service_monthly = service_trends.groupby([
                    service_trends["Date"].dt.to_period("M"), "Service"
                ]).size().reset_index(name="Count")
                service_monthly["Month"] = service_monthly["Date"].dt.start_time
                
                fig_service = px.line(
                    service_monthly, x="Month", y="Count", color="Service",
                    title="Top 5 Services Over Time (Monthly)"
                )
                st.plotly_chart(fig_service, use_container_width=True)
        
        # 3. Body Part Trends
        if "Body Part" in treatment_df.columns and "Date" in treatment_df.columns:
            st.markdown("### 🦵 Body Part Trends Over Time")
            
            # Get top 5 body parts
            top_body_parts = treatment_df["Body Part"].value_counts().head(5).index
            bp_trends = treatment_df[treatment_df["Body Part"].isin(top_body_parts)]
            
            if not bp_trends.empty:
                bp_monthly = bp_trends.groupby([
                    bp_trends["Date"].dt.to_period("M"), "Body Part"
                ]).size().reset_index(name="Count")
                bp_monthly["Month"] = bp_monthly["Date"].dt.start_time
                
                fig_bp = px.line(
                    bp_monthly, x="Month", y="Count", color="Body Part",
                    title="Top 5 Body Parts Over Time (Monthly)"
                )
                st.plotly_chart(fig_bp, use_container_width=True)
        
        # 4. Provider Workload Trends
        if "Treating Provider" in treatment_df.columns and "Date" in treatment_df.columns:
            st.markdown("### 👨‍⚕️ Provider Workload Trends")
            
            # Get top 3 providers
            top_providers = treatment_df["Treating Provider"].value_counts().head(3).index
            provider_trends = treatment_df[treatment_df["Treating Provider"].isin(top_providers)]
            
            if not provider_trends.empty:
                provider_weekly = provider_trends.groupby([
                    provider_trends["Date"].dt.to_period("W"), "Treating Provider"
                ]).size().reset_index(name="Count")
                provider_weekly["Week"] = provider_weekly["Date"].dt.start_time
                
                fig_provider = px.line(
                    provider_weekly, x="Week", y="Count", color="Treating Provider",
                    title="Top 3 Providers Workload (Weekly)"
                )
                st.plotly_chart(fig_provider, use_container_width=True)
        
        # 7. Seasonal Analysis
        if "Date" in treatment_df.columns:
            st.markdown("### 🍂 Seasonal Analysis")
            
            # Add season column
            treatment_seasonal = treatment_df.copy()
            treatment_seasonal["Month"] = treatment_seasonal["Date"].dt.month
            treatment_seasonal["Season"] = treatment_seasonal["Month"].map({
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Fall", 10: "Fall", 11: "Fall"
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                season_counts = treatment_seasonal["Season"].value_counts().reindex(
                    ["Spring", "Summer", "Fall", "Winter"]
                ).fillna(0)
                
                fig_season = px.bar(
                    x=season_counts.index, y=season_counts.values,
                    title="Encounters by Season",
                    labels={"x": "Season", "y": "Encounters"}
                )
                st.plotly_chart(fig_season, use_container_width=True)
            
            with col2:
                # Monthly pattern
                monthly_counts = treatment_seasonal.groupby("Month").size()
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                
                fig_monthly = px.line(
                    x=[month_names[i-1] for i in monthly_counts.index],
                    y=monthly_counts.values,
                    title="Monthly Encounter Pattern",
                    labels={"x": "Month", "y": "Encounters"}
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    def render_providers_tab(self, treatment_df: pd.DataFrame):
        """Render providers analysis tab."""
        st.subheader("🧍 Encounters by Treating Provider")
        
        if "Treating Provider" not in treatment_df.columns:
            st.warning("No 'Treating Provider' column found in treatment data.")
        else:
            provider_counts = (
                treatment_df.groupby("Treating Provider")
                .size()
                .sort_values(ascending=False)
                .reset_index(name="Encounters")
            )
            
            st.dataframe(provider_counts)
            
            fig = px.bar(
                provider_counts,
                x="Treating Provider",
                y="Encounters",
                title="Total Encounters per Provider"
            )
            fig.update_layout(xaxis_title="Provider", yaxis_title="Encounters")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_body_parts_tab(self, treatment_df: pd.DataFrame, injury_df: pd.DataFrame):
        """Render body parts analysis tab."""
        st.subheader("🦵 Body Parts – Treatments vs Injuries")
        
        # Treatment body parts
        if "Body Part" in treatment_df.columns and not treatment_df.empty:
            treat_bp = treatment_df["Body Part"].value_counts().reset_index()
            treat_bp.columns = ["Body Part", "Treatment Count"]
        else:
            treat_bp = pd.DataFrame(columns=["Body Part", "Treatment Count"])
        
        # Injury body parts
        if "Body Part" in injury_df.columns and not injury_df.empty:
            inj_bp = injury_df["Body Part"].value_counts().reset_index()
            inj_bp.columns = ["Body Part", "Injury Count"]
        else:
            inj_bp = pd.DataFrame(columns=["Body Part", "Injury Count"])
        
        # Merge data
        merged_bp = pd.merge(treat_bp, inj_bp, on="Body Part", how="outer").fillna(0)
        
        top_n = st.slider("Show top N body parts", 5, 20, 10)
        merged_bp["Total"] = merged_bp["Treatment Count"] + merged_bp["Injury Count"]
        merged_bp_top = merged_bp.sort_values("Total", ascending=False).head(top_n)
        
        # Side-by-side charts
        col_bp1, col_bp2 = st.columns(2)
        with col_bp1:
            fig_treat = px.bar(
                merged_bp_top,
                x="Body Part",
                y="Treatment Count",
                title="Treatments by Body Part"
            )
            fig_treat.update_layout(xaxis_title="Body Part", yaxis_title="Treatments")
            st.plotly_chart(fig_treat, use_container_width=True)
        
        with col_bp2:
            fig_inj = px.bar(
                merged_bp_top,
                x="Body Part",
                y="Injury Count",
                title="Injuries by Body Part"
            )
            fig_inj.update_layout(xaxis_title="Body Part", yaxis_title="Injuries")
            st.plotly_chart(fig_inj, use_container_width=True)
        
        st.markdown("### Combined Table")
        st.dataframe(merged_bp_top[["Body Part", "Treatment Count", "Injury Count"]])
    
    def render_services_tab(self, treatment_df: pd.DataFrame):
        """Render service types analysis tab."""
        st.subheader("💉 Service Types")
        
        if "Service" not in treatment_df.columns:
            st.warning("No 'Service' column found in treatment data.")
        else:
            service_counts = (
                treatment_df["Service"].value_counts().reset_index(name="Encounters")
            )
            top_n_services = st.slider("Show top N services", 5, 30, 15)
            
            service_top = service_counts.head(top_n_services)
            st.dataframe(service_top)
            
            fig = px.bar(
                service_top,
                x="Service",
                y="Encounters",
                title="Most Common Service Types"
            )
            fig.update_layout(xaxis_title="Service", yaxis_title="Encounters")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_injury_profile_tab(self, injury_df: pd.DataFrame):
        """Render injury profile analysis tab."""
        st.subheader("🚑 Injury Profile")
        
        col_ip1, col_ip2 = st.columns(2)
        
        # Injury conditions
        if "Condition" in injury_df.columns:
            cond_counts = injury_df["Condition"].value_counts().reset_index(name="Count")
            with col_ip1:
                st.markdown("**Injury Conditions**")
                st.dataframe(cond_counts)
                fig_cond = px.bar(
                    cond_counts.head(15),
                    x="Condition",
                    y="Count",
                    title="Most Common Injury Conditions"
                )
                st.plotly_chart(fig_cond, use_container_width=True)
        else:
            st.warning("No 'Condition' column found in injury data.")
        
        # Problem status
        if "Problem Status" in injury_df.columns:
            status_counts = injury_df["Problem Status"].value_counts().reset_index(name="Count")
            with col_ip2:
                st.markdown("**Problem Status**")
                st.dataframe(status_counts)
                fig_status = px.bar(
                    status_counts,
                    x="Problem Status",
                    y="Count",
                    title="Injury Status Distribution"
                )
                st.plotly_chart(fig_status, use_container_width=True)
        
        st.markdown("---")
        
        # Days out analysis
        if "Days Out" in injury_df.columns:
            st.subheader("⏱ Days Out (for resolved injuries)")
            valid_days = injury_df["Days Out"].dropna()
            if not valid_days.empty:
                fig_days = px.histogram(
                    valid_days,
                    nbins=20,
                    title="Distribution of Days Out (where Actual Return Date is known)"
                )
                fig_days.update_layout(xaxis_title="Days Out", yaxis_title="Number of Injuries")
                st.plotly_chart(fig_days, use_container_width=True)
            else:
                st.info("No valid 'Days Out' data to display.")
        else:
            st.info("No 'Days Out' column found. Check injury data processing.")
    
    def render_raw_data_tab(self, injury_df: pd.DataFrame, treatment_df: pd.DataFrame):
        """Render raw data tab with D-Tale integration."""
        st.subheader("📄 Raw Data Explorer")
        
        # Dataset selector
        dataset_choice = st.radio(
            "Select dataset to explore:",
            ["Treatment Data", "Injury Data"],
            horizontal=True
        )
        
        selected_df = treatment_df if dataset_choice == "Treatment Data" else injury_df
        
        # Display options
        st.markdown("### Quick Preview")
        st.dataframe(selected_df.head(100))
        
        st.markdown("---")
        
        # D-Tale integration
        st.markdown("### 🔍 Interactive D-Tale Explorer")
        st.markdown(
            "Click the button below to open an interactive D-Tale window for advanced data exploration, "
            "filtering, visualization, and analysis."
        )
        
        if st.button(f"🚀 Launch D-Tale for {dataset_choice}", type="primary"):
            # Launch D-Tale
            d = dtale.show(selected_df, ignore_duplicate=True)
            
            # Display D-Tale URL
            st.success(f"✅ D-Tale launched successfully!")
            st.markdown(f"**D-Tale URL:** [{d._url}]({d._url})")
            st.info(
                "💡 **Tip:** The D-Tale window will open in a new browser tab. "
                "You can filter, sort, create charts, and perform advanced analysis there."
            )
            
            # Embed D-Tale iframe (optional)
            if st.checkbox("Embed D-Tale in dashboard (experimental)"):
                html(f'<iframe src="{d._url}" width="100%" height="800"></iframe>', height=800)
    
    def run(self):
        """Run the Streamlit dashboard."""
        # Load data automatically
        injury_df, treatment_df, encounter_df = self.render_header()
        
        # Use encounter data as primary treatment data if available
        primary_treatment_df = encounter_df if not encounter_df.empty else treatment_df
        
        # Compute summary
        summary = self.compute_summary(primary_treatment_df, injury_df)
        
        # Render KPIs
        self.render_kpis(summary)
        
        # Create tabs
        tabs = st.tabs([
            "📈 Trends Over Time",
            "🧍 Providers", 
            "🦵 Body Parts",
            "💉 Service Types",
            "🚑 Injury Profile",
            "📄 Raw Data"
        ])
        
        with tabs[0]:
            self.render_trends_tab(summary, primary_treatment_df)
        
        with tabs[1]:
            self.render_providers_tab(primary_treatment_df)
        
        with tabs[2]:
            self.render_body_parts_tab(primary_treatment_df, injury_df)
        
        with tabs[3]:
            self.render_services_tab(primary_treatment_df)
        
        with tabs[4]:
            self.render_injury_profile_tab(injury_df)
        
        with tabs[5]:
            self.render_raw_data_tab(injury_df, treatment_df)


def main():
    """Main function to run the dashboard."""
    dashboard = ATDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()