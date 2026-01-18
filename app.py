import streamlit as st
import pandas as pd
from anomaly_detection import AnomalyDetector, generate_sample_data
import matplotlib.pyplot as plt
import traceback  # For detailed error logging

st.title("Market Anomaly Detection Platform")

# Debug mode toggle
debug = st.checkbox("Enable Debug Mode (shows error details)")

# Load or generate data
if st.button("Load Sample Data"):
    try:
        data = generate_sample_data()
        st.session_state['data'] = data
        st.success("Data Loaded!")
        st.write(data.head())
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        if debug:
            st.code(traceback.format_exc())

if 'data' in st.session_state:
    detector = AnomalyDetector(st.session_state['data'])
    
    # Run detections with updates, checks, and error handling
    st.header("Anomaly Scoring")
    
    if st.button("Run Statistical Scoring"):
        try:
            if 'volume' not in st.session_state['data'].columns:
                st.error("Data missing 'volume' column. Reload data.")
            else:
                detector.statistical_scoring()
                st.session_state['data'] = detector.data
                st.success("Statistical Anomalies Detected")
                st.write(st.session_state['data'][['timestamp', 'volume', 'z_score', 'anomaly_stat']].head())
        except Exception as e:
            st.error(f"Error in Statistical Scoring: {str(e)}")
            if debug:
                st.code(traceback.format_exc())
    
    if st.button("Run ML Scoring (Isolation Forest)"):
        try:
            if not all(col in st.session_state['data'].columns for col in ['volume', 'price']):
                st.error("Data missing 'volume' or 'price' columns. Reload data.")
            else:
                detector.ml_scoring_isolation_forest()
                st.session_state['data'] = detector.data
                st.success("ML Anomalies Detected (Isolation Forest)")
                st.write(st.session_state['data'][['timestamp', 'volume', 'anomaly_ml']].head())
        except Exception as e:
            st.error(f"Error in Isolation Forest Scoring: {str(e)}")
            if debug:
                st.code(traceback.format_exc())
    
    if st.button("Run ML Scoring (Autoencoder)"):
        try:
            if not all(col in st.session_state['data'].columns for col in ['volume', 'price']):
                st.error("Data missing 'volume' or 'price' columns. Reload data.")
            else:
                detector.ml_scoring_autoencoder()
                st.session_state['data'] = detector.data
                st.success("ML Anomalies Detected (Autoencoder)")
                st.write(st.session_state['data'][['timestamp', 'volume', 'anomaly_ae']].head())
        except Exception as e:
            st.error(f"Error in Autoencoder Scoring: {str(e)}")
            if debug:
                st.code(traceback.format_exc())
    
    if st.button("Combine Scores and Prioritize"):
        try:
            required_cols = ['anomaly_stat', 'anomaly_ml', 'anomaly_ae']
            if not all(col in st.session_state['data'].columns for col in required_cols):
                st.error(f"Run individual scorings first. Missing: {[col for col in required_cols if col not in st.session_state['data'].columns]}")
            else:
                detector.combined_score()
                alerts = detector.prioritize_alerts()
                st.session_state['data'] = detector.data
                st.success("Scores Combined and Alerts Prioritized")
                st.write("Top Alerts:", alerts[['timestamp', 'volume', 'anomaly_score']].head())
        except Exception as e:
            st.error(f"Error in Combining Scores: {str(e)}")
            if debug:
                st.code(traceback.format_exc())
    
    # Pattern Recognition
    if st.button("Detect Patterns"):
        try:
            if 'anomaly_score' not in st.session_state['data'].columns:
                st.error("Run 'Combine Scores' first to generate 'anomaly_score'.")
            else:
                detector.pattern_recognition()
                st.session_state['data'] = detector.data
                st.success("Patterns Detected (e.g., Volume Spikes)")
                st.write(st.session_state['data'][['timestamp', 'volume', 'volume_spike', 'pattern_alert']].head())
        except Exception as e:
            st.error(f"Error in Pattern Detection: {str(e)}")
            if debug:
                st.code(traceback.format_exc())
    
    # Visualization
    st.header("Visualizations")
    try:
        if 'anomaly_score' in st.session_state['data'].columns:
            fig, ax = plt.subplots()
            ax.plot(st.session_state['data']['timestamp'], st.session_state['data']['volume'], label='Volume')
            anomalies = st.session_state['data'][st.session_state['data']['anomaly_score'] > 0.5]
            ax.scatter(anomalies['timestamp'], anomalies['volume'], color='red', label='Anomalies')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Run scorings and combine scores to enable visualizations.")
    except Exception as e:
        st.error(f"Error in Visualization: {str(e)}")
        if debug:
            st.code(traceback.format_exc())
    
    # Investigation Workflow
    st.header("Investigation Workflow")
    try:
        if 'anomaly_score' in st.session_state['data'].columns and not st.session_state['data'][st.session_state['data']['anomaly_score'] > 0.5].empty:
            alerts_df = st.session_state['data'][st.session_state['data']['anomaly_score'] > 0.5]
            selected_alert = st.selectbox("Select Alert to Investigate", alerts_df.index)
            action = st.selectbox("Action", ["Review", "Escalate", "Dismiss as False Positive"])
            if st.button("Submit Action"):
                if action == "Dismiss as False Positive":
                    st.success("Feedback noted. Model will be updated in next run.")
                st.write(f"Action '{action}' recorded for alert {selected_alert}.")
        else:
            st.warning("No alerts available. Run detections first.")
    except Exception as e:
        st.error(f"Error in Investigation Workflow: {str(e)}")
        if debug:
            st.code(traceback.format_exc())
    
    # False Positive Management
    st.header("False Positive Management")
    try:
        if 'anomaly_score' in st.session_state['data'].columns:
            threshold = st.slider("Adjust Anomaly Threshold", 0.0, 1.0, 0.5)
            if st.button("Update Threshold"):
                st.session_state['data']['anomaly_score'] = (st.session_state['data']['anomaly_score'] > threshold).astype(int)
                st.success("Threshold updated.")
        else:
            st.warning("Run scorings first to adjust thresholds.")
    except Exception as e:
        st.error(f"Error in False Positive Management: {str(e)}")
        if debug:
            st.code(traceback.format_exc())
else:
    st.warning("Load data first to proceed.")