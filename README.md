# Market Anomaly Detection System
A market surveillance and anomaly detection platform designed to identify potential fraud, market manipulation, and unusual trading activity using statistical and machine learning techniques.

---

# Project Overview

Financial markets generate high-frequency data that makes manual monitoring ineffective.  
This project implements an automated anomaly detection system that detects suspicious trading behavior, prioritizes alerts, and supports investigation workflows while managing false positives.

---

# Objectives

- Detect market anomalies using statistical and ML-based techniques
- Identify potential fraud and manipulation patterns
- Prioritize alerts for investigation teams
- Implement analyst investigation workflows
- Reduce false positives through threshold tuning

---

# Detection Techniques

### 1️⃣ Statistical Detection
- Z-score based outlier detection
- Flags extreme deviations in trading volume

# Machine Learning – Isolation Forest
- Unsupervised anomaly detection
- Detects abnormal behavior in volume and price data

# Machine Learning – Autoencoder
- Neural network-based reconstruction error detection
- Flags observations with high reconstruction loss

---

# Alert Scoring & Prioritization

- Combines multiple anomaly signals into a unified anomaly score
- Alerts are prioritized based on severity
- Enables analysts to focus on high-risk events first

---

# Pattern Recognition

- Detects rapid volume spikes
- Correlates anomalies with suspicious trading patterns
- Enhances fraud and manipulation detection

---

# False Positive Management

- Adjustable anomaly score threshold
- Analyst-driven investigation actions
- Helps reduce alert fatigue in real-world systems

---

# System Architecture

Market Data  
→ Statistical & ML Scoring  
→ Anomaly Score Aggregation  
→ Alert Prioritization  
→ Investigation Workflow  
→ Threshold Adjustment

---

# Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

# Project Structure

Market-Anomaly-Detection/
- anomaly_detection.py
- app.py
- README.md
- requirements.txt

---

# How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
