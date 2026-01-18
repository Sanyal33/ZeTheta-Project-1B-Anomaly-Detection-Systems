import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.models = {}

    def statistical_scoring(self, column='volume'):
        # Z-score for outliers
        mean = self.data[column].mean()
        std = self.data[column].std()
        self.data['z_score'] = (self.data[column] - mean) / std
        self.data['anomaly_stat'] = (self.data['z_score'].abs() > 3).astype(int)  # Threshold: 3 SD

    def ml_scoring_isolation_forest(self, features=['volume', 'price']):
        X = self.scaler.fit_transform(self.data[features])
        model = IsolationForest(contamination=0.1, random_state=42)
        self.data['anomaly_ml'] = model.fit_predict(X)
        self.data['anomaly_ml'] = (self.data['anomaly_ml'] == -1).astype(int)  # -1 is anomaly
        self.models['isolation_forest'] = model

    def ml_scoring_autoencoder(self, features=['volume', 'price'], epochs=50):
        X = self.scaler.fit_transform(self.data[features])
        model = Sequential([
            Dense(64, activation='relu', input_shape=(len(features),)),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(features), activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, X, epochs=epochs, verbose=0)
        reconstructions = model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 95)  # Top 5% as anomalies
        self.data['anomaly_ae'] = (mse > threshold).astype(int)
        self.models['autoencoder'] = model

    def combined_score(self):
        # Aggregate scores (e.g., average)
        self.data['anomaly_score'] = self.data[['anomaly_stat', 'anomaly_ml', 'anomaly_ae']].mean(axis=1)

    def prioritize_alerts(self):
        # Sort by score descending
        alerts = self.data[self.data['anomaly_score'] > 0.5].sort_values('anomaly_score', ascending=False)
        return alerts

    def pattern_recognition(self):
        # Simple pattern: Detect rapid volume spikes (e.g., >2x previous)
        self.data['volume_spike'] = (self.data['volume'] > 2 * self.data['volume'].shift(1)).astype(int)
        # Combine with anomalies
        self.data['pattern_alert'] = self.data['anomaly_score'] * self.data['volume_spike']

# Sample data generation
def generate_sample_data(n=1000):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='H')
    volumes = np.random.normal(1000, 200, n)
    prices = np.random.normal(100, 10, n)
    # Inject anomalies
    volumes[100:110] *= 5  # Spike
    prices[200:210] += 50  # Manipulation
    return pd.DataFrame({'timestamp': dates, 'volume': volumes, 'price': prices})