#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import IsolationForest

def isolation_forest_model(data):
    """Train Isolation Forest model to detect anomalies."""
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)
    return model

def predict_anomalies(model, data):
    """Predict anomalies using the trained model."""
    return model.predict(data)  # -1 for anomaly, 1 for normal

