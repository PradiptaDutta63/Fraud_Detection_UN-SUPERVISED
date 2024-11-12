#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.svm import OneClassSVM

def one_class_svm_model(data):
    """Train One-Class SVM model to detect anomalies."""
    model = OneClassSVM(kernel="rbf", gamma=0.001, nu=0.05)
    model.fit(data)
    return model

def predict_anomalies(model, data):
    """Predict anomalies using the trained model."""
    return model.predict(data)  # -1 for anomaly, 1 for normal

