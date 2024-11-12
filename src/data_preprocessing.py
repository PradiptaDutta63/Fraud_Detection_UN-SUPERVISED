#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from a specified filepath."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    # Handling missing values
    df.dropna(inplace=True)
    
    # Scaling numerical features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    
    return df_scaled

