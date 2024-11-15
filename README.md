# Fraud Detection model

## Introduction
This project implements an anomaly detection model using the Isolation Forest algorithm. The goal is to identify anomalies or outliers within a dataset. Anomalies are data points that deviate significantly from the majority of the data. 
I decided to create an anomaly detection model for the Open Banking project because ensuring the security and integrity of financial transactions is paramount in the era of digital finance. Open Banking initiatives have ushered in a new era of transparency and accessibility in the financial sector, but they also introduce new challenges, particularly in identifying and mitigating fraudulent activities. By developing an effective anomaly detection model, I aim to contribute to the security of Open Banking platforms, safeguarding the interests of users, financial institutions, and the broader financial ecosystem. This project serves as a proactive step towards maintaining the trust and confidence of users in the open banking landscape while fostering innovation and financial inclusivity.

## Features
- Anomaly detection using the Isolation Forest algorithm.
- If applicable add additional features.

## Data
The model was trained and tested on "Sample getTransactions API data". The dataset is available in the repo.

## Importance of fraud detection in Banking:

1.	Protection of Customer Assets:

    •	Fraud detection helps protect customers from unauthorized transactions, identity theft, and financial losses. It ensures that their hard-earned money and assets are secure.
  	
2.	Maintaining Trust:
   
    •	Trust is the foundation of the banking industry. Detecting and preventing fraud ensures that customers trust their banks with their financial transactions and information.
  	
3.	Financial Stability:

    •	Fraud can have a detrimental impact on a bank's financial stability. Large-scale fraud incidents can lead to significant financial losses, which may ultimately affect the bank's viability.


## Requirements
The project relies on several Python libraries commonly used in data science and machine learning. Install them using the ```requirements.txt``` file:

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Setup and Usage
1. Clone the Repository:

```bash
git clone https://github.com/pradiptadutta63/Fraud_Detection_UN-SUPERVISED.git
cd Fraud_Detection_UN-SUPERVISED
```

2. Install Dependencies:

```bash
pip install -r requirements.txt
```

3. Data Preparation:

- Run data preprocessing to prepare the data for analysis.

4. Run the Notebook:

- Open ```notebooks/Un_supervised_Fraud_detection``` to see the full workflow, including data exploration, model training, and evaluation.

5. Execute Python Scripts:

- Run the individual scripts in the ```src/``` directory for each part of the project (data preprocessing, model training, and evaluation).

## Data Preprocessing
To preprocess the data:

```bash
python src/data_preprocessing.py
```

## Model Training and Anomaly Detection
Each of these scripts can be used to train and detect anomalies using different algorithms:

- **Isolation Forest:**
```bash
python src/isolation_forest.py
```

- **One-Class SVM:**
```bash
python src/one_class_svm.py
```

- **K-Means Clustering:**
```bash
python src/k_means_clustering.py
```

## Evaluation
After model training, evaluate and visualize the results with:

```bash
python src/evaluation.py
```

## Results
Each model generates anomaly scores that indicate how likely each data point is to be an anomaly. The evaluation script provides:

- **Visualization:** Graphs of anomaly scores and potential clusters

- **Metrics:** Descriptive statistics of anomaly detection accuracy (where applicable)

- **Summary:** A final report with findings and potential next steps

## Contributors
Pradipta Dutta - Data Scientist
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pradiptadutta63)
