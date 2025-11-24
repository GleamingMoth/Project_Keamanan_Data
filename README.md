# Email Spam Classification Project

This project aims to classify emails as Spam or Ham (non-spam) using various Machine Learning and Deep Learning algorithms.

## Overview
The project explores multiple approaches for text classification, ranging from traditional machine learning models to advanced deep learning architectures and transformer-based models.

## Dataset
The dataset consists of email texts labeled as either 'spam' or 'ham'.
- **Exploratory Data Analysis (EDA)**: The project includes an EDA script (`src/eda.py`) that analyzes:
    - Class distribution
    - Text length statistics
    - Missing values
    - Word clouds for spam and ham emails

## Models Implemented
We have implemented and evaluated the following models:

### Traditional Machine Learning
1. **Naive Bayes** (MultinomialNB)
2. **Logistic Regression**
3. **Random Forest**
4. **XGBoost**
5. **LightGBM**
6. **Isolation Forest** (Anomaly Detection approach)

### Deep Learning
7. **RNN** (Recurrent Neural Network)
8. **LSTM** (Long Short-Term Memory)
9. **Bidirectional LSTM**
10. **Autoencoder** (Reconstruction-based classification)
11. **DistilBERT** (Fine-tuned Transformer)

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Confusion Matrix
- Feature Importance (for Tree-based models, Linear models, and Isolation Forest)
- False Positive & False Negative Analysis

## Project Structure
```
Project_Keamanan_Data/
├── app/                # Streamlit Web Application
├── data/               # Dataset and saved plots/results
├── models/             # Saved trained models
├── src/                # Source code
│   ├── eda.py          # Exploratory Data Analysis script
│   └── train_model.py  # Model training and evaluation script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation
1. Clone the repository.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Run EDA
```bash
python src/eda.py
```

### Train Models
```bash
python src/train_model.py
```

### Run Web App
```bash
streamlit run app/app.py
```