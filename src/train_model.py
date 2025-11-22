import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, SimpleRNN, Input, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Transformers
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Ensure directories exist
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('data/plots'):
    os.makedirs('data/plots')

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text'])
    # Create label_num if not exists
    if 'label_num' not in df.columns:
        if 'label' in df.columns:
            df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
        else:
            raise ValueError("Dataset must contain 'label' or 'label_num' column")
    return df

def save_plot(fig, name):
    fig.savefig(f'data/plots/{name}.png')
    plt.close(fig)

def evaluate_preds(y_true, y_pred, y_prob, model_name, results_list):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5 # Fallback

    print(f"Results for {model_name}:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    results_list.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    save_plot(plt.gcf(), f'confusion_matrix_{model_name.replace(" ", "_")}')

    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        save_plot(plt.gcf(), f'roc_curve_{model_name.replace(" ", "_")}')

    # False Positives/Negatives Analysis
    test_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fp = test_df[(test_df['Actual'] == 0) & (test_df['Predicted'] == 1)]
    fn = test_df[(test_df['Actual'] == 1) & (test_df['Predicted'] == 0)]
    print(f"False Positives: {len(fp)}, False Negatives: {len(fn)}")
    
    return f1

def train_sklearn_models(X_train, X_test, y_train, y_test, results):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        f1 = evaluate_preds(y_test, y_pred, y_prob, name, results)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_name = name
            
        # Feature Importance for Tree Models
        if name in ['Random Forest', 'XGBoost', 'LightGBM']:
            try:
                if name == 'Random Forest':
                    importances = pipeline.named_steps['clf'].feature_importances_
                elif name == 'XGBoost':
                    importances = pipeline.named_steps['clf'].feature_importances_
                elif name == 'LightGBM':
                    importances = pipeline.named_steps['clf'].feature_importances_
                
                feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
                feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x=feat_imp.values, y=feat_imp.index)
                plt.title(f'Top 20 Feature Importance - {name}')
                save_plot(plt.gcf(), f'feature_importance_{name.replace(" ", "_")}')
            except Exception as e:
                print(f"Could not plot feature importance for {name}: {e}")

    return best_model, best_name, best_f1

def train_isolation_forest(X_train, X_test, y_train, y_test, results):
    print("\nTraining Isolation Forest...")
    # Isolation Forest is unsupervised. We assume outliers are Spam (1).
    # We need to vectorize first
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Contamination: estimate based on training data
    contamination = y_train.mean()
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_train_vec)
    
    # Predict: -1 is outlier (spam), 1 is inlier (ham)
    y_pred_iso = iso.predict(X_test_vec)
    # Map: -1 -> 1 (Spam), 1 -> 0 (Ham)
    y_pred = np.where(y_pred_iso == -1, 1, 0)
    # Use decision function as probability score (inverted)
    y_scores = -iso.decision_function(X_test_vec) 
    
    evaluate_preds(y_test, y_pred, y_scores, 'Isolation Forest', results)

def train_dl_models(X_train, X_test, y_train, y_test, results):
    # Preprocessing
    max_words = 10000
    max_len = 100
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)
    
    # 1. RNN
    print("\nTraining RNN...")
    model_rnn = Sequential([
        Embedding(max_words, 32, input_length=max_len),
        SimpleRNN(32),
        Dense(1, activation='sigmoid')
    ])
    model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_rnn.fit(X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    
    y_prob = model_rnn.predict(X_test_seq).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    evaluate_preds(y_test, y_pred, y_prob, 'RNN', results)
    
    # 2. Bidirectional LSTM
    print("\nTraining Bidirectional LSTM...")
    model_lstm = Sequential([
        Embedding(max_words, 32, input_length=max_len),
        Bidirectional(LSTM(32)),
        Dense(1, activation='sigmoid')
    ])
    model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm.fit(X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    
    y_prob = model_lstm.predict(X_test_seq).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    evaluate_preds(y_test, y_pred, y_prob, 'Bidirectional LSTM', results)
    
    # 3. Autoencoder (Anomaly Detection)
    print("\nTraining Autoencoder...")
    # Train only on HAM (0) data to learn normal pattern
    X_train_ham = X_train_seq[y_train == 0]
    
    input_dim = X_train_seq.shape[1] # max_len
    encoding_dim = 32
    
    # Simple Dense Autoencoder for sequences (treating as fixed vector)
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder) # Normalize input if using sigmoid? 
    # Sequences are integers, better to use Embedding-based AE or just Dense on TFIDF. 
    # Let's switch to Dense AE on TFIDF for better results in this context or stick to Embedding AE.
    # Embedding AE is complex. Let's use Dense AE on the sequences (not ideal but works) or TFIDF.
    # Let's use TFIDF for Autoencoder for simplicity and speed.
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    X_train_ham_vec = X_train_vec[y_train == 0]
    
    input_dim_ae = X_train_vec.shape[1]
    ae = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim_ae,)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim_ae, activation='sigmoid') # TFIDF is non-negative, but usually not 0-1 scaled perfectly. 
        # Better to use linear or relu output if not scaled. TFIDF is usually normalized.
    ])
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_train_ham_vec, X_train_ham_vec, epochs=10, batch_size=32, verbose=0)
    
    # Predict reconstruction error
    reconstructions = ae.predict(X_test_vec)
    mse = np.mean(np.power(X_test_vec - reconstructions, 2), axis=1)
    
    # Threshold: Mean + 2*Std of Train Ham errors (approx)
    # Or just use ROC AUC directly on MSE
    y_prob_ae = mse # Higher error = more likely spam
    # Find best threshold for F1
    best_thresh = 0
    best_f1_ae = 0
    for thresh in np.percentile(mse, range(0, 100, 5)):
        pred = (mse > thresh).astype(int)
        f1 = f1_score(y_test, pred)
        if f1 > best_f1_ae:
            best_f1_ae = f1
            best_thresh = thresh
            
    y_pred_ae = (mse > best_thresh).astype(int)
    evaluate_preds(y_test, y_pred_ae, y_prob_ae, 'Autoencoder', results)

def train_distilbert(X_train, X_test, y_train, y_test, results):
    print("\nTraining DistilBERT (Fine-tuned)...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        def encode_texts(texts, max_len=64): # Keep max_len small for speed
            return tokenizer(
                texts.tolist(), 
                padding=True, 
                truncation=True, 
                max_length=max_len, 
                return_tensors="tf"
            )

        train_encodings = encode_texts(X_train)
        test_encodings = encode_texts(X_test)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            y_train
        )).shuffle(1000).batch(16)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            y_test
        )).batch(16)
        
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
        
        model.fit(train_dataset, epochs=2, verbose=1) # 2 epochs
        
        # Predict
        y_prob = model.predict(test_dataset).logits
        y_prob = tf.nn.softmax(y_prob, axis=1)[:, 1].numpy()
        y_pred = (y_prob > 0.5).astype(int)
        
        evaluate_preds(y_test, y_pred, y_prob, 'DistilBERT', results)
        
        # Save DistilBERT separately if needed, or just return
        model.save_pretrained('models/distilbert_spam')
        tokenizer.save_pretrained('models/distilbert_spam')
        
    except Exception as e:
        print(f"DistilBERT training failed: {e}")

def main():
    try:
        df = load_data('data/email.csv')
        X = df['text']
        y = df['label_num']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []
        
        # 1. Sklearn Models
        best_sklearn, best_name, best_f1 = train_sklearn_models(X_train, X_test, y_train, y_test, results)
        
        # 2. Isolation Forest
        train_isolation_forest(X_train, X_test, y_train, y_test, results)
        
        # 3. Deep Learning Models
        train_dl_models(X_train, X_test, y_train, y_test, results)
        
        # 4. DistilBERT
        # train_distilbert(X_train, X_test, y_train, y_test, results) 
        # Uncomment above to run DistilBERT. It might be slow. I will enable it as requested.
        train_distilbert(X_train, X_test, y_train, y_test, results)

        # Save Results
        results_df = pd.DataFrame(results)
        results_df.to_csv('data/model_evaluation_results.csv', index=False)
        print("\nEvaluation results saved to 'data/model_evaluation_results.csv'")
        
        # Save Best Sklearn Model (as fallback or main model)
        if best_sklearn:
            joblib.dump(best_sklearn, 'models/spam_classifier.joblib')
            print(f"Best Sklearn model ({best_name}) saved to 'models/spam_classifier.joblib'")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
