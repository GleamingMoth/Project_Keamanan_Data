from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf

try:
    print("Attempting to load DistilBERT with use_safetensors=False...")
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, use_safetensors=False)
    print("SUCCESS: Model loaded successfully!")
except Exception as e:
    print(f"FAILURE: {e}")
