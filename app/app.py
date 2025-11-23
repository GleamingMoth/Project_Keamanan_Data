import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page Config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    h1 {
        color: #2c3e50;
    }
    h2, h3 {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists('models/spam_classifier.joblib'):
        return joblib.load('models/spam_classifier.joblib')
    return None

model = load_model()

# Sidebar
st.sidebar.title("📧 Spam Detector")
st.sidebar.info("This app uses Machine Learning to classify emails as Spam or Ham (Not Spam).")
st.sidebar.markdown("---")
st.sidebar.write("Developed for Data Security Project")

# Main Content
st.title("Email Spam Classification System")

tabs = st.tabs(["🔍 Prediction", "📊 Model Performance", "📈 EDA"])

# Tab 1: Prediction
with tabs[0]:
    st.header("Analyze Email")
    st.write("Enter the email content below to check if it's spam.")
    
    email_text = st.text_area("Email Content", height=200, placeholder="Type or paste email here...")
    
    if st.button("Classify Email"):
        if email_text:
            if model:
                prediction = model.predict([email_text])[0]
                probability = model.predict_proba([email_text])[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1: # Assuming 1 is Spam
                        st.error("🚨 **SPAM DETECTED**")
                        st.image("https://img.icons8.com/fluency/96/000000/spam.png", width=100)
                    else:
                        st.success("✅ **HAM (SAFE)**")
                        st.image("https://img.icons8.com/fluency/96/000000/verified-account.png", width=100)
                
                with col2:
                    st.write("### Confidence Score")
                    st.progress(float(probability[prediction]))
                    st.write(f"Probability: **{probability[prediction]*100:.2f}%**")
            else:
                st.error("Model not found! Please train the model first.")
        else:
            st.warning("Please enter some text to analyze.")

# Tab 2: Model Performance
with tabs[1]:
    st.header("Model Evaluation Metrics")
    
    if os.path.exists('data/model_evaluation_results.csv'):
        results_df = pd.read_csv('data/model_evaluation_results.csv')
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        st.subheader("Visualizations")
        model_names = results_df['Model'].unique()
        selected_model = st.selectbox("Select Model to View Plots", model_names)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Confusion Matrix")
            cm_path = f'data/plots/confusion_matrix_{selected_model.replace(" ", "_")}.png'
            if os.path.exists(cm_path):
                st.image(cm_path)
            else:
                st.write("Plot not available.")
                
        with col2:
            st.write("#### ROC Curve")
            roc_path = f'data/plots/roc_curve_{selected_model.replace(" ", "_")}.png'
            if os.path.exists(roc_path):
                st.image(roc_path)
            else:
                st.write("Plot not available.")

        st.markdown("---")
        st.subheader("Error Analysis (False Positives & Negatives)")
        
        error_path = f'data/error_analysis_{selected_model.replace(" ", "_")}.csv'
        if os.path.exists(error_path):
            error_df = pd.read_csv(error_path)
            
            col_err1, col_err2, col_err3 = st.columns(3)
            col_err1.metric("Total Errors", len(error_df))
            col_err2.metric("False Positives", len(error_df[error_df['Error_Type'] == 'False Positive']))
            col_err3.metric("False Negatives", len(error_df[error_df['Error_Type'] == 'False Negative']))
            
            st.write("### Misclassified Emails")
            st.dataframe(error_df, use_container_width=True)
            
            st.download_button(
                label="Download Error Analysis CSV",
                data=error_df.to_csv(index=False).encode('utf-8'),
                file_name=f'error_analysis_{selected_model.replace(" ", "_")}.csv',
                mime='text/csv',
            )
        else:
            st.info(f"No error analysis file found for {selected_model}. (Perfect score or file missing)")
    else:
        st.warning("No evaluation results found. Please run the training script.")

# Tab 3: EDA
with tabs[2]:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Class Distribution")
        if os.path.exists('data/plots/class_distribution.png'):
            st.image('data/plots/class_distribution.png')
            
    with col2:
        st.write("#### Text Length Distribution")
        if os.path.exists('data/plots/text_length_distribution.png'):
            st.image('data/plots/text_length_distribution.png')
            
    st.write("#### Word Clouds")
    col3, col4 = st.columns(2)
    with col3:
        st.write("Spam")
        if os.path.exists('data/plots/wordcloud_spam.png'):
            st.image('data/plots/wordcloud_spam.png')
    with col4:
        st.write("Ham")
        if os.path.exists('data/plots/wordcloud_ham.png'):
            st.image('data/plots/wordcloud_ham.png')

