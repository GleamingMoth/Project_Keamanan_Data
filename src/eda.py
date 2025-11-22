import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Create plots directory if it doesn't exist
if not os.path.exists('data/plots'):
    os.makedirs('data/plots')

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def perform_eda(df):
    print("=== EDA Report ===")
    
    # 1. Sample Count (Jumlah Data)
    print(f"\nTotal Samples (Jumlah Data): {len(df)}")
    
    # 2. Class Distribution
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    print(df['label'].value_counts(normalize=True))
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution')
    plt.savefig('data/plots/class_distribution.png')
    plt.close()
    
    # 3. Missing Values
    print("\nMissing Values Analysis:")
    missing_values = df.isnull().sum()
    print(missing_values)
    if missing_values.sum() > 0:
        print(f"Total Missing Values: {missing_values.sum()}")
    else:
        print("No missing values found.")
    
    # 4. Text Length Analysis
    df['text_length'] = df['text'].astype(str).apply(len)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', hue='label', kde=True, bins=50)
    plt.title('Text Length Distribution by Class')
    plt.savefig('data/plots/text_length_distribution.png')
    plt.close()
    
    print("\nText Length Statistics:")
    print(df.groupby('label')['text_length'].describe())

    # 5. Word Clouds
    spam_text = ' '.join(df[df['label'] == 'spam']['text'].astype(str))
    ham_text = ' '.join(df[df['label'] == 'ham']['text'].astype(str))
    
    if spam_text:
        wordcloud_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_spam, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Spam')
        plt.savefig('data/plots/wordcloud_spam.png')
        plt.close()
        
    if ham_text:
        wordcloud_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_ham, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Ham')
        plt.savefig('data/plots/wordcloud_ham.png')
        plt.close()

    print("\nEDA Completed. Plots saved in 'data/plots/'.")

if __name__ == "__main__":
    try:
        df = load_data('data/email.csv')
        perform_eda(df)
    except Exception as e:
        print(f"Error: {e}")
