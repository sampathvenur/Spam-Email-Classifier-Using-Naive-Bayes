import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
def load_data():
    data = pd.read_csv('data/spam_dataset.csv')
    return data

# Text Cleaning function
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text

# Preprocess the dataset
def preprocess_data():
    # Load dataset
    data = load_data()
    
    # Clean text data
    data['subject'] = data['subject'].apply(clean_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    data['subject'] = data['subject'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    
    # Vectorize the text (Convert text to numerical data)
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for efficiency
    X = vectorizer.fit_transform(data['subject']).toarray()
    
    # Labels (spam/ham)
    y = data['label'].map({'spam': 1, 'ham': 0}).values  # Spam -> 1, Ham -> 0
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer
