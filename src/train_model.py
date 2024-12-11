import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('./data/spam_dataset.csv', encoding='latin-1')

# Data preprocessing
df = df[['subject', 'label']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Label encoding

# Splitting the data
X = df['subject']  # Feature: the email subject
y = df['label']    # Target: label (ham or spam)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training the Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Model evaluation
y_pred = nb_classifier.predict(X_test_tfidf)
print("Accuracy:", (y_pred == y_test).mean())
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(nb_classifier, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')