import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load data and preprocess
mails = pd.read_csv('spam.csv', encoding='latin-1')
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
mails.rename(columns={'v1': 'spam', 'v2': 'message'}, inplace=True)
mails['spam'] = mails['spam'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X = mails['message']
y = mails['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'model24.pkl')
joblib.dump(tfidf_vectorizer, 'model24_vectorizer.pkl')

# Print success message
print("Model saved to model24.pkl")