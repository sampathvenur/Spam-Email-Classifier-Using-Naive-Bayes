import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
file_path = './Data/merged_Spam_emails_file.csv'

try:
    df = pd.read_csv(file_path, encoding='latin-1')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Inspect dataset columns
print("Columns in the dataset:", df.columns)

# Adjust column names based on the dataset
# Update 'subject' and 'label' to match the actual column names in your dataset
df = df.rename(columns={
    'message': 'subject',    # Replace 'message' with the actual column name for the email content
    'category': 'label'      # Replace 'category' with the actual column name for labels
})

# Keep only relevant columns
try:
    df = df[['subject', 'label']]
except KeyError:
    print("Error: Required columns 'subject' and 'label' are missing from the dataset.")
    print("Columns available:", df.columns)
    exit()

# Map labels to numeric values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
if df['label'].isnull().any():
    print("Error: Dataset contains labels other than 'ham' and 'spam'.")
    exit()

# Features and labels
X = df['subject']  # Email subject/content
y = df['label']    # Labels: ham (0) or spam (1)

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
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
model_path = 'spam_classifier2.pkl'
vectorizer_path = 'tfidf_vectorizer2.pkl'

joblib.dump(nb_classifier, model_path)
print(f"Model saved to {model_path}")

joblib.dump(vectorizer, vectorizer_path)
print(f"Vectorizer saved to {vectorizer_path}")