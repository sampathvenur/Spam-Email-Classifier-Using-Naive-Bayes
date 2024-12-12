import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = './Data/merged_Spam_emails_file.csv'

try:
    df = pd.read_csv(file_path, encoding='latin-1')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Inspect dataset columns
print("Columns in the dataset:", df.columns)

# Adjust column names based on the dataset
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

# Load the pre-trained model and vectorizer
model = joblib.load('spam_classifier_tuned.pkl')
vectorizer = joblib.load('tfidf_vectorizer_tuned.pkl')

# Transform the data using the same vectorizer
X_tfidf = vectorizer.transform(X)

# Predict using the model
y_pred = model.predict(X_tfidf)

# Calculate and print accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report (precision, recall, f1-score)
print("Classification Report:\n", classification_report(y, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)
