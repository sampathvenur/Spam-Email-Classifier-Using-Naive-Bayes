import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Load dataset
file_path = './Data/merged_Spam_emails_file.csv'

try:
    df = pd.read_csv(file_path, encoding='latin-1')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Adjust column names based on the dataset
df = df.rename(columns={
    'message': 'subject',    # Replace 'message' with the actual column name for the email content
    'category': 'label'      # Replace 'category' with the actual column name for labels
})

# Keep only relevant columns
df = df[['subject', 'label']]

# Map labels to numeric values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check if there are any missing values or invalid labels
if df['label'].isnull().any():
    print("Error: Dataset contains labels other than 'ham' and 'spam'.")
    exit()

# Features and labels
X = df['subject']  # Email subject/content
y = df['label']    # Labels: ham (0) or spam (1)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization with bigrams (ngrams of 1 and 2)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=5, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Naive Bayes model
nb_classifier = MultinomialNB(alpha=0.5)  # Using smoothing to avoid overfitting

# Train the model
nb_classifier.fit(X_train_tfidf, y_train)

# Model evaluation
y_pred = nb_classifier.predict(X_test_tfidf)
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV (Optional)
param_grid = {'alpha': [0.1, 0.5, 1, 2, 5]}  # Trying different values for alpha
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Print best parameters from GridSearch
print("Best parameters from GridSearchCV:", grid_search.best_params_)

# Save the model and vectorizer
model_path = 'spam_classifier_tuned.pkl'
vectorizer_path = 'tfidf_vectorizer_tuned.pkl'

joblib.dump(grid_search.best_estimator_, model_path)  # Saving the best model after GridSearch
print(f"Model saved to {model_path}")

joblib.dump(vectorizer, vectorizer_path)
print(f"Vectorizer saved to {vectorizer_path}")

