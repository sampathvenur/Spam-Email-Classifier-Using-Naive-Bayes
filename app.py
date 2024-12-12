from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('spam_classifier_tuned.pkl')
vectorizer = joblib.load('tfidf_vectorizer_tuned.pkl')

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')  # Ensure your index.html is in the templates folder

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the form input
        email_content = request.form['email_content']
        
        # Transform the input text into the same vectorized form as the training data
        email_tfidf = vectorizer.transform([email_content])
        
        # Predict probabilities for both classes (Spam, Not Spam)
        prediction_prob = model.predict_proba(email_tfidf)[0]
        
        # Get the confidence percentages for each class
        spam_prob = round(prediction_prob[1] * 100, 2)  # Round to 2 decimal places
        not_spam_prob = round(prediction_prob[0] * 100, 2)  # Round to 2 decimal places
        
        # Predict the final label
        if prediction_prob[1] > prediction_prob[0]:
            result = "Spam"
        else:
            result = "Not Spam"
        
        return render_template('index.html', prediction_result=result, spam_prob=spam_prob, not_spam_prob=not_spam_prob)

if __name__ == '__main__':
    app.run(debug=True)
