from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
 # Load the saved vectorizer

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
        
        # Predict whether it's spam or not using the loaded model
        prediction = model.predict(email_tfidf)
        
        # Return the result to the user
        if prediction == 1:
            result = "Spam"
        else:
            result = "Not Spam"
        
        return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)