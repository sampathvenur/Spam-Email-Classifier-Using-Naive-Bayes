from flask import Flask, render_template, request
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('model24.pkl')
vectorizer = joblib.load('model24_vectorizer.pkl') 

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the form input
        email_content = request.form['email_content']

        # Transform the input text using the loaded vectorizer
        email_tfidf = vectorizer.transform([email_content]) 

        # Predict using the loaded model
        prediction = model.predict(email_tfidf) 

        # Determine the result (Spam or Not Spam)
        if prediction[0] == 1:
            result = "Spam"
        else:
            result = "Not Spam"

        return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)