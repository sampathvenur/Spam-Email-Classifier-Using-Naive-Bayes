from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model = joblib.load('spam_classifier_tuned.pkl')
vectorizer = joblib.load('tfidf_vectorizer_tuned.pkl')

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_content = request.form['email_content']
        
        email_tfidf = vectorizer.transform([email_content])
        
        prediction_prob = model.predict_proba(email_tfidf)[0]
        
        spam_prob = round(prediction_prob[1] * 100, 2)  
        not_spam_prob = round(prediction_prob[0] * 100, 2) 
        
        if prediction_prob[1] > prediction_prob[0]:
            result = "Spam"
        else:
            result = "Not Spam"
        
        return render_template('index.html', prediction_result=result, spam_prob=spam_prob, not_spam_prob=not_spam_prob)

if __name__ == '__main__':
    app.run(debug=True)