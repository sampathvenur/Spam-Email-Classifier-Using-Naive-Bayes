# Spam Email Classifier Using Naive Bayes

This project implements a **Spam Email Classifier** using the **Naive Bayes algorithm**. It uses the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique for feature extraction from text data (email subjects). The model predicts whether an email is **spam** or **not spam** based on the subject provided.

The backend of the project is built using **Flask**, a lightweight Python web framework, and is designed to allow users to interact with the model via a simple web interface. This README will guide you through the process of setting up, running, and using the spam email classifier.

## Features

- **Spam Detection**: Classifies emails as either "spam" or "not spam" based on the subject text.
- **User-Friendly Interface**: Users can enter email subjects and get immediate predictions on whether the email is spam.
- **Model Training**: Uses a **Naive Bayes classifier** trained on a dataset of labeled emails.
- **Flask Web Application**: A simple web interface to interact with the model.

## Requirements

- Python 3.x
- Flask
- scikit-learn
- pandas
- joblib
- HTML, CSS (for frontend)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Set up a virtual environment

Create a virtual environment to install the dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

You can manually create the `requirements.txt` file with the following contents:

```
Flask==2.0.1
scikit-learn==0.24.2
pandas==1.2.3
joblib==1.0.1
```

### 4. Train the Model (Optional)

If you wish to train the model from scratch, run the following command:

```bash
python train_model.py
```

This will:

- Load the **spam.csv** dataset.
- Preprocess the data (convert labels to numeric format).
- Split the data into training and testing sets.
- Apply **TF-IDF Vectorization**.
- Train a **Naive Bayes model**.
- Save the trained model as **spam_classifier.pkl** using **joblib**.

### 5. Run the Flask Application

Once the dependencies are installed, and the model is ready, you can run the Flask web application:

```bash
python app.py
```

The application will start on `http://127.0.0.1:5000/`. You can open this URL in your web browser.

## Usage

1. **Enter Email Subject**: On the homepage (`http://127.0.0.1:5000/`), you will find a form where you can enter an email subject.
2. **Click Submit**: After typing the subject, click the "Check" button to submit the form.
3. **View Prediction**: The application will show whether the email is **spam** or **not spam** based on the subject.

Example:

- Email Subject: "Hello Sir, Congrats on Winning the Lottery"
  - Prediction: **Spam**
  
- Email Subject: "Important Information Regarding Your Account"
  - Prediction: **Not Spam**

## Project Structure

Here’s the structure of the project:

```
spam-email-classifier/
├── app.py                    # Flask app for interacting with the model
├── train_model.py            # Script for training the model
├── spam.csv                  # Dataset containing labeled spam and non-spam emails
├── spam_classifier.pkl       # Trained model
├── templates/
│   └── index.html            # HTML file for the web interface
└── static/
    ├── css/
    │   └── style.css         # Stylesheet for the frontend
```

- `app.py`: This file contains the Flask web application, which serves the web interface and handles requests for predictions.
- `train_model.py`: This script trains the Naive Bayes model using the `spam.csv` dataset and saves the model to `spam_classifier.pkl`.
- `spam.csv`: A CSV file containing labeled email subjects. You can replace it with your own dataset.
- `spam_classifier.pkl`: The trained machine learning model, which is used in `app.py` for predictions.
- `templates/index.html`: HTML page for the user interface.
- `static/css/style.css`: CSS file for styling the page.

## How It Works

1. **Model Training**:
   - The model is trained using the **Naive Bayes** algorithm, which is a simple probabilistic classifier. It uses the frequency of words in the subject lines to predict whether an email is spam or not.
   - The dataset is preprocessed using **TF-IDF Vectorization** to convert text data into numerical features.
   - After training, the model is saved using **joblib** so that it can be used later without retraining.

2. **Prediction**:
   - When a user inputs a subject in the web interface, the subject is transformed into the same vectorized form as the training data.
   - The trained Naive Bayes model is then used to predict whether the email is spam or not.
   - The result is displayed on the web page.

## Contributions

Feel free to contribute to this project. You can:

- Open issues to report bugs or request features.
- Fork the repository and submit pull requests with improvements or bug fixes.

## Acknowledgements

- **Naive Bayes** algorithm and **TF-IDF** vectorization for spam detection.
- **Flask** for the web application framework.
- **Scikit-learn**, **pandas**, and **joblib** for machine learning and model serialization.
- Dataset source: [SpamAssassin Public Corpus](https://spamassassin.apache.org/).

---
For any issues, please feel free to raise them in the Issues section of the repository.
