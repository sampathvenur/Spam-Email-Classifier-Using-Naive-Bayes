import collections
import nltk
nltk.download('punkt_tab')
import numpy as np
import pandas as pd
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Load data and preprocess
mails = pd.read_csv('spam.csv', encoding='latin-1')  # Assuming 'spam.csv' is in the same directory
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
mails.rename(columns={'v1': 'spam', 'v2': 'message'}, inplace=True)
mails['spam'] = mails['spam'].map({'ham': False, 'spam': True})

def process_message(message):
    words = message.lower()  # lowercase
    words = word_tokenize(words)  # tokenization
    words = [word for word in words if len(word) > 1]  # non-absurd words
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]  # non-stop words
    words = [PorterStemmer().stem(word) for word in words]  # stemming
    return words

def count_processed_words(data):
    counter = collections.OrderedDict()
    for message in data:
        words = process_message(message)
        for word in set(words):
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    return counter

spam_messages = set(mails[mails['spam'] == True]['message'])
spam_words = count_processed_words(spam_messages)

ham_messages = set(mails[mails['spam'] == False]['message'])
ham_words = count_processed_words(ham_messages)

# Combine word frequencies from spam and ham messages 
all_words = {'spam': spam_words, 'ham': ham_words}

# Save the model (all_words dictionary) as a pickle file
import pickle
with open('model24.pkl', 'wb') as f:
    pickle.dump(all_words, f)

print("Model saved to spam_classifier.pkl")