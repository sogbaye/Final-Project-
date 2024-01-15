import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

with open('en_final_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

News = pd.read_csv('News.csv')

X = News['training_text']

vector = TfidfVectorizer()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return " ".join(lemmas)

def machineresponse(news):

    vector.fit_transform(X)

    # uploadeddata = request.json
    # user_input = uploadeddata.get('newsinput')
    user_input = re.sub("[-,\.!\'?]", '', news).lower()
    preprocessed_data = preprocess_text(user_input)
    vectorized_preprocessed_data = vector.transform([preprocessed_data])
    prediction = loaded_model.predict(vectorized_preprocessed_data)
    if prediction[0] == 0:
        print('The News appears real')
    else:
        print('The News appears fake')

machineresponse("This will install scikit-learn version 1.2.2 in your Python environment. Make sure to restart your Python environment (kernel or interpreter) after the installation to apply the changes.")
