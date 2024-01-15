from flask import Flask, render_template, request, jsonify

# Importing libraries

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


app = Flask(__name__)

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

@app.route('/')
def homepage():
	return render_template('home.html')
	
	
@app.route('/about')
def aboutpage():
	return render_template('about.html')
	
	
@app.route('/predict')
def predictpage():
	return render_template('predict.html')
	

@app.route('/contact')
def contactpage():
	return render_template('contact.html')
	

@app.route('/news')
def newspage():
	return render_template('news.html')
	
	
@app.route('/project')
def projectpage():
	return render_template('project.html')
	

@app.route('/faq')
def faqpage():
	return render_template('faq.html')
	
@app.route('/machineresponse', methods=['POST'])
def machineresponse():
    vector.fit_transform(X)
    uploadeddata = request.json
    user_input = uploadeddata.get('newsinput')
    user_input = re.sub("[-,\.!\'?]", '', user_input).lower()
    preprocessed_data = preprocess_text(user_input)
    vectorized_preprocessed_data = vector.transform([preprocessed_data])
    prediction = loaded_model.predict(vectorized_preprocessed_data)
    if prediction[0] == 0:
        prediction_result = "This is likely a real news"
        message = f'<p style="color: green; font-size: 50px;">{prediction_result}</p>'
        return jsonify({'status': 'success', 'message': message})
        # return jsonify( { 'status': 'success', 'message' : "This is likely a real news" } )
    elif prediction[0] == 1:
        prediction_result = "This is likely a fake news"
        message = f'<p style="color: red; font-size: 50px;">{prediction_result}</p>'
        return jsonify({'status': 'success', 'message': message})
        # return jsonify( { 'status': 'success', 'message' : "This is likely a fake news" } )
    else:
        return jsonify( { 'status': 'error', 'message' : "Unexpected prediction" } )


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8080, debug=True)


#### FLASK_APP=app flask run ####