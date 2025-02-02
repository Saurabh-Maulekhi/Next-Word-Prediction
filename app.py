from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import time
import os

import kagglehub

# Download latest version
model_folder = kagglehub.model_download("saurabhmaulekhi/next_word_prediction/keras/version-1")

files_in_folder = os.listdir(model_folder)
model_name = files_in_folder[0]

model_path = os.path.join(model_folder,model_name)


model = load_model(model_path)

with open("tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def home_click():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def predict_show():
    if request.method == 'POST':
        try:
            user_input = request.form['user_input']
            num_of_words = int(request.form['num'])
            result = prediction(user_input, num_of_words)
        except ValueError:
            result = "Invalid input. Please enter a number."
    return render_template('home.html', result=result)

@app.route('/document')
def document():
    return render_template('document.html')

def prediction(text, no_of_words):
    sentence = []
    sentence.append(text+" ")
    for i in range(no_of_words):
        # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
        # predict
        pos = np.argmax(model.predict(padded_token_text))  ## predicting new word

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = word
                sentence.append(text)
                sentence.append(" ")

    sentence.pop()
    return sentence

# for i in sentence:
#     print(i, end=" ")
#     time.sleep(0.6)  ## time in seconds to predict new word

if __name__ == "__main__":
    app.run(debug=True)