
'''
from flask import Flask, render_template, request, url_for, redirect, session
import numpy as np
import re
import os
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import load_model

# Configuration for image folder
IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model
    # Load the pre-trained Keras model
    model = load_model("optimized_lstm_sentiment_model.h5")

###### CODE FOR SENTIMENT ANALYSIS ######

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("new_home.html")    

@app.route('/sentiment_analysis_prediction', methods=['POST'])
def sent_analy_prediction():
    if request.method == 'POST':        
        text = request.form['text']
        sentiment = ''
        probability = 0.0
        img_filename = ''

        # Parameters
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        
        # Preprocess the input text
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())
        words = text.split()
        X_test = [[word_to_id.get(word, 0) for word in words]]
        X_test = pad_sequences(X_test, maxlen=max_review_length)

        # Predict the sentiment
        probability = model.predict(X_test)[0][0]
        sentiment = 'Positive' if probability >= 0.5 else 'Negative'
        img_filename = 'Smiling_Emoji.jpg' if sentiment == 'Positive' else 'Sad_Emoji.jpeg'

    return render_template(
        "new_home.html", 
        text=text, 
        sentiment=sentiment, 
        probability=probability, 
        image=url_for('static', filename='img_pool/' + img_filename)
    )

########### CODE FOR SENTIMENT ANALYSIS ##########

if __name__ == "__main__":
    init()
    app.run()
'''


from flask import Flask, render_template, request, url_for
import numpy as np
import re
import os
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import load_model

# Configuration for image folder
IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


def init():
    global model, word_to_id
    # Load the pre-trained Keras model
    model = load_model("optimized_lstm_sentiment_model.h5")

    # Load the word index from the IMDb dataset
    _, _ = imdb.load_data(num_words=5000)  # Only load to get the word_index
    word_to_id = imdb.get_word_index()

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = ''
    probability = 0.0
    img_filename = ''

    if request.method == 'POST':
        text = request.form['text']
        
        # Parameters
        max_review_length = 500
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        
        # Preprocess the input text
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())
        words = text.split()
        X_test = [[word_to_id.get(word, 0) for word in words]]
        X_test = pad_sequences(X_test, maxlen=max_review_length)

        # Predict the sentiment
        probability = model.predict(X_test)[0][0]
        sentiment = 'Positive' if probability >= 0.5 else 'Negative'
        img_filename = 'Smiling_Emoji.jpg' if sentiment == 'Positive' else 'Sad_Emoji.jpeg'

        return render_template(
            "new_home.html", 
            text=text, 
            sentiment=sentiment, 
            probability=probability, 
            image=url_for('static', filename='img_pool/' + img_filename)
        )
    
    return render_template("new_home.html", sentiment=sentiment, probability=probability, image=url_for('static', filename='img_pool/' + img_filename))


