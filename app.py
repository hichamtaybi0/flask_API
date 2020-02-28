from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from flask import Response
from flask import jsonify
import requests
from dragnet import extract_content
import logging
import gensim
import requests
from dragnet import extract_content
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
from textblob import TextBlob
from gensim import models
from categories import category


category = category


def cat_prediction(text):
    # tokenizing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)
    words = X

    # using 50 for padding length
    maxlen = 50
    X = list(sequence.pad_sequences(words, maxlen=maxlen))
    # prepare data
    x = np.array(X)

    pred = category_model.predict(x)
    return pred.argmax(axis=-1)


def text_prep(text):
    # remove \n \t and multiple white space
    text_space = ' '.join(text.split())
    # remove links
    text_link = re.sub(r'http\S+', '', text_space)
    # remove number
    text_number = re.sub(r'\d+', '', text_link)
    # Remove all non words (" ! @ , . ... $ and so on)
    text_words = ' '.join(w for w in re.split(r"\W", text_number) if w)
    # Make all alphabets in lower case
    text_lower = text_words.lower()
    stop_words = set(stopwords.words('english'))
    # Tokenization
    words_tok = word_tokenize(text_lower)
    words_token = [w for w in words_tok if not w in stop_words]
    # Lemmatization
    lemma = WordNetLemmatizer()
    text_lem = [lemma.lemmatize(w, pos="v") for w in words_token]
    # Merge tokens
    text_merge = " ".join(text_lem)

    return text_merge


app = Flask(__name__)

analyser = SentimentIntensityAnalyzer()

# load category classification model
category_model = keras.models.load_model('Trained Models/TextCNN.h5')

#  ##LDA processing
# load LDA model
lda_model = models.LdaModel.load('Trained Models/lda.model06')

# create dictionary for topics
topics = {}

for topic_id in range(lda_model.num_topics):
    topk = lda_model.show_topic(topic_id, 5)
    topk_words = [w for w, _ in topk]
    topics[topic_id] = topk_words
#  ##end of LDA processing


def headline_func(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, "html.parser")
    h1 = soup.find_all('h1')
    return h1[0].text


@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')


@app.route('/results', methods=['POST'])
def results():
    # form = request.form
    if request.method == 'POST':
        # write your function that loads the model
        # model = get_model() #you can use pickle to load the trained model
        # model = pickle.load(open('model.pkl', 'rb'))

        # Extract the content
        url = request.form['url']
        r = requests.get(url)
        content = extract_content(r.content)
        # text = content.split('\n')[0] + content.split('\n')[1]  ## get the first and second sentence

        # Extract the headline
        headline = headline_func(url)

        # merge the headline and the first sentence
        text = headline + " " + content.split('\n')[0]

        # pass the text into preprocessing function
        preprocessed_text = text_prep(text)

        # #predict gategory
        predicted = cat_prediction(preprocessed_text)[0]
        predicted = category.get(predicted)

        # #Predicting the topics for a document
        doc = preprocessed_text.split()
        doc_vector = lda_model.id2word.doc2bow(doc)
        doc_topics = lda_model[doc_vector]
        sorted_by_prob = sorted(doc_topics, key=lambda tup: tup[1], reverse=True)

        # return render_template('resultsform.html', text=text, predicted_category=predicted)
        # return Response()

        # #Sentiment of the News by Vader
        text_series = pd.Series(preprocessed_text)
        score = text_series.apply(lambda t: analyser.polarity_scores(t)['compound'])
        sentiment_vader = 'positive' if score[0] > 0 else 'negative' if score[0] < 0 else 'neutral'

        # Sentiment by textblob: PatterAnalyzer
        blob = TextBlob(text)
        pol = blob.sentences[0].sentiment.polarity
        print(text)

        return jsonify({'data': {"category": predicted,
                                 "url": url,
                                 "body": content,
                                 "topics": topics.get(sorted_by_prob[0][0]),
                                 "Sentiment_vader": sentiment_vader,
                                 "Sentiment_score_vader": score[0],
                                 "Sentiment_textblob": pol,
                                 "headline": headline}})


if __name__ == '__main__':
    app.run("localhost", "7777", debug=True)
